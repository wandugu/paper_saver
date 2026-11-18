"""Dev-only checkpoint comparison with explicit missing-artifact blocking."""

import json
import math
from pathlib import Path

from .phase2_audit import post_metrics, read_parent
from .protocol import digest, read_records, task_f1


def checkpoint_plan(parent):
    parent = Path(parent)
    path = parent / 'epochs.json'
    if not path.exists():
        return {'complete': False, 'reason': 'No retained epoch index; phase1 saved only best.pt',
                'available_checkpoints': sorted(item.name for item in parent.glob('*.pt'))}
    epochs = json.loads(path.read_text())['epochs']
    if not epochs or any(not (parent / row['checkpoint']).is_file() for row in epochs):
        return {'complete': False, 'reason': 'Epoch index is empty or references missing checkpoints'}
    if any(not math.isfinite(row[metric]) for row in epochs for metric in ('soft_dev_f1', 'hard_dev_f1')):
        raise ValueError('Non-finite development score')
    if len({row['epoch'] for row in epochs}) != len(epochs):
        raise ValueError('Duplicate epoch metadata')
    ordered = sorted(epochs, key=lambda row: row['epoch'])
    return {'complete': True, 'selected': {
        'soft_selected': max(ordered, key=lambda row: row['soft_dev_f1']),
        'hard_selected': max(ordered, key=lambda row: row['hard_dev_f1']),
        'last_epoch': ordered[-1]}}


def policy_disagreement(rows, reference):
    reference_by_id = {row['id']: row for row in reference}
    if {row['id'] for row in rows} != set(reference_by_id) or len(rows) != len(reference):
        raise ValueError('Policies must use the same test identities')
    disagreements, count = 0, 0
    for row in rows:
        other = reference_by_id[row['id']]
        if row['gold'] != other['gold'] or len(row['active']) != len(other['active']):
            raise ValueError('Policy units or gold labels do not agree')
        disagreements += sum(left != right for left, right in zip(row['active'], other['active']))
        count += len(row['active'])
    return disagreements / count if count else 0.


def main(ctx, config):
    parent, source, entry = read_parent(config['parent_run'], ctx.seed)
    ctx.tags.update(task='mre', stage='phase2', study='E6', parent_run_id=entry['run_id'])
    plan = checkpoint_plan(parent)
    ctx.save_json('checkpoint_inventory.json', plan)
    records = read_records(source['manifest'])
    existing = read_records(parent / 'raw/test.jsonl')
    soft = post_metrics(existing, records)
    ctx.set_metrics({'checkpoint_complete': int(plan['complete']),
                     'soft_selected': {'micro_f1': soft['micro_f1'], 'post_harm': soft['post_harm'],
                                       'detail_coverage': soft['detail_coverage'], 'policy_disagreement': 0.}})
    if not plan['complete']:
        ctx.save_json('missing_artifacts.json', {'reason': plan['reason'],
                      'required': ['all evaluated epoch checkpoints', 'soft/hard dev F1 per epoch'],
                      'action': 'No retraining, no substitution, no new test evaluation'})
        return
    import torch
    from .encoders import FrozenEncoders
    from .experiment import _calibrate, evaluate, gpu_guard, verify_resources
    from .model import SaverCore
    from .protocol import FeatureStore

    identity = json.loads((parent / 'data_identity.json').read_text())
    ctx.save_json('METHOD_LOCK.json', {'selected': plan['selected'],
                  'checkpoint_hashes': {name: digest(parent / row['checkpoint'])
                                        for name, row in plan['selected'].items()},
                  'test_evaluations_allowed_per_distinct_configuration': 1})
    outputs = {}
    encoders, store = None, None
    completed = {}
    for name, chosen in plan['selected'].items():
        checkpoint = parent / chosen['checkpoint']
        checkpoint_hash = digest(checkpoint)
        if checkpoint_hash in completed:
            outputs[name] = completed[checkpoint_hash]
            continue
        if checkpoint_hash == digest(parent / 'best.pt'):
            rows = existing
            ctx.save_json(f'{name}_reuse.json', {'source': str(parent / 'raw/test.jsonl'),
                          'sha256': digest(parent / 'raw/test.jsonl')})
        else:
            if encoders is None:
                ctx.save_json('gpu_initial.json', gpu_guard(False))
                torch.set_num_threads(4)
                encoders = FrozenEncoders(source['text_model'], source['vision_model'])
                store = FeatureStore(source['cache'], encoders, identity=verify_resources(source, 'mre'))
            model = SaverCore(len(identity['labels']), text_dim=encoders.text_hidden,
                              vision_dim=encoders.vision_hidden).cuda()
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cuda'))
            calibration_ids = set(identity['split_ids']['calibration'])
            calibration_rows = evaluate(model, store, [record for record in records if record['id'] in calibration_ids],
                                        'mre', identity['labels'], 'saver', 1000, audit=True, seed=ctx.seed)
            calibration = _calibrate(calibration_rows, 'mre', ctx.seed)
            ctx.write_rows(f'{name}_calibration', calibration_rows)
            ctx.save_json(f'{name}_lock.json', {'checkpoint_sha256': checkpoint_hash,
                          'calibration': calibration, 'test_evaluations_allowed': 1})
            with (ctx.run_dir / f'{name}_TEST_STARTED').open('x') as stream:
                stream.write('One test evaluation for this checkpoint and locked policy.\n')
            test_ids = set(identity['split_ids']['test'])
            rows = evaluate(model, store, [record for record in records if record['id'] in test_ids],
                            'mre', identity['labels'], 'saver', 1000, seed=ctx.seed,
                            threshold=math.inf if calibration['threshold'] is None else calibration['threshold'],
                            test=True, ctx=ctx, raw_name=f'{name}_test')
        outputs[name] = rows
        completed[checkpoint_hash] = rows
    reference = outputs['soft_selected']
    for name, rows in outputs.items():
        metrics = post_metrics(rows, records)
        ctx.set_metrics({name: {'micro_f1': task_f1(rows, 'mre'), 'post_harm': metrics['post_harm'],
                               'detail_coverage': metrics['detail_coverage'],
                               'policy_disagreement': policy_disagreement(rows, reference)}})
