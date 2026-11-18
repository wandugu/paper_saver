"""Frozen-prediction, post-grouped structured-harm calibration."""

import json
import math
from collections import defaultdict
from pathlib import Path

from .calibration import THRESHOLD_GRID, cp_upper
from .protocol import digest, post_key, read_records, task_f1


def read_parent(parent, seed):
    parent = Path(parent).resolve()
    ledger = parent.parent.parent / 'ledger.jsonl'
    entries = [row for row in read_records(ledger)
               if (ledger.parent / row['run_dir']).resolve() == parent]
    if len(entries) != 1 or entries[0]['status'] != 'SUCCESS' or entries[0]['seed'] != seed:
        raise ValueError('Expected exactly one successful same-seed parent ledger entry')
    if (parent / 'SUPERSEDED.json').exists():
        raise ValueError('Superseded parent cannot be used')
    config = json.loads((parent / 'config.json').read_text())
    if config.get('task') != 'mre' or config.get('pilot'):
        raise ValueError('Phase2 post-harm experiment requires a formal MRE parent')
    return parent, config, entries[0]


def validate_population(rows, records, expected_ids):
    identities = [row['id'] for row in rows]
    if len(identities) != len(set(identities)) or set(identities) != set(expected_ids):
        raise ValueError('Frozen prediction IDs differ from the locked split')
    manifest = {record['id']: record for record in records}
    for row in rows:
        if row['id'] not in manifest:
            raise ValueError('Prediction absent from manifest')
        count = len(row['gold'])
        if count != 1 or any(len(row[key]) != count for key in
                             ('gate', 'eligible', 'text_prediction', 'vision_prediction')):
            raise ValueError('Expected one MRE relation pair per manifest row')
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in row['gate']):
            raise ValueError('Gate score must be finite and in [0,1]')
    expected = set(expected_ids)
    keys = {post_key(manifest[identity]) for identity in identities}
    omitted = [record['id'] for record in records
               if post_key(record) in keys and record['id'] not in expected
               and record['split'] == manifest[identities[0]]['split']]
    if omitted:
        raise ValueError('Calibration/test post has missing relation pairs')


def routed_rows(rows, threshold):
    result = []
    for row in rows:
        active = [valid and threshold is not None and score >= threshold
                  for score, valid in zip(row['gate'], row['eligible'])]
        prediction = [visual if enabled else text for text, visual, enabled in
                      zip(row['text_prediction'], row['vision_prediction'], active)]
        result.append({**row, 'prediction': prediction, 'active': active})
    return result


def post_metrics(rows, manifest):
    records = {record['id']: record for record in manifest}
    posts = defaultdict(list)
    for row in rows:
        posts[post_key(records[row['id']])].append(row)
    if not posts:
        raise ValueError('Post audit population cannot be empty')
    details, units, harmful, harms, raw = 0, 0, 0, [], []
    for key, pairs in sorted(posts.items()):
        count = sum(len(row['gold']) for row in pairs)
        text_errors = sum(prediction != label for row in pairs for prediction, label in
                          zip(row['text_prediction'], row['gold']))
        routed_errors = sum(prediction != label for row in pairs for prediction, label in
                            zip(row['prediction'], row['gold']))
        harm = max(0, routed_errors - text_errors) / count
        harms.append(harm)
        harmful += harm > 0
        activated = sum(sum(row['active']) for row in pairs)
        details += activated
        units += count
        raw.append({'post_key': key, 'ids': [row['id'] for row in pairs], 'units': count,
                    'text_errors': text_errors, 'routed_errors': routed_errors,
                    'harm': harm, 'active_units': activated})
    return {'post_harm': sum(harms) / len(harms), 'harmful_posts': harmful,
            'n_posts': len(posts), 'detail_coverage': details / units,
            'micro_f1': task_f1(rows, 'mre'), 'posts': raw}


def calibrate_posts(rows, manifest, alpha=.10, delta=.05):
    if not 0 <= alpha <= 1 or not 0 < delta < 1:
        raise ValueError('Invalid risk/confidence parameters')
    curve, best = [], None
    for threshold in THRESHOLD_GRID:
        metrics = post_metrics(routed_rows(rows, threshold), manifest)
        upper = cp_upper(metrics['harmful_posts'], metrics['n_posts'], 1 - delta / len(THRESHOLD_GRID))
        entry = {key: value for key, value in metrics.items() if key != 'posts'}
        entry.update(threshold=threshold, upper=upper, feasible=upper <= alpha)
        curve.append(entry)
        if entry['feasible'] and (best is None or entry['detail_coverage'] > best['detail_coverage']):
            best = entry
    if best is None:
        metrics = post_metrics(routed_rows(rows, None), manifest)
        best = {key: value for key, value in metrics.items() if key != 'posts'}
        best.update(threshold=None, upper=0., feasible=False, fallback='deterministic_all_off')
    return {**best, 'curve': curve, 'alpha': alpha, 'delta': delta,
            'bound_definition': 'CP on binary positive-net-post-harm envelope; all posts are trials',
            'grid_size': len(THRESHOLD_GRID)}


def frozen_test(ctx, parent, config, checkpoint, threshold, existing=None):
    if existing is not None and all(
            not valid or threshold is None or score < threshold or old_active
            for row in existing for score, valid, old_active in
            zip(row['gate'], row['eligible'], row['active'])):
        ctx.save_json('test_provenance.json', {'source': str(parent / 'raw/test.jsonl'),
                      'sha256': digest(parent / 'raw/test.jsonl'), 'new_forward_passes': 0})
        return existing
    import torch
    from .encoders import FrozenEncoders
    from .experiment import evaluate, gpu_guard, verify_resources
    from .model import SaverCore
    from .protocol import FeatureStore

    ctx.save_json('gpu_initial.json', gpu_guard(False))
    torch.set_num_threads(4)
    identity = json.loads((parent / 'data_identity.json').read_text())
    records = read_records(config['manifest'])
    test_ids = set(identity['split_ids']['test'])
    population = [record for record in records if record['id'] in test_ids]
    with (ctx.run_dir / 'TEST_STARTED').open('x') as stream:
        stream.write('Frozen checkpoint, new policy locked before one forced-branch extraction.\n')
    encoders = FrozenEncoders(config['text_model'], config['vision_model'])
    store = FeatureStore(config['cache'], encoders, identity=verify_resources(config, 'mre'))
    model = SaverCore(len(identity['labels']), text_dim=encoders.text_hidden,
                      vision_dim=encoders.vision_hidden).cuda()
    model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
    rows = evaluate(model, store, population, 'mre', identity['labels'], 'saver', 1000,
                    threshold=math.inf if threshold is None else threshold, seed=ctx.seed,
                    audit=True, test=True, ctx=ctx, raw_name='frozen_test')
    ctx.save_json('test_provenance.json', {'checkpoint_sha256': digest(checkpoint),
                  'new_forward_passes': 1, 'forced_branch_extraction': True})
    return rows


def main(ctx, config):
    parent, source, entry = read_parent(config['parent_run'], ctx.seed)
    ctx.tags.update(task='mre', stage='phase2', study='E1', parent_run_id=entry['run_id'])
    records = read_records(source['manifest'])
    identity = json.loads((parent / 'data_identity.json').read_text())
    if digest(source['manifest']) != identity['sha256']:
        raise ValueError('Manifest no longer matches the parent run')
    calibration = read_records(parent / 'raw/calibration.jsonl')
    validate_population(calibration, records, identity['split_ids']['calibration'])
    result = calibrate_posts(calibration, records)
    pair = json.loads((parent / 'calibration.json').read_text())
    ctx.save_json('calibration_post.json', result)
    ctx.save_json('parent_identity.json', {'run': entry, 'checkpoint_sha256': digest(parent / 'best.pt'),
                  'manifest_sha256': digest(source['manifest']),
                  'calibration_sha256': digest(parent / 'raw/calibration.jsonl')})
    ctx.save_json('METHOD_LOCK.json', {'checkpoint_sha256': digest(parent / 'best.pt'),
                  'post_threshold': result['threshold'], 'pair_threshold': pair['threshold'],
                  'test_evaluations_allowed': 1})
    existing = read_records(parent / 'raw/test.jsonl')
    validate_population(existing, records, identity['split_ids']['test'])
    test = frozen_test(ctx, parent, source, parent / 'best.pt', result['threshold'], existing)
    post_test = routed_rows(test, result['threshold'])
    post = post_metrics(post_test, records)
    pair_cal = post_metrics(routed_rows(calibration, pair['threshold']), records)
    pair_test = post_metrics(existing, records)
    ctx.write_rows('post_policy_test', post_test)
    ctx.write_rows('post_audit', post['posts'])
    ctx.write_rows('calibration_curve', result['curve'])
    ctx.set_metrics({'calibration_harm': result['post_harm'], 'harm_upper': result['upper'],
                     'calibration_detail_coverage': result['detail_coverage'],
                     'calibration_threshold': result['threshold'], 'test_post_harm': post['post_harm'],
                     'micro_f1': post['micro_f1'], 'detail_coverage': post['detail_coverage'],
                     'n_calibration_posts': result['n_posts'], 'n_test_posts': post['n_posts'],
                     'useful_coverage': result['feasible'] and post['detail_coverage'] >= .15,
                     'pair_calibration_harm': pair_cal['post_harm'], 'pair_harm_upper': pair['upper'],
                     'pair_forced_error_risk': pair['risk'], 'pair_test_post_harm': pair_test['post_harm'],
                     'pair_micro_f1': pair_test['micro_f1'], 'pair_detail_coverage': pair_test['detail_coverage'],
                     'paired_delta_f1': post['micro_f1'] - pair_test['micro_f1'],
                     'paired_delta_harm': post['post_harm'] - pair_test['post_harm'],
                     'paired_delta_coverage': post['detail_coverage'] - pair_test['detail_coverage']})
