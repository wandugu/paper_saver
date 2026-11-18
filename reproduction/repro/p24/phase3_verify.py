"""R5 population grouping, logged latency scope, and executed router audit."""

from collections import defaultdict
from pathlib import Path
import subprocess

from .phase3_replay import fingerprint, json_read, parents, post_id
from .protocol import digest, post_key, read_records


def grouped(records, identity):
    result = defaultdict(list)
    for record in records:
        result[identity(record)].append(record)
    return result


def cross_split(groups):
    collisions = []
    for key, records in groups.items():
        splits = sorted({row['split'] for row in records})
        if len(splits) > 1:
            collisions.append({'post_id': fingerprint(key)[:20], 'splits': splits,
                               'ids': [row['id'] for row in records]})
    return collisions


def canonical(record):
    return fingerprint([record['tokens'], record.get('source_image_ids', [Path(value).name for value in record['images']])])


def line_evidence(path, fragments):
    lines = Path(path).read_text().splitlines()
    return [{'file': str(path), 'line': index + 1, 'code': line.strip()}
            for index, line in enumerate(lines) if any(fragment in line for fragment in fragments)]


def main(ctx, config):
    import torch
    from .phase2_training import Phase2Saver

    torch.set_num_threads(2)
    ctx.tags.update(stage='phase3', study='R5', device='cpu')
    repo = Path(config['source_repo'])
    root1 = Path(config['phase1_root'])
    root2 = Path(config['phase2_root'])
    source = parents(root1, 42)['saver']
    manifest_path = repo / json_read(source / 'config.json')['manifest']
    records = read_records(manifest_path)
    all_metrics = {'dataset': {}, 'grouping': {}, 'routers': {}, 'timing': {}, 'training_updates': 0,
                   'gpu_hours_actual': 0.}
    identities = {'historical': post_key, 'canonical_post': canonical,
                  'text': lambda row: fingerprint(row['tokens']),
                  'image': lambda row: fingerprint(row.get('source_image_ids', [Path(value).name for value in row['images']]))}
    collision_reports = {}
    for name, identity in identities.items():
        groups = grouped(records, identity)
        collisions = cross_split(groups)
        collision_reports[name] = collisions
        all_metrics['dataset'][name] = {'all_posts': len(groups), 'cross_split_posts': len(collisions),
                                        'cross_split_pairs': sum(len(row['ids']) for row in collisions)}
        for split in ('train', 'dev', 'test'):
            population = [row for row in records if row['split'] == split]
            post_groups = grouped(population, identity)
            all_metrics['dataset'][name][split] = {'posts': len(post_groups), 'pairs': len(population),
                'multi_pair_posts': sum(len(rows) > 1 for rows in post_groups.values()),
                'max_pairs_per_post': max(map(len, post_groups.values()))}
    ctx.save_json('official_split_collisions.json', collision_reports)
    by_id = {row['id']: row for row in records}
    split_details = {}
    for seed in (42, 43, 44):
        identity = json_read(parents(root1, seed)['saver'] / 'data_identity.json')
        summary = {}
        keys_by_split = {}
        for split, ids in identity['split_ids'].items():
            selected = [by_id[key] for key in ids]
            keys = {canonical(row) for row in selected}
            keys_by_split[split] = keys
            official_name = 'train' if split in ('fit', 'calibration') else split
            official = [row for row in records if row['split'] == official_name]
            omitted = [row['id'] for row in official if canonical(row) in keys and row['id'] not in set(ids)]
            summary[split + '_posts'] = len(keys)
            summary[split + '_pairs'] = len(ids)
            summary[split + '_partial_posts'] = len({canonical(by_id[key]) for key in omitted})
        for left, right in (('fit', 'calibration'), ('fit', 'dev'), ('fit', 'test'),
                            ('calibration', 'dev'), ('calibration', 'test'), ('dev', 'test')):
            summary[left + '_' + right + '_overlap'] = len(keys_by_split[left] & keys_by_split[right])
        all_metrics['grouping'][f's{seed}'] = summary
        split_details[f's{seed}'] = {'canonical_keys_by_split': {key: sorted(value) for key, value in keys_by_split.items()},
                                   'summary': summary, 'identity_sha256': digest(parents(root1, seed)['saver'] / 'data_identity.json')}
    ctx.save_json('split_grouping_audit.json', split_details)
    code_evidence = {'source_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                     'files': {}, 'routers': {}, 'scope': 'Executed phase 1 and phase 2 MNRE configurations'}
    for name, fragments in {
        'model.py': ['def utility_targets', 'vision_losses.detach()', 'self.gate_mlp =',
                     'statistics = torch.stack', 'interaction = torch.cat', 'gate = self.gate_mlp',
                     'def compute_losses', 'F.binary_cross_entropy', 'budget_loss =',
                     'consistency_loss =', 'loss = task_loss', 'rho: float', 'utility_margin: float'],
        'phase2_training.py': ['class Phase2Saver', 'self.router ==', 'self.gate_mlp =',
                               'targets = utility_targets', 'losses["loss"] =',
                               '+ .05 * losses["budget_loss"]', 'teacher_mode =',
                               'optimizer =', 'parameters()'],
    }.items():
        path = repo / 'repro/p24' / name
        code_evidence['files'][name] = {'sha256': digest(path), 'evidence': line_evidence(path, fragments)}
    formal = [row for row in read_records(root2 / 'ledger.jsonl') if row['experiment'] == 'phase2_training'
              and row['status'] == 'SUCCESS' and row.get('tags', {}).get('stage') == 'formal']
    for router in ('linear', 'utility_mlp', 'saver'):
        model = Phase2Saver(23, router=router)
        total_parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
        gate_parameters = sum(value.numel() for value in model.gate_mlp.parameters())
        matches = [row for row in formal if row['tags']['router'] == router]
        all_metrics['routers'][router] = {'parameters': total_parameters, 'gate_parameters': gate_parameters,
                                          'formal_runs': len(matches), 'utility_weight': .25,
                                          'budget_weight': .05, 'consistency_weight': .05,
                                          'input_features': 4 * model.query_dim + 4}
        code_evidence['routers'][router] = {
            'architecture': str(model.gate_mlp), 'parameters': total_parameters, 'gate_parameters': gate_parameters,
            'features': '[q; best_global; q*best_global; abs(q-best_global); max,mean,std,entropy similarities]',
            'label': 'stop-gradient 1[vision_CE + 0.05 < text_CE], instantaneous for all three E3 routers',
            'loss': 'task_CE + 0.25 * gate_BCE + 0.05 * (mean_gate-0.35)^2 + 0.05 * query_evidence_cosine_loss',
            'teacher_variants': sorted({row['tags']['teacher'] for row in matches}),
            'optimizer': 'AdamW lr=1e-4, default weight_decay=0.01; exact executed call in source evidence',
            'formal_run_ids': [row['run_id'] for row in matches],
            'interpretation': 'utility_mlp is a deepcopy of the SAVER gate: same features, parameter count and objective. linear shares those features and objectives but has one affine layer.'}
    ctx.save_json('learned_router_code_audit.json', code_evidence)
    timing_sources = []
    for root in (root1, root2):
        for row in read_records(root / 'ledger.jsonl'):
            if row['experiment'] not in ('profile_run', 'batch_serving') or row['status'] != 'SUCCESS':
                continue
            directory = root / row['run_dir']
            path = directory / ('raw/latency.jsonl' if row['experiment'] == 'profile_run' else 'raw/batch_timings.jsonl')
            sample = read_records(path)[0]
            timing_sources.append({'run_id': row['run_id'], 'path': str(path), 'sha256': digest(path),
                                   'columns': sorted(sample), 'component_timestamps': False})
    ctx.save_json('latency_log_audit.json', {'sources': timing_sources,
                  'screen64_ms': None, 'detail256_ms': None, 'fusion_ms': None,
                  'reason': 'Historical logs time synchronized end-to-end calls only. No component CUDA events or CPU timestamps exist. FLOPs and call counts cannot identify component latency.',
                  'new_gpu_measurement': False})
    all_metrics['timing'] = {'logged_runs': len(timing_sources), 'component_timed_runs': 0,
                            'screen64_ms': None, 'detail256_ms': None, 'fusion_ms': None}
    ctx.save_json('input_identity.json', {'manifest_path': str(manifest_path), 'manifest_sha256': digest(manifest_path),
                                         'source_commit': code_evidence['source_commit']})
    ctx.set_metrics(all_metrics)
