"""Materialize locked phase2 commands without probing or launching GPUs."""

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess

from ..common.prereg import load_prereg_with_sha256
from .phase2_queue import entries


def _write(path, payload):
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError(f'Existing configuration differs; preserve and review: {path}')
    path.write_text(json.dumps(payload, indent=2) + '\n')


def build(phase1_root, root, deadline):
    _, prereg_sha256 = load_prereg_with_sha256('repro/p24/prereg_p24_phase2.yaml')
    phase1_root, root = Path(phase1_root), Path(root)
    queue = root / 'queue'
    queue.mkdir(parents=True, exist_ok=True)
    ledger = entries(phase1_root / 'runs/ledger.jsonl')
    parents = {}
    for seed in (42, 43, 44):
        for method in ('saver', 'always_vision'):
            matches = [row for row in ledger if row['status'] == 'SUCCESS'
                       and row['experiment'] == 'experiment' and row['seed'] == seed
                       and row.get('tags', {}).get('method') == method
                       and row.get('tags', {}).get('stage') == 'formal']
            if len(matches) != 1:
                raise ValueError(f'Expected unique successful phase1 parent {seed}/{method}')
            parents[seed, method] = phase1_root / 'runs' / matches[0]['run_dir']
    lanes = [[], [], []]

    def append(gpu, name, experiment, seed, payload, seconds, **extra):
        path = queue / (name + '.json')
        if payload is not None:
            _write(path, payload)
        lanes[gpu].append({'name': name, 'experiment': experiment, 'seed': seed, 'gpu': gpu,
                           'host': os.environ.get('P24_FORMAL_HOST', 'formal-host'), 'config': str(path), 'max_seconds': seconds, **extra})

    for gpu, seed in enumerate((42, 43, 44)):
        parent = parents[seed, 'saver']
        source = json.loads((parent / 'config.json').read_text())
        append(gpu, f'audit-s{seed}', 'phase2_audit', seed, {'parent_run': str(parent)}, 3600)
        append(gpu, f'e6-s{seed}', 'checkpoint_selection', seed, {'parent_run': str(parent)}, 3600, cpu=True)
        for router, teacher in (('linear', 'instantaneous'), ('utility_mlp', 'instantaneous'),
                                ('saver', 'instantaneous'), ('saver', 'ema'), ('saver', 'frozen')):
            base = {**source, 'router': router, 'teacher': teacher, 'parent_run': str(parent),
                    'checkpoint_selection': 'hard_dev'}
            name = f'{router}-{teacher}-s{seed}'
            append(gpu, 'smoke-' + name, 'phase2_training', seed,
                   {**base, 'pilot': True, 'max_steps': 2, 'pilot_epochs': 1,
                    'pilot_fit_cap': 32, 'pilot_dev_cap': 8, 'pilot_diagnostic_cap': 8}, 600)
            append(gpu, name, 'phase2_training', seed, base, 4 * 3600)
        append(gpu, f'e6-prospective-s{seed}', 'checkpoint_selection', seed, None, 3600,
               parent_selector={'router': 'saver', 'teacher': 'instantaneous'})
    for seed in (42, 43, 44):
        append(0, f'e7-s{seed}', 'batch_serving', seed,
               {'parent_run': str(parents[seed, 'saver']),
                'always_top1_run': str(parents[seed, 'always_vision']), 'sample_posts': 32}, 3600)
    for teacher in ('instantaneous', 'ema', 'frozen'):
        append(0, f'agreement-{teacher}', 'utility_agreement', 0,
               {'source_runs_root': str(root / 'runs'), 'teacher': teacher,
                'deadline': deadline.isoformat()}, 4 * 3600, cpu=True)
    now = datetime.now().astimezone()
    for lane in lanes:
        elapsed = 0
        for job in lane:
            estimate = 30 * 60 if job['experiment'] == 'phase2_training' and not job['name'].startswith('smoke-') else (
                15 * 60 if job['experiment'] == 'batch_serving' else 5 * 60)
            job['estimated_start'] = (now + timedelta(seconds=elapsed)).isoformat()
            elapsed += estimate
            job['estimated_finish'] = (now + timedelta(seconds=elapsed)).isoformat()
            job['estimate_basis'] = 'Operational estimate, 30min/full 10-epoch run versus measured phase1 ~10-20min; dependency waits excluded, not measured results.'
    plan = {'root': str(root), 'phase1_root': str(phase1_root), 'lanes': lanes,
            'created_at': now.isoformat(), 'deadline': deadline.isoformat(),
            'prereg_sha256': prereg_sha256,
            'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'phase1_priority': 'GPU0 waits all 21 profiles; every lane waits its phase1 session exit and <500MiB.'}
    _write(queue / 'phase2_plan.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase1-root', default='tmp/p24_phase1')
    parser.add_argument('--root', default='tmp/p24_phase2')
    parser.add_argument('--deadline', required=True, type=datetime.fromisoformat)
    arguments = parser.parse_args()
    if arguments.deadline.utcoffset() is None or arguments.deadline <= datetime.now().astimezone():
        parser.error('--deadline must be a future ISO 8601 timestamp with a time zone')
    plan = build(arguments.phase1_root, arguments.root, arguments.deadline)
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
