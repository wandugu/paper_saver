"""Collect compact P24 evidence and invoke the sole numeric table generator."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .protocol import digest, read_records
from .phase3_replay import json_read, post_id


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    if len(encoded) >= 3000000:
        raise ValueError(f'Export JSON exceeds declared size: {path} ({len(encoded)})')
    path.write_bytes(encoded)


def compact_rows(path, grouping):
    fields = ('id', 'gold', 'prediction', 'text_prediction', 'vision_prediction',
              'gate', 'eligible', 'selected', 'active', 'unit_ce_gain')
    rows = read_records(path)
    return [{**{key: row[key] for key in fields if key in row},
             'post_id': grouping[row['id']]} for row in rows]


def build(destination, source_repo, phase3_root, phase4_root=None, extra_ledger_roots=()):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Use a fresh evidence destination; never replace a previous bundle')
    destination.mkdir(parents=True)
    source_repo, phase3_root = Path(source_repo), Path(phase3_root)
    roots = [
        Path(os.environ.get('P24_PHASE1_RUNS_ROOT', source_repo / 'tmp/p24_phase1/runs')),
        Path(os.environ.get('P24_PHASE2_RUNS_ROOT', source_repo / 'tmp/p24_phase2/runs')),
        phase3_root,
    ]
    if phase4_root is not None:
        roots.append(Path(phase4_root))
    roots.extend(Path(root) for root in extra_ledger_roots)
    dataset_root = Path(os.environ.get('DATASET_ROOT', source_repo / 'data'))
    manifest = read_records(dataset_root / 'p24/mnre_remote.jsonl')
    grouping = {row['id']: post_id(row) for row in manifest}
    ledger, source_receipts = [], []
    excluded_historical = {'data_identity.json', 'resource_inventory.json', 'calibration.json',
                           'epochs.json', 'batch_population.json'}
    phase3_extra = ('input_identity.json', 'inference_scope.json', 'METHOD_LOCK.json',
                   'selection_audit.json', 'test_provenance.json', 'timing_endpoints.json',
                   'feasible_frontier.json', 'post_predictions_dev.json', 'post_predictions_test.json',
                   'learned_router_code_audit.json', 'latency_log_audit.json',
                   'official_split_collisions.json', 'split_grouping_audit.json')
    for root in roots:
        for entry in read_records(root / 'ledger.jsonl'):
            if entry['paper'] != 'P24':
                continue
            source = root / entry['run_dir']
            target = destination / 'runs' / entry['run_id']
            target.mkdir(parents=True)
            ledger.append({**entry, 'run_dir': f'runs/{entry["run_id"]}',
                           'source_run_dir': str(source), 'source_ledger': str(root / 'ledger.jsonl')})
            files = []
            for path in sorted(source.rglob('*')):
                if not path.is_file():
                    continue
                files.append({'path': str(path.relative_to(source)), 'bytes': path.stat().st_size,
                              'sha256': digest(path)})
            save_json(target / 'source_artifacts.json', {'source_run_dir': str(source), 'artifacts': files,
                       'omissions': 'No checkpoint, image, tensor cache, raw probability tensor or training trajectory copied; their source SHA256 and byte sizes are retained. Compact replay records are separately labeled.'})
            for path in source.glob('*.json'):
                if path.name in excluded_historical:
                    continue
                if entry['experiment'].startswith('phase3_') or path.name not in phase3_extra:
                    save_json(target / path.name, json_read(path))
            if entry['experiment'] == 'experiment' and entry.get('tags', {}).get('method') == 'saver':
                path = source / 'raw/calibration.jsonl'
                if path.exists():
                    save_json(target / 'replay_calibration.json', compact_rows(path, grouping))
            if entry['experiment'] == 'phase2_audit':
                path = source / 'raw/frozen_test.jsonl'
                if path.exists():
                    save_json(target / 'replay_frozen_test.json', compact_rows(path, grouping))
            if entry['experiment'] == 'phase3_extract':
                for name in ('saver_dev', 'independent_dev', 'calibration_extension'):
                    path = source / 'raw' / (name + '.jsonl')
                    save_json(target / (name + '.json'), compact_rows(path, grouping))
            if entry['experiment'] == 'phase4_extract':
                path = source / 'raw/maxsim_dev.jsonl'
                if path.exists():
                    save_json(target / 'maxsim_dev.json', compact_rows(path, grouping))
            source_receipts.append({'run_id': entry['run_id'], 'source_ledger_sha256': digest(root / 'ledger.jsonl'),
                                    'source_run_dir': str(source)})
    if len({entry['run_id'] for entry in ledger}) != len(ledger):
        raise ValueError('Run IDs collide across source ledgers')
    (destination / 'ledger.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in ledger))
    save_json(destination / 'provenance' / 'source_ledgers.json', source_receipts)
    code = destination / 'provenance' / 'code'
    code.mkdir(parents=True)
    for name in ('prereg_p24.yaml', 'prereg_p24_phase2.yaml', 'prereg_p24_phase3.yaml', 'prereg_p24_phase4.yaml',
                 'phase3_extract.py', 'phase3_replay.py', 'phase3_audit.py', 'phase3_verify.py',
                 'phase3_tables.py', 'phase3_export.py', 'phase3_deliver.py', 'phase4_extract.py', 'phase4_replay.py',
                 'phase4_audit.py', 'phase4_tables.py', 'tables.py', 'resources.lock.json'):
        shutil.copyfile(Path('repro/p24') / name, code / name)
    template_hashes = {path.name: digest(path) for path in sorted(code.glob('prereg_p24*.yaml'))}
    save_json(destination / 'provenance' / 'code_revision.json', {
        'worktree': str(Path.cwd()), 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'status': subprocess.check_output(['git', 'status', '--short'], text=True).strip(),
        'historical_registration_commits': {'phase3': '6da1b3492c8aed718869af53ff942f6f963ab749',
                                            'phase4': '709e8e8b08ac02038e0ec46c9382541bbddba4b2'},
        'historical_original_head': 'ddcd96373319c5386392fa2660dd6997eaa07332',
        'copied_public_template_sha256': template_hashes,
        'run_status_counts': dict(Counter(entry['status'] for entry in ledger)),
        'phase3_gpu_hours': 0., 'phase3_training_updates': 0})
    environment = {**os.environ, 'P24_PHASE3_RUNS_ROOT': str(destination), 'CUDA_VISIBLE_DEVICES': ''}
    for spec in ('repro/p24/tables.py', 'repro/p24/phase3_tables.py', 'repro/p24/phase4_tables.py'):
        subprocess.run([sys.executable, '-m', 'repro.make_tables', '--runs-root', str(destination),
                        '--spec', spec, '--out-dir', str(destination / 'tables')], check=True, env=environment)
    files = [{'path': str(path.relative_to(destination)), 'bytes': path.stat().st_size, 'sha256': digest(path)}
             for path in sorted(destination.rglob('*')) if path.is_file()]
    total = sum(row['bytes'] for row in files)
    save_json(destination / 'MANIFEST.json', {'created_at': datetime.now(timezone.utc).isoformat(),
              'files': files, 'total_bytes_excluding_manifest': total,
              'manifest_self_excluded': True, 'ledger_runs': len(ledger),
              'scope': 'P24 phase1, phase2, phase3 and phase4 remote ledgers. Source paths retain host provenance. Tables generated only by python -m repro.make_tables.'})
    total = sum(path.stat().st_size for path in destination.rglob('*') if path.is_file())
    if total > 30000000:
        raise ValueError(f'Evidence exceeds 30MB: {total}')
    print(json.dumps({'destination': str(destination), 'runs': len(ledger), 'bytes': total}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--source-repo', default=os.environ.get('P24_REPO_ROOT', '.'))
    parser.add_argument('--phase3-root', default=Path(os.environ.get('P24_RUN_ROOT', 'runs')) / 'p24_phase3')
    parser.add_argument('--phase4-root', default=Path(os.environ.get('P24_RUN_ROOT', 'runs')) / 'p24_phase4_runs')
    parser.add_argument('--extra-ledger-root', action='append', default=[])
    args = parser.parse_args()
    build(args.destination, args.source_repo, args.phase3_root, args.phase4_root, args.extra_ledger_root)


if __name__ == '__main__':
    main()
