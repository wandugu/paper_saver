"""Agreement of independently trained seeds on a fixed utility population."""

from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import statistics
import time

from .protocol import digest, read_records


HISTORICAL_DEADLINE = '2026-11-15T23:59:59+08:00'


def agreement(left, right, target='target'):
    populations = [{row['unit_id']: row for row in rows if row['eligible']} for rows in (left, right)]
    if any(len(population) != sum(bool(row['eligible']) for row in rows)
           for population, rows in zip(populations, (left, right))):
        raise ValueError('Duplicate diagnostic identity')
    if not populations[0] or populations[0].keys() != populations[1].keys():
        raise ValueError('Seed diagnostic identities must match exactly')
    ordered = sorted(populations[0])
    values = [[population[identity][target] for identity in ordered] for population in populations]
    if any(value not in (0, 1) for population in values for value in population):
        raise ValueError('Utility targets must be binary')
    observed = sum(left == right for left, right in zip(*values)) / len(ordered)
    positive_left, positive_right = (statistics.fmean(population) for population in values)
    expected = positive_left * positive_right + (1 - positive_left) * (1 - positive_right)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {'agreement': observed, 'kappa': kappa, 'n': len(ordered),
            'positive_left': positive_left, 'positive_right': positive_right,
            'kappa_reason': 'constant_identical_labels_undefined' if kappa is None else None}


def source_runs(root, teacher):
    selected = {}
    for entry in read_records(Path(root) / 'ledger.jsonl'):
        tags = entry.get('tags', {})
        if (entry['experiment'] != 'phase2_training' or entry['status'] != 'SUCCESS'
                or tags.get('stage') != 'formal' or tags.get('router') != 'saver'
                or tags.get('teacher') != teacher):
            continue
        directory = Path(root) / entry['run_dir']
        if (directory / 'SUPERSEDED.json').exists():
            continue
        if entry['seed'] in selected:
            raise ValueError('Multiple formal runs for the same teacher/seed require explicit review')
        selected[entry['seed']] = (entry, directory)
    return selected


def main(ctx, config):
    root, teacher = Path(config['source_runs_root']), config['teacher']
    if 'deadline' in config:
        deadline = datetime.fromisoformat(config['deadline'])
    elif set(config) == {'source_runs_root', 'teacher'} and ctx.prereg and ctx.prereg['id'] != 'P24-phase2-v1':
        deadline = datetime.fromisoformat(HISTORICAL_DEADLINE)
    else:
        raise ValueError('Agreement config requires the phase 2 plan deadline')
    if deadline.utcoffset() is None:
        raise ValueError('Agreement deadline must include a time zone')
    deadline = deadline.astimezone(timezone.utc)
    ctx.tags.update(study='E4_agreement', teacher=teacher, stage='phase2_aggregate')
    while True:
        selected = source_runs(root, teacher)
        if set(selected) == {42, 43, 44}:
            break
        if datetime.now(timezone.utc) >= deadline:
            raise RuntimeError('Deadline before all registered seeds were available')
        ctx.log_event(kind='waiting_for_seeds', available=sorted(selected), teacher=teacher)
        time.sleep(20)
    diagnostics = {}
    sources = []
    hashes = set()
    for seed, (entry, directory) in selected.items():
        identity = json.loads((directory / 'diagnostic_ids.json').read_text())
        hashes.add(identity['sha256'])
        diagnostics[seed] = read_records(directory / 'raw/utility_final.jsonl')
        sources.append({'seed': seed, 'run_id': entry['run_id'],
                        'final_diagnostic_sha256': digest(directory / 'raw/utility_final.jsonl')})
    if len(hashes) != 1:
        raise ValueError('Common diagnostic identity hash differs across seeds')
    ctx.save_json('source_runs.json', {'sources': sources, 'diagnostic_ids_sha256': hashes.pop()})
    metrics = {}
    for target, prefix in (('target', ''), ('student_target', 'student_')):
        pairs = [{**agreement(diagnostics[left], diagnostics[right], target), 'left_seed': left,
                  'right_seed': right, 'target': target} for left, right in combinations(sorted(selected), 2)]
        ctx.write_rows('seed_pairs', pairs)
        kappas = [row['kappa'] for row in pairs if row['kappa'] is not None]
        metrics.update({prefix + 'seed_agreement': statistics.fmean(row['agreement'] for row in pairs),
                        prefix + 'cohen_kappa': statistics.fmean(kappas) if kappas else None,
                        prefix + 'kappa_pairs_defined': len(kappas), prefix + 'seed_pairs': len(pairs)})
    ctx.set_metrics(metrics)
