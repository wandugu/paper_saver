"""Locked R1/R4 CPU selection followed by once-only test replay and R2/R3 audits."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..common.prereg import require_fixed_prereg
from .phase3_replay import (Population, add_regret, agreement, calibrate, calibration_populations,
                            fingerprint, json_read, paired, parents, pareto_flags, policy_family, select)
from .protocol import digest, read_records
from .calibration import cp_upper


DISTINCT_LAMBDAS = (0., .01, .05, .1, .25, .5, 1.)
LATENCY_LAMBDAS = (0., .0001, .0005, .001, .005, .01, .05)
PENALTIES = ((0.,), (0., .5, 1.), (0., .25, .5, 1., 2.))
PREREG = Path(__file__).with_name('prereg_p24_phase3.yaml')


def timing_endpoints(phase2_root, seed):
    root = Path(phase2_root)
    matches = [row for row in read_records(root / 'ledger.jsonl')
               if row['experiment'] == 'batch_serving' and row['seed'] == seed and row['status'] == 'SUCCESS']
    if len(matches) != 1:
        raise ValueError('Whole-post measured timing source missing or ambiguous')
    path = root / matches[0]['run_dir'] / 'raw/batch_timings.jsonl'
    rows = [row for row in read_records(path) if row['status'] == 'OK'
            and row['batch_size'] == 1 and not row['bucketed']]
    groups = defaultdict(list)
    for row in rows:
        groups[row['method']].append(row)
    off, on = groups['saver_k1'], groups['always_top1']
    if any(row['detail_image_encodes'] != 0 for row in off):
        raise ValueError('Existing SAVER timing endpoint is not all-off')
    if any(row['detail_image_encodes'] != row['posts'] for row in on):
        raise ValueError('Existing always-vision endpoint is not one detail image per post')
    if {tuple(row['post_ids']) for row in off} != {tuple(row['post_ids']) for row in on}:
        raise ValueError('Timing endpoints use different whole posts')
    return {'off_ms': float(np.mean([row['latency_ms'] for row in off])),
            'on_ms': float(np.mean([row['latency_ms'] for row in on])),
            'off_p90_ms': float(np.quantile([row['latency_ms'] for row in off], .9)),
            'on_p90_ms': float(np.quantile([row['latency_ms'] for row in on], .9)),
            'measured_posts': len({tuple(row['post_ids']) for row in off}), 'samples_each': len(off),
            'source_run_id': matches[0]['run_id'], 'source_sha256': digest(path),
            'scope': 'T4 GPU0 historical phase2 batch1, cold visual, same posts, three passes; no new GPU access',
            'limitation': 'Matched-architecture distinct checkpoints, not same-checkpoint direct policy latency. Population replay linearly mixes measured endpoint means. P90 replay is an endpoint-mixture proxy, not a measured policy P90.',
            'direct_policy_latency_available': False, 'stage_timestamps_available': False}


def selectors():
    return [('max_coverage', 'max_coverage', 0.), ('max_dev_f1', 'max_dev_f1', 0.)] + [
        (f'd{index}', 'value_distinct', value) for index, value in enumerate(DISTINCT_LAMBDAS)] + [
        (f'l{index}', 'value_latency', value) for index, value in enumerate(LATENCY_LAMBDAS)]


def main(ctx, config):
    require_fixed_prereg(ctx, PREREG)
    ctx.tags.update(stage='phase3', study='R1_R2_R3_R4', method='saver', device='cpu')
    source_parents = parents(config['phase1_root'], ctx.seed)
    parent = source_parents['saver']
    identity = json_read(parent / 'data_identity.json')
    manifest_path = Path(config['source_repo']) / json_read(parent / 'config.json')['manifest']
    manifest = read_records(manifest_path)
    extracted = Path(config['extraction_run'])
    source_identity = json_read(extracted / 'input_identity.json')
    if source_identity['checkpoint_sha256']['saver'] != digest(parent / 'best.pt'):
        raise ValueError('Frozen extraction checkpoint identity mismatch')
    timings = timing_endpoints(config['phase2_root'], ctx.seed)
    ctx.save_json('timing_endpoints.json', timings)
    dev_rows = read_records(extracted / 'raw/saver_dev.jsonl')
    expected_dev = {row['id'] for row in manifest if row['split'] == 'dev'}
    if {row['id'] for row in dev_rows} != expected_dev:
        raise ValueError('Full official development population missing')
    development = Population(dev_rows, manifest, timings)
    original_cal = read_records(parent / 'raw/calibration.jsonl')
    extension = read_records(extracted / 'raw/calibration_extension.jsonl')
    calibration_index = {row['id']: row for row in original_cal + extension}
    populations = calibration_populations(manifest, identity, ctx.seed)
    curves, locks = {}, {}
    for penalties in PENALTIES:
        for fraction in (.02, .05, .1, .2):
            setting = f'h{len(penalties)}_c{round(fraction * 100):02d}'
            records = populations[fraction]
            calibration = Population([calibration_index[row['id']] for row in records], manifest, timings)
            curve = calibrate(calibration, policy_family(penalties))
            curves[setting] = (curve, calibration)
            chosen = selectors() if setting == 'h1_c10' else [
                ('max_coverage', 'max_coverage', 0.), ('max_dev_f1', 'max_dev_f1', 0.),
                ('d3', 'value_distinct', .1), ('l5', 'value_latency', .01)]
            locks[setting] = {}
            for name, selector, penalty in chosen:
                policy, fallback = select(curve, development, selector, penalty)
                locks[setting][name] = {'policy': {key: policy[key] for key in ('id', 'eta', 'threshold')},
                                         'selector': selector, 'lambda': penalty, 'fallback': fallback,
                                         'calibration': policy, 'dev': development.evaluate(policy)}
            ctx.log_event(kind='cpu_calibration_complete', setting=setting, posts=len(calibration.post_ids))
    ctx.save_json('METHOD_LOCK.json', {'selections': locks, 'test_labels_opened': False,
                  'all_threshold_test_frontier_locked': True, 'prereg_sha256': ctx.prereg_sha256,
                  'calibration_source_sha256': digest(parent / 'raw/calibration.jsonl'),
                  'dev_source_sha256': digest(extracted / 'raw/saver_dev.jsonl'),
                  'test_protocol': 'One evaluation per distinct frozen action mask; descriptive feasible oracle only, no test-based selection'})
    audit_root = Path(config['phase2_root'])
    sources = [row for row in read_records(audit_root / 'ledger.jsonl') if row['experiment'] == 'phase2_audit'
               and row['seed'] == ctx.seed and row['status'] == 'SUCCESS']
    if len(sources) != 1:
        raise ValueError('Expected one frozen complete phase2 branch extraction')
    test_path = audit_root / sources[0]['run_dir'] / 'raw/frozen_test.jsonl'
    test_rows = read_records(test_path)
    if {row['id'] for row in test_rows} != set(identity['split_ids']['test']):
        raise ValueError('Test population identity mismatch')
    if any(not row['active'][0] for row in test_rows if row['eligible'][0]):
        raise ValueError('Cannot reuse incomplete forced-on test branches')
    test = Population(test_rows, manifest, timings)
    ctx.save_json('test_provenance.json', {'source': str(test_path), 'sha256': digest(test_path),
                  'source_run_id': sources[0]['run_id'], 'new_test_forward_passes': 0,
                  'historical_test_already_evaluated': True, 'method_lock_sha256': digest(ctx.run_dir / 'METHOD_LOCK.json')})
    metrics = {'r1': {}, 'r2': {}, 'r3': {}, 'r4': {}, 'frontier': {}, 'timing': timings}
    main_curve = curves['h1_c10'][0]
    feasible_main = [row for row in main_curve if row['feasible']]
    frontier_rows = []
    for row in main_curve:
        if row['feasible']:
            frontier_rows.append({'policy': {key: row[key] for key in ('id', 'eta', 'threshold')},
                                  'calibration': row, 'dev': development.evaluate(row), 'test': test.evaluate(row)})
    for split in ('dev', 'test'):
        flags = pareto_flags([row[split] for row in frontier_rows])
        for row, flag in zip(frontier_rows, flags):
            row[split]['pareto'] = int(flag)
    for row in frontier_rows:
        metrics['frontier'][row['policy']['id']] = {
            'threshold': row['policy']['threshold'], 'ucb': row['calibration']['ucb'],
            'calibration_harm': row['calibration']['harm'], 'dev': row['dev'], 'test': row['test']}
    for setting, choices in locks.items():
        curve, calibration = curves[setting]
        feasible = [row for row in curve if row['feasible']]
        effective = {np.packbits(calibration.active(row)).tobytes() for row in curve}
        summary = {'nominal_size': len(curve) - 1, 'effective_size': len(effective),
                   'calibration_posts': len(calibration.post_ids), 'calibration_units': len(calibration.rows),
                   'feasible_policies': sum(row['feasible'] for row in curve[:-1]),
                   'feasible_rate': float(np.mean([row['feasible'] for row in curve[:-1]])),
                   'zero_harm_ucb_floor': cp_upper(
                       0, len(calibration.post_ids), 1 - .05 / (len(curve) - 1)), 'choices': {}}
        for name, locked in choices.items():
            policy = locked['policy']
            selected = test.evaluate(policy)
            selected = add_regret(selected, [test.evaluate(row) for row in feasible])
            dev_selected = add_regret(development.evaluate(policy), [development.evaluate(row) for row in feasible])
            summary['choices'][name] = {'threshold': policy['threshold'], 'eta': policy['eta'],
                  'ucb': locked['calibration']['ucb'], 'ucb_slack': locked['calibration']['ucb_slack'],
                  'calibration_harm': locked['calibration']['harm'], 'fallback': int(locked['fallback']),
                  'all_off_chosen': int(selected['coverage'] == 0), 'lambda': locked['lambda'],
                  'dev': dev_selected, 'test': selected}
            if setting == 'h1_c10':
                metrics['r1'][name] = {**summary['choices'][name], 'n_calibration_posts': len(calibration.post_ids)}
        metrics['r4'][setting] = summary
    independent = {'dev': read_records(extracted / 'raw/independent_dev.jsonl'),
                   'test': read_records(source_parents['text_only'] / 'raw/test.jsonl')}
    paired_cache = {}
    for split, population in (('dev', development), ('test', test)):
        independent_index = {row['id']: row for row in independent[split]}
        if set(independent_index) != set(population.ids):
            raise ValueError('Independent text predictions not paired to identical population')
        external = np.array([independent_index[key]['prediction'][0] for key in population.ids])
        if any(independent_index[key]['gold'][0] != label for key, label in zip(population.ids, population.gold)):
            raise ValueError('Text references disagree on gold label identity')
        result = {'independent': population.metrics(external), 'all_off': population.metrics(population.text),
                  'all_off_minus_independent': paired(population, population.text, external, ctx.seed), 'deploy': {}}
        for name, locked in locks['h1_c10'].items():
            active = population.active(locked['policy'])
            prediction = np.where(active, population.vision, population.text)
            signature = (split, prediction.tobytes())
            if signature not in paired_cache:
                paired_cache[signature] = {'versus_all_off': paired(population, prediction, population.text, ctx.seed),
                                          'versus_independent': paired(population, prediction, external, ctx.seed)}
            result['deploy'][name] = paired_cache[signature]
        metrics['r2'][split] = result
        metrics['r3'][split] = agreement(population)
        primary = population.active(locks['h1_c10']['d3']['policy'])
        primary_pred = np.where(primary, population.vision, population.text)
        per_post = []
        for index, key in enumerate(population.post_ids):
            units = np.flatnonzero(population.groups == index)
            per_post.append({'post_id': key, 'ids': [population.ids[item] for item in units],
                             'gold': population.gold[units].tolist(), 'independent': external[units].tolist(),
                             'all_off': population.text[units].tolist(), 'forced_on': population.vision[units].tolist(),
                             'primary_deploy': primary_pred[units].tolist(), 'gate': population.gate[units].tolist()})
        ctx.save_json(f'post_predictions_{split}.json', per_post)
    metrics.update({'training_updates': 0, 'gpu_hours_actual': 0., 'test_action_patterns_scored': len(test.cache),
                    'n_dev_posts': len(development.post_ids), 'n_test_posts': len(test.post_ids),
                    'primary_distinct_saves': int(metrics['r1']['d3']['test']['detail'] < 1),
                    'any_distinct_lambda_saves': int(any(metrics['r1'][f'd{index}']['test']['detail'] < 1
                                                       for index in range(len(DISTINCT_LAMBDAS)))),
                    'latency_rule_is_proxy': True})
    ctx.save_json('feasible_frontier.json', frontier_rows)
    ctx.save_json('selection_audit.json', {'locks_sha256': digest(ctx.run_dir / 'METHOD_LOCK.json'),
                  'test_action_patterns_scored': len(test.cache), 'test_forward_calls': 0,
                  'primary_distinct': 'd3', 'primary_latency': 'l5',
                  'independent_text_test_sha256': digest(source_parents['text_only'] / 'raw/test.jsonl'),
                  'numeric_tables_generator': 'python -m repro.make_tables'})
    ctx.set_metrics(metrics)
