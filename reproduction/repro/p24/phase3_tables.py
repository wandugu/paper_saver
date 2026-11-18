"""Numeric phase3 tables; values are read exclusively by repro.make_tables."""

import json
import os
from pathlib import Path

from repro.common.tables import ColumnSpec, RowSpec, TableSpec


TABLES = []
ROOT = Path(os.environ.get('P24_PHASE3_RUNS_ROOT', Path(os.environ.get('P24_RUN_ROOT', 'runs')) / 'p24_phase3'))


def add(name, caption, rows, columns):
    TABLES.append(TableSpec(name, caption, rows, [ColumnSpec(*column) for column in columns]))


def selector(seed):
    return {'paper': 'P24', 'experiment': 'phase3_audit', 'status': 'SUCCESS', 'seed': seed}


choices = ['max_coverage', 'max_dev_f1'] + [f'd{index}' for index in range(7)] + [f'l{index}' for index in range(7)]
for choice in choices:
    prefix = f'r1.{choice}.'
    add(f'p24_phase3_R1_{choice}', f'R1 {choice}; F1 fractions, cold latency from measured endpoint replay; n/a threshold denotes exact all-off.',
        [RowSpec(f'seed{seed}', selector(seed)) for seed in (42, 43, 44)],
        [('tau', prefix + 'threshold', '{:.2f}', 'single'), ('nCalPost', prefix + 'n_calibration_posts', '{:.0f}', 'single'),
         ('CalUCB', prefix + 'ucb', '{:.5f}', 'single'), ('DevF1', prefix + 'dev.f1', '{:.5f}', 'single'),
         ('TestF1', prefix + 'test.f1', '{:.5f}', 'single'), ('TestHarm', prefix + 'test.harm', '{:.5f}', 'single'),
         ('D/post', prefix + 'test.detail', '{:.5f}', 'single'), ('Replay ms', prefix + 'test.latency_ms', '{:.3f}', 'single'),
         ('F1Regret', prefix + 'test.f1_regret', '{:.5f}', 'single'),
         ('LatencyRegret ms', prefix + 'test.latency_regret_ms', '{:.3f}', 'single'),
         ('Fallback', prefix + 'fallback', '{:.0f}', 'single')])

for split in ('dev', 'test'):
    add(f'p24_phase3_R2_{split}', f'R2 {split}: all-off minus independently trained ModernBERT-only; paired whole-post bootstrap 2000, descriptive 95% CI.',
        [RowSpec(f'seed{seed}', selector(seed)) for seed in (42, 43, 44)],
        [('IndependentF1', f'r2.{split}.independent.f1', '{:.5f}', 'single'),
         ('AllOffF1', f'r2.{split}.all_off.f1', '{:.5f}', 'single'),
         ('DeltaF1', f'r2.{split}.all_off_minus_independent.f1_delta', '{:.5f}', 'single'),
         ('CI low', f'r2.{split}.all_off_minus_independent.ci_low', '{:.5f}', 'single'),
         ('CI high', f'r2.{split}.all_off_minus_independent.ci_high', '{:.5f}', 'single'),
         ('PostDisagree', f'r2.{split}.all_off_minus_independent.post_disagreement', '{:.5f}', 'single')])
    for choice in ('max_coverage', 'max_dev_f1', 'd3', 'l5'):
        prefix = f'r2.{split}.deploy.{choice}.'
        add(f'p24_phase3_R2_{split}_{choice}', f'R2 {split} {choice}: deployment minus reference; external harm has no calibrated guarantee.',
            [RowSpec(f'seed{seed}', selector(seed)) for seed in (42, 43, 44)],
            [('DeltaOffF1', prefix + 'versus_all_off.f1_delta', '{:.5f}', 'single'),
             ('DeltaExternalF1', prefix + 'versus_independent.f1_delta', '{:.5f}', 'single'),
             ('OffHarm', prefix + 'versus_all_off.harm_vs_reference', '{:.5f}', 'single'),
             ('ExternalHarm', prefix + 'versus_independent.harm_vs_reference', '{:.5f}', 'single'),
             ('OffDisagree', prefix + 'versus_all_off.post_disagreement', '{:.5f}', 'single'),
             ('ExternalDisagree', prefix + 'versus_independent.post_disagreement', '{:.5f}', 'single')])
    add(f'p24_phase3_R3_{split}', f'R3 {split}: forced-on post gain vs mean gate, zero gain is negative for AUROC; MNRE only.',
        [RowSpec(f'seed{seed}', selector(seed)) for seed in (42, 43, 44)],
        [('PostAUROC', f'r3.{split}.post_auroc', '{:.5f}', 'single'),
         ('UnitAUROC', f'r3.{split}.unit_parent_post_auroc', '{:.5f}', 'single'),
         ('RankAgree', f'r3.{split}.rank_concordance', '{:.5f}', 'single'),
         ('SignAgree', f'r3.{split}.sign_agreement', '{:.5f}', 'single'),
         ('Positive', f'r3.{split}.positive_posts', '{:.0f}', 'single'),
         ('Negative', f'r3.{split}.negative_posts', '{:.0f}', 'single'),
         ('Zero', f'r3.{split}.zero_posts', '{:.0f}', 'single')])

for penalty_count in (1, 3, 5):
    for fraction in (2, 5, 10, 20):
        setting = f'h{penalty_count}_c{fraction:02d}'
        prefix = f'r4.{setting}.'
        add(f'p24_phase3_R4_{setting}', f'R4 {penalty_count} harm proxy penalties / {fraction}% unused training posts. Nominal Bonferroni, d3 primary.',
            [RowSpec(f'seed{seed}', selector(seed)) for seed in (42, 43, 44)],
            [('Nominal', prefix + 'nominal_size', '{:.0f}', 'single'), ('Effective', prefix + 'effective_size', '{:.0f}', 'single'),
             ('nPost', prefix + 'calibration_posts', '{:.0f}', 'single'), ('FeasibleRate', prefix + 'feasible_rate', '{:.5f}', 'single'),
             ('UCB floor', prefix + 'zero_harm_ucb_floor', '{:.5f}', 'single'),
             ('ChosenUCB', prefix + 'choices.d3.ucb', '{:.5f}', 'single'), ('UCBSlack', prefix + 'choices.d3.ucb_slack', '{:.5f}', 'single'),
             ('TestF1', prefix + 'choices.d3.test.f1', '{:.5f}', 'single'), ('D/post', prefix + 'choices.d3.test.detail', '{:.5f}', 'single'),
             ('Replay ms', prefix + 'choices.d3.test.latency_ms', '{:.3f}', 'single'),
             ('Fallback', prefix + 'choices.d3.fallback', '{:.0f}', 'single')])
        add(f'p24_phase3_R4_{setting}_aggregate', f'R4 {penalty_count} proxy penalties / {fraction}%; mean and sample SD across all three seeds; fallback mean is empirical fallback rate.',
            [RowSpec('seeds42/43/44', {'paper': 'P24', 'experiment': 'phase3_audit', 'status': 'SUCCESS', 'seed': [42, 43, 44]})],
            [('UCB', prefix + 'choices.d3.ucb'), ('FeasibleRate', prefix + 'feasible_rate'),
             ('TestF1', prefix + 'choices.d3.test.f1'), ('D/post', prefix + 'choices.d3.test.detail'),
             ('Replay ms', prefix + 'choices.d3.test.latency_ms'), ('FallbackRate', prefix + 'choices.d3.fallback')])

r5 = {'paper': 'P24', 'experiment': 'phase3_verify', 'status': 'SUCCESS'}
for identity in ('historical', 'canonical_post', 'text', 'image'):
    add(f'p24_phase3_R5_posts_{identity}', f'R5 {identity}: distinct posts and pairs per official split.',
        [RowSpec('MNRE', r5)],
        [('TrainPosts', f'dataset.{identity}.train.posts', '{:.0f}', 'single'),
         ('DevPosts', f'dataset.{identity}.dev.posts', '{:.0f}', 'single'),
         ('TestPosts', f'dataset.{identity}.test.posts', '{:.0f}', 'single'),
         ('AllPosts', f'dataset.{identity}.all_posts', '{:.0f}', 'single'),
         ('CrossSplitPosts', f'dataset.{identity}.cross_split_posts', '{:.0f}', 'single'),
         ('CrossSplitPairs', f'dataset.{identity}.cross_split_pairs', '{:.0f}', 'single')])
for seed in (42, 43, 44):
    add(f'p24_phase3_R5_grouping_s{seed}', f'R5 seed{seed}: original fit/dev caps can truncate whole posts; calibration retains whole posts.',
        [RowSpec(f'seed{seed}', r5)],
        [(label, f'grouping.s{seed}.{metric}', '{:.0f}', 'single') for label, metric in (
            ('FitPosts', 'fit_posts'), ('CalPosts', 'calibration_posts'), ('FitPartial', 'fit_partial_posts'),
            ('DevPartial', 'dev_partial_posts'), ('CalPartial', 'calibration_partial_posts'),
            ('FitCalOverlap', 'fit_calibration_overlap'), ('FitTestOverlap', 'fit_test_overlap'),
            ('CalTestOverlap', 'calibration_test_overlap'))])
for router in ('linear', 'utility_mlp', 'saver'):
    add(f'p24_phase3_R5_router_{router}', f'R5 executed {router} configuration; code evidence in learned_router_code_audit.json.',
        [RowSpec(router, r5)], [(label, f'routers.{router}.{metric}', '{:.5f}' if 'weight' in metric else '{:.0f}', 'single')
        for label, metric in (('TotalParams', 'parameters'), ('GateParams', 'gate_parameters'), ('Features', 'input_features'),
                              ('Utility', 'utility_weight'), ('Budget', 'budget_weight'), ('Consistency', 'consistency_weight'))])
add('p24_phase3_R5_latency_components', 'R5 component latency is missing in historical synchronized end-to-end logs; n/a is intentional.',
    [RowSpec('logged timings', r5)], [('LoggedRuns', 'timing.logged_runs', '{:.0f}', 'single'),
        ('StageTimedRuns', 'timing.component_timed_runs', '{:.0f}', 'single'),
        ('Screen64 ms', 'timing.screen64_ms', '{:.3f}', 'single'), ('Detail256 ms', 'timing.detail256_ms', '{:.3f}', 'single'),
        ('Fusion ms', 'timing.fusion_ms', '{:.3f}', 'single')])

if (ROOT / 'ledger.jsonl').exists():
    for entry in map(json.loads, (ROOT / 'ledger.jsonl').read_text().splitlines()):
        if entry['experiment'] != 'phase3_audit' or entry['status'] != 'SUCCESS':
            continue
        metric = json.loads((ROOT / entry['run_dir'] / 'metrics.json').read_text())
        for split in ('dev', 'test'):
            for policy in metric['frontier']:
                prefix = f'frontier.{policy}.'
                add(f'p24_phase3_frontier_s{entry["seed"]}_{split}_{policy}',
                    f'All feasible policies, seed {entry["seed"]} {split}; test frontier descriptive, once locked.',
                    [RowSpec(policy, selector(entry['seed']))],
                    [('tau', prefix + 'threshold', '{:.2f}', 'single'), ('UCB', prefix + 'ucb', '{:.5f}', 'single'),
                     ('F1', prefix + split + '.f1', '{:.5f}', 'single'), ('Harm', prefix + split + '.harm', '{:.5f}', 'single'),
                     ('D/post', prefix + split + '.detail', '{:.5f}', 'single'),
                     ('Replay ms', prefix + split + '.latency_ms', '{:.3f}', 'single'),
                     ('Pareto', prefix + split + '.pareto', '{:.0f}', 'single')])
