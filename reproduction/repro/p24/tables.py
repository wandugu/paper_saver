"""Generate every displayed number from completed P24 ledger runs only."""

import json
import os
from pathlib import Path

from repro.common.tables import ColumnSpec, RowSpec, TableSpec
from repro.p24.protocol import METHODS


def _prereg_ids():
    path = os.environ.get("P24_PREREG_ID_MAP")
    historical = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    if not isinstance(historical, dict) or any(not isinstance(value, str) or not value for value in historical.values()):
        raise ValueError("P24_PREREG_ID_MAP must map public IDs to nonempty historical IDs")
    return historical


_HISTORICAL_IDS = _prereg_ids()
_PHASE1_ID = "P24-phase1-v1"
_PHASE2_ID = "P24-phase2-v1"

TABLES = [TableSpec(
    id=f'p24_{task}_{section}',
    caption=f'P24 reduced frozen-backbone {task.upper()} {section}; three registered seeds, missing runs remain n/a.',
    rows=[RowSpec(method, {'paper': 'P24', 'experiment': 'experiment' if section == 'main' else 'profile_run', 'status': 'SUCCESS',
                          'prereg_id': _HISTORICAL_IDS.get(_PHASE1_ID, _PHASE1_ID), 'tags.task': task, 'tags.method': method})
          for method in METHODS],
    columns=[ColumnSpec(name, key, fmt, 'mean_sd') for name, key, fmt in columns])
    for task in ('mre', 'mner')
    for section, columns in (
        ('main', [('micro F1', 'micro_f1', '{:.4f}'), ('activation', 'activation_rate', '{:.4f}'),
                  ('detail/sample', 'detail_encodes_per_sample', '{:.3f}')]),
        ('efficiency', [('cold ms', 'cold_latency_ms', '{:.2f}'), ('hot ms', 'warm_latency_ms', '{:.2f}'),
                        ('encoder dense GFLOPs (estimate)', 'encoder_dense_gflops', '{:.3f}'),
                        ('peak MiB', 'peak_gpu_mib', '{:.1f}')]))]

TABLES += [TableSpec(
    id=f'p24_{task}_calibration',
    caption='Frozen finite-grid CP calibration; absent threshold means no feasible nonempty subset, never a relaxed risk bound.',
    rows=[RowSpec(method, {'paper': 'P24', 'experiment': 'experiment', 'status': 'SUCCESS',
                          'prereg_id': _HISTORICAL_IDS.get(_PHASE1_ID, _PHASE1_ID), 'tags.task': task, 'tags.method': method})
          for method in ('saver', 'maxsim', 'mean', 'topk2')],
    columns=[ColumnSpec(name, key, fmt, 'mean_sd') for name, key, fmt in (
        ('threshold', 'calibration_threshold', '{:.3f}'), ('coverage', 'calibration_coverage', '{:.4f}'),
        ('forced vision risk', 'calibration_risk', '{:.4f}'), ('CP upper', 'calibration_upper', '{:.4f}'),
        ('active n', 'calibration_n', '{:.0f}'))]) for task in ('mre', 'mner')]

PHASE2 = {'paper': 'P24', 'prereg_id': _HISTORICAL_IDS.get(_PHASE2_ID, _PHASE2_ID), 'status': 'SUCCESS'}

TABLES += [TableSpec(
    id='p24_phase2_E1',
    caption='E1: independently calibrated seeds; CP for post harm bounds the binary positive-harm envelope. Pair CP bounds forced-vision error, a different estimand.',
    rows=[RowSpec('post structured harm', {**PHASE2, 'experiment': 'phase2_audit'})],
    columns=[ColumnSpec(name, key) for name, key in (
        ('Cal. harm', 'calibration_harm'), ('Post UCB', 'harm_upper'),
        ('Test harm', 'test_post_harm'), ('F1', 'micro_f1'), ('DetailCov', 'detail_coverage'))]),
    TableSpec(
    id='p24_phase2_E1_pair_control',
    caption='Original pair CP control; its UCB concerns selected forced-vision error, not post harm.',
    rows=[RowSpec('original pair CP', {**PHASE2, 'experiment': 'phase2_audit'})],
    columns=[ColumnSpec(name, key) for name, key in (
        ('Cal. post harm', 'pair_calibration_harm'), ('Pair error UCB', 'pair_harm_upper'),
        ('Test post harm', 'pair_test_post_harm'), ('F1', 'pair_micro_f1'), ('DetailCov', 'pair_detail_coverage'))]),
    TableSpec(
    id='p24_phase2_E1_paired', caption='Paired post minus pair policy differences, using each seed own threshold.',
    rows=[RowSpec('post minus pair', {**PHASE2, 'experiment': 'phase2_audit'})],
    columns=[ColumnSpec(name, key) for name, key in (
        ('Delta F1', 'paired_delta_f1'), ('Delta post harm', 'paired_delta_harm'),
        ('Delta DetailCov', 'paired_delta_coverage'))])]

TABLES += [TableSpec(
    id=f'p24_phase2_E6_{policy}',
    caption='E6: missing epoch checkpoints block the comparison; an available soft-selected result remains identifiable.',
    rows=[RowSpec(policy, {**PHASE2, 'experiment': 'checkpoint_selection',
                          'status': 'BLOCKED', 'allow_blocked': True})],
    columns=[ColumnSpec(name, f'{policy}.{key}') for name, key in (
        ('Hard test F1', 'micro_f1'), ('Disagreement', 'policy_disagreement'),
        ('Post harm', 'post_harm'), ('DetailCov', 'detail_coverage'))])
    for policy in ('soft_selected', 'hard_selected', 'last_epoch')]

TABLES += [TableSpec(
    id=f'p24_phase2_E6_prospective_{policy}',
    caption='Prospective E6 on newly trained phase2 instantaneous SAVER checkpoints; distinct from the blocked phase1 comparison.',
    rows=[RowSpec(policy, {**PHASE2, 'experiment': 'checkpoint_selection'})],
    columns=[ColumnSpec(name, f'{policy}.{key}') for name, key in (
        ('Hard test F1', 'micro_f1'), ('Disagreement', 'policy_disagreement'),
        ('Post harm', 'post_harm'), ('DetailCov', 'detail_coverage'))])
    for policy in ('soft_selected', 'hard_selected', 'last_epoch')]

TABLES += [TableSpec(
    id='p24_phase2_E3',
    caption='One hyperparameter configuration per router, 10 complete epochs, identical gate input and utility-target definition; UtilityMLP is the same two-layer architecture as the reduced SAVER gate.',
    rows=[RowSpec(router, {**PHASE2, 'experiment': 'phase2_training', 'tags.stage': 'formal',
                          'tags.router': router, 'tags.teacher': 'instantaneous'})
          for router in ('linear', 'utility_mlp', 'saver')],
    columns=[ColumnSpec(name, key, fmt) for name, key, fmt in (
        ('F1', 'micro_f1', '{:.4f}'), ('Router params', 'router_parameters', '{:.0f}'),
        ('Total params', 'total_parameters', '{:.0f}'), ('GPU h', 'gpu_hours', '{:.3f}'),
        ('Utility AUROC', 'utility_auroc', '{:.4f}'), ('Brier', 'utility_brier', '{:.4f}'))]),
    TableSpec(
    id='p24_phase2_E4',
    caption='Target stability and final student utility diagnostics on a fixed shared dev population; all three registered seeds.',
    rows=[RowSpec(teacher, {**PHASE2, 'experiment': 'phase2_training', 'tags.stage': 'formal',
                           'tags.router': 'saver', 'tags.teacher': teacher})
          for teacher in ('instantaneous', 'ema', 'frozen')],
    columns=[ColumnSpec(name, key) for name, key in (
        ('Epoch flip', 'epoch_flip_rate'), ('Student flip', 'student_epoch_flip_rate'),
        ('Final AUROC', 'utility_auroc'), ('Final Brier', 'utility_brier'), ('F1', 'micro_f1'))]),
    TableSpec(
    id='p24_phase2_E4_agreement',
    caption='Pairwise seed agreement and Cohen kappa on identical diagnostic units; constant-label kappa remains undefined.',
    rows=[RowSpec(teacher, {**PHASE2, 'experiment': 'utility_agreement', 'tags.teacher': teacher})
          for teacher in ('instantaneous', 'ema', 'frozen')],
    columns=[ColumnSpec(name, key) for name, key in (
        ('Seed agreement', 'seed_agreement'), ('Cohen kappa', 'cohen_kappa'),
        ('Student agreement', 'student_seed_agreement'), ('Student kappa', 'student_cohen_kappa'))])]

TABLES += [TableSpec(
    id=f'p24_phase2_E7_{method}_b{batch}_bucket{bucket}',
    caption='Same physical T4, three independent profiling passes per seed; synchronized batch response latency, no visual feature cache.',
    rows=[RowSpec(f'{method} b={batch} bucket={bucket}', {**PHASE2, 'experiment': 'batch_serving'})],
    columns=[ColumnSpec(name, f'settings.{method}_b{batch}_bucket{bucket}.{key}', fmt)
             for name, key, fmt in (
        ('Posts/s', 'posts_per_second', '{:.2f}'), ('P50 ms', 'p50_ms', '{:.2f}'),
        ('P90 ms', 'p90_ms', '{:.2f}'), ('Peak MiB', 'peak_gpu_mib', '{:.1f}'),
        ('Padding waste', 'padding_waste', '{:.4f}'), ('GPU util', 'gpu_utilization', '{:.1f}'))])
    for method in ('always_top1', 'saver_k1') for batch in (1, 8, 16, 32) for bucket in (0, 1)]
