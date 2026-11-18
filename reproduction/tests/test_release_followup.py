from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from repro import run
from repro.common import envinfo
from repro.common.ledger import RunContext
from repro.common.prereg import load_prereg, require_fixed_prereg
from repro.p24 import phase2_plan, phase3_audit, phase3_deliver, phase5_checkpoint, phase6_e1, phase6_e2, utility_agreement


def test_phase5_registration_retains_reporting_and_launch_rules(monkeypatch):
    monkeypatch.setenv('P24_FORMAL_HOST', 'test-host')
    monkeypatch.setenv('P24_REPO_ROOT', 'test repo')
    monkeypatch.setenv('P24_RUN_ROOT', 'test runs')
    doc = load_prereg(Path(phase6_e1.__file__).with_name('prereg_p24_phase5.yaml'))
    assert doc['A4']['reporting'] == (
        'Report every available scorer/cap/fraction/seed cell, both endpoints, harm/cost UCB, '
        'dev/test F1, D/post, F1-text and fallback; unavailable cells explicit.'
    )
    assert doc['A3b']['prerequisites'].endswith('and optional window is available.')
    assert doc['A7']['pilot'] == 's42 after C2, concurrently with A6.'
    assert doc['A7']['remaining_prerequisites'].startswith('C2 and C4 pass, C3 go, start by')
    assert datetime.fromisoformat(doc['deadlines']['A7_remaining_start']).utcoffset() is not None
    assert doc['A9']['UtilityMLP_retraining'].startswith('Optional only in')
    assert len(doc['deadlines']['optional_window']) == 2


def test_c3_no_go_artifact_records_both_stop_decisions(monkeypatch):
    monkeypatch.setattr(phase5_checkpoint, 'find_run', lambda *args, **kwargs: (Path('parent'), {'run_id': 'sample'}))
    monkeypatch.setattr(phase5_checkpoint, 'read_records', lambda path: [
        {'id': value, 'prediction': [0]} for value in range(1624)])
    monkeypatch.setattr(phase5_checkpoint, 'json_read', lambda path: {'manifest': 'sample', 'development_f1': 0.5})
    monkeypatch.setattr(phase5_checkpoint, 'Population', lambda *args: None)
    monkeypatch.setattr(phase5_checkpoint, 'paired', lambda *args: {'f1_delta': 0., 'ci_low': 0.})
    metrics, receipts = {}, {}
    ctx = SimpleNamespace(set_metrics=metrics.update, save_json=lambda name, data: receipts.update({name: data}))
    phase5_checkpoint.checkpoint_c3(ctx, {'phase5_root': 'runs'})
    assert metrics['go'] == 0
    assert receipts['decision.json'] == metrics
    assert 'including running A7' in metrics['decision']
    assert 'do not start remaining A7' in metrics['decision']


@pytest.mark.parametrize('log,expected', [
    ('3 failed, 4 errors', [{'passed': 0, 'failed': 3, 'errors': 4}]),
    ('\x1b[32m=== 9 passed, 1 error in 0.42s ===\x1b[0m', [{'passed': 9, 'failed': 0, 'errors': 1}]),
    ('4 passed in 1.0s\n1 failed in 0.2s\n239 tests collected in 4.54s',
     [{'passed': 4, 'failed': 0, 'errors': 0}, {'passed': 0, 'failed': 1, 'errors': 0}]),
    ('239 tests collected in 4.54s', []),
])
def test_phase3_reports_pytest_execution_summaries(log, expected):
    assert phase3_deliver.validation_summaries(log) == expected


def test_phase3_findings_follow_the_given_measurements():
    metrics = {
        seed: {'r1': {'d3': {'test': {'n_posts': 10, 'detail': detail, 'f1_regret': 0.25}}},
               'r4': {setting: {'choices': {'d3': {'fallback': int(seed == 43 and setting == 'h3_c10')}}}
                      for setting in phase3_deliver._R4_SETTINGS}}
        for seed, detail in ((42, 1), (43, 0.6), (44, 0.8))
    }
    savings, fallbacks = phase3_deliver.replay_findings(metrics)
    assert '减少 4 次（40.00000%）' in savings
    assert '减少 2 次（20.00000%）' in savings
    assert 'h3_c10：1/3，种子 43' in fallbacks
    assert 'h1_c02：0/3，种子 无' in fallbacks


def test_phase2_plan_shares_its_timezone_aware_deadline(monkeypatch, tmp_path):
    monkeypatch.setenv('P24_FORMAL_HOST', 'test-host')
    monkeypatch.setenv('P24_TMUX_PREFIX', 'test-')
    monkeypatch.setattr(phase2_plan.subprocess, 'check_output', lambda *args, **kwargs: 'test-commit\n')
    root = tmp_path / 'phase1' / 'runs'
    root.mkdir(parents=True)
    rows = []
    for seed in (42, 43, 44):
        for method in ('saver', 'always_vision'):
            run_dir = f'{seed}-{method}'
            (root / run_dir).mkdir()
            (root / run_dir / 'config.json').write_text('{}', encoding='utf-8')
            rows.append({'experiment': 'experiment', 'status': 'SUCCESS', 'seed': seed, 'run_dir': run_dir,
                         'tags': {'method': method, 'stage': 'formal'}})
    (root / 'ledger.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    deadline = datetime.now(timezone(timedelta(hours=8))) + timedelta(days=1)
    result = phase2_plan.build(root.parent, tmp_path / 'phase2', deadline)
    expected = deadline.isoformat()
    assert result['deadline'] == expected
    jobs = [job for lane in result['lanes'] for job in lane if job['experiment'] == 'utility_agreement']
    assert len(jobs) == 3
    assert {json.loads(Path(job['config']).read_text())['deadline'] for job in jobs} == {expected}


def test_agreement_waits_while_the_last_seed_is_running(monkeypatch, tmp_path):
    root = tmp_path / 'runs'
    root.mkdir()
    ledger = root / 'ledger.jsonl'
    rows = []
    for seed in (42, 43, 44):
        target = root / str(seed)
        (target / 'raw').mkdir(parents=True)
        (target / 'diagnostic_ids.json').write_text('{"sha256":"shared"}', encoding='utf-8')
        (target / 'raw/utility_final.jsonl').write_text(
            '{"unit_id":"one","eligible":true,"target":1,"student_target":1}\n', encoding='utf-8')
        rows.append({'experiment': 'phase2_training', 'status': 'SUCCESS' if seed != 44 else 'RUNNING',
                     'seed': seed, 'run_dir': str(seed), 'run_id': str(seed),
                     'tags': {'stage': 'formal', 'router': 'saver', 'teacher': 'ema'}})
    ledger.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    sleeps = []

    def finish_seed(seconds):
        sleeps.append(seconds)
        with ledger.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({**rows[-1], 'status': 'SUCCESS'}) + '\n')

    monkeypatch.setattr(utility_agreement.time, 'sleep', finish_seed)
    events, metrics = [], {}
    ctx = SimpleNamespace(tags={}, prereg={'id': 'P24-phase2-v1'},
                          log_event=lambda **event: events.append(event), set_metrics=metrics.update,
                          save_json=lambda *args: None, write_rows=lambda *args: None)
    utility_agreement.main(ctx, {'source_runs_root': str(root), 'teacher': 'ema',
                                'deadline': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()})
    assert sleeps == [20]
    assert events == [{'kind': 'waiting_for_seeds', 'available': [42, 43], 'teacher': 'ema'}]
    assert metrics['seed_pairs'] == 3


@pytest.mark.parametrize('config,prereg,error', [
    ({}, {'id': 'P24-phase2-v1'}, ValueError),
    ({'deadline': '2030-01-01T00:00:00'}, {'id': 'P24-phase2-v1'}, ValueError),
    ({}, {'id': 'historical-phase2'}, RuntimeError),
])
def test_agreement_only_defaults_legacy_deadline(monkeypatch, config, prereg, error):
    monkeypatch.setattr(utility_agreement, 'source_runs', lambda *args: {})
    ctx = SimpleNamespace(prereg=prereg, tags={})
    with pytest.raises(error, match='[Dd]eadline'):
        utility_agreement.main(ctx, {'source_runs_root': 'unused', 'teacher': 'ema', **config})


def test_cli_records_the_exact_bytes_it_parsed(monkeypatch, tmp_path):
    raw = b'id: sample\r\npaper: P24\r\ngates: []\r\n'
    alternate = raw + b'# same parsed registration\r\n'
    register = tmp_path / 'registration.yaml'
    register.write_bytes(raw)
    config = tmp_path / 'config.yaml'
    config.write_text('{}', encoding='utf-8')
    reads = []
    read_bytes = Path.read_bytes

    def read_and_replace(path):
        contents = read_bytes(path)
        if path == register:
            reads.append(contents)
            if len(reads) == 1:
                path.write_bytes(alternate)
        return contents

    monkeypatch.setattr(Path, 'read_bytes', read_and_replace)
    monkeypatch.setattr(envinfo, 'collect_env', lambda path: {})
    monkeypatch.setattr(RunContext, '_resource_metrics', lambda *args: {})
    monkeypatch.setitem(sys.modules, 'repro.p24.receipt_test', SimpleNamespace(main=lambda ctx, config: None))
    root = tmp_path / 'runs'
    args = ['p24', 'receipt_test', '--config', str(config), '--runs-root', str(root), '--prereg', str(register)]
    assert run.main(args) == 0
    assert run.main(args) == 0
    assert reads == [raw, alternate]
    ledger = [json.loads(line) for line in (root / 'ledger.jsonl').read_text().splitlines()]
    for row, snapshot in zip(ledger, reads, strict=True):
        digest = sha256(snapshot).hexdigest()
        assert row['prereg_sha256'] == digest
        assert json.loads((root / row['run_dir'] / 'preregistration.json').read_text()) == {'id': 'sample', 'sha256': digest}
    with pytest.raises(SystemExit) as exc:
        run.main([*args, '--expect-prereg-sha256', sha256(raw).hexdigest()])
    assert exc.value.code == 2
    assert len((root / 'ledger.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('module', [phase3_audit, phase6_e1, phase6_e2])
def test_fixed_entrypoints_reject_equal_yaml_with_wrong_digest(monkeypatch, tmp_path, module):
    template = tmp_path / 'fixed.yaml'
    raw = b'id: sample\npaper: P24\ngates: []\n'
    template.write_bytes(raw)
    monkeypatch.setattr(module, 'PREREG', template)
    if module is phase6_e2:
        monkeypatch.setattr(module, 'runtime', lambda: None)
        monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: True)
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    prereg = load_prereg(template)
    good_ctx = SimpleNamespace(prereg=prereg, prereg_sha256=sha256(raw).hexdigest())
    assert require_fixed_prereg(good_ctx, template) == prereg
    ctx = SimpleNamespace(prereg=prereg, prereg_sha256=sha256(raw + b'# comment\n').hexdigest())
    with pytest.raises(ValueError, match='exact preregistration'):
        module.main(ctx, {})


def test_epsilon_sources_its_digest_from_the_run_receipt(tmp_path):
    (tmp_path / '_SUCCESS').touch()
    (tmp_path / 'metrics.json').write_text(
        '{"exact_replay":true,"E1a":{"dev":{"pooled":{"se":0.02}}}}', encoding='utf-8')
    actual = tmp_path / 'actual.yaml'
    actual.write_bytes(b'id: actual\n')
    registered = sha256(actual.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Historical E1 runs require'):
        phase6_e1.calculate_epsilon(tmp_path)
    assert phase6_e1.calculate_epsilon(tmp_path, prereg_path=actual)['e1_prereg_sha256'] == registered
    (tmp_path / 'preregistration.json').write_text(json.dumps({'sha256': registered}), encoding='utf-8')
    actual.write_bytes(b'id: other\n')
    assert phase6_e1.calculate_epsilon(tmp_path)['e1_prereg_sha256'] == registered
    with pytest.raises(ValueError, match='does not match the E1 run receipt'):
        phase6_e1.calculate_epsilon(tmp_path, prereg_path=actual)
