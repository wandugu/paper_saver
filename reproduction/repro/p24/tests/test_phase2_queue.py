import json
from types import SimpleNamespace

import pytest

from repro.p24 import phase2_queue
from repro.p24.phase2_queue import phase1_ready


def test_lane_done_is_insufficient_while_phase1_session_still_exists(tmp_path, monkeypatch):
    monkeypatch.setenv('P24_TMUX_PREFIX', 'test-session-')
    queue = tmp_path / 'queue'
    queue.mkdir()
    (queue / 'lane1.done').touch()
    plan = {'phase1_root': str(tmp_path)}
    assert not phase1_ready(plan, 1, lambda name: name == 'test-session-mre-s43-gpu1')
    assert phase1_ready(plan, 1, lambda name: False)


def test_gpu0_waits_for_all_phase1_profiles_even_after_training_done(tmp_path, monkeypatch):
    monkeypatch.setenv('P24_TMUX_PREFIX', 'test-session-')
    (tmp_path / 'queue').mkdir()
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'queue/lane0.done').touch()
    rows = [{'experiment': 'experiment', 'status': 'SUCCESS', 'run_id': str(index),
             'tags': {'stage': 'formal'}} for index in range(21)]
    ledger = tmp_path / 'runs/ledger.jsonl'
    ledger.write_text('\n'.join(json.dumps(row) for row in rows))
    plan = {'phase1_root': str(tmp_path)}
    assert not phase1_ready(plan, 0, lambda name: False)
    rows += [{'experiment': 'profile_run', 'status': 'SUCCESS',
              'tags': {'parent_run_id': str(index)}} for index in range(21)]
    ledger.write_text('\n'.join(json.dumps(row) for row in rows))
    assert phase1_ready(plan, 0, lambda name: False)


@pytest.mark.parametrize('value', [None, ''])
def test_missing_session_prefix_does_not_release_the_lane(tmp_path, monkeypatch, value):
    (tmp_path / 'queue').mkdir()
    (tmp_path / 'queue/lane1.done').touch()
    if value is None:
        monkeypatch.delenv('P24_TMUX_PREFIX', raising=False)
    else:
        monkeypatch.setenv('P24_TMUX_PREFIX', value)
    with pytest.raises(RuntimeError, match='P24_TMUX_PREFIX'):
        phase1_ready({'phase1_root': str(tmp_path)}, 1, lambda name: pytest.fail('unexpected tmux probe'))


def test_tmux_queries_exact_session_target(monkeypatch):
    def run(args, **kwargs):
        assert args == ['tmux', 'has-session', '-t', '=test-session']
        return SimpleNamespace(returncode=1, stdout='', stderr="can't find session: =test-session\n")

    monkeypatch.setattr(phase2_queue.subprocess, 'run', run)
    assert not phase2_queue._session_exists('test-session')


def test_tmux_confirms_existing_session(monkeypatch):
    monkeypatch.setattr(phase2_queue.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    assert phase2_queue._session_exists('test-session')


@pytest.mark.parametrize('code,diagnostic', [(1, 'no server running'), (1, ''),
                                              (2, "can't find session: =test-mre-s43-gpu1"),
                                              (1, "can't find session: another-session")])
def test_tmux_failure_does_not_release_phase1_dependency(monkeypatch, tmp_path, code, diagnostic):
    monkeypatch.setenv('P24_TMUX_PREFIX', 'test-')
    (tmp_path / 'queue').mkdir()
    (tmp_path / 'queue/lane1.done').touch()
    monkeypatch.setattr(phase2_queue.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=code, stdout='', stderr=diagnostic))
    with pytest.raises(RuntimeError) as caught:
        phase1_ready({'phase1_root': str(tmp_path)}, 1, phase2_queue._session_exists)
    assert f'exit {code}' in str(caught.value)
    assert (diagnostic or 'no diagnostics') in str(caught.value)


def test_missing_tmux_binary_reports_diagnostics(monkeypatch):
    def run(*args, **kwargs):
        raise FileNotFoundError('tmux unavailable')

    monkeypatch.setattr(phase2_queue.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='tmux unavailable'):
        phase2_queue._session_exists('test-session')
