import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from repro.common.paths import expand_config_paths
from repro.common.prereg import load_prereg
from repro.p24 import phase6_e1, phase6_e2, resources
from repro.p24.phase5_training import cache_path, excluded_cache_entries


@pytest.mark.parametrize(('script', 'message'), [
    ('import repro; import trace_store', 'blocks module'),
    ('import repro; import locked_inputs', 'blocks module'),
    ('import sys, types; sys.modules["locked_inputs"] = types.ModuleType("locked_inputs"); import repro',
     'already imported'),
    ('import repro; open("trace-store/ledger.json")', 'blocks path'),
    ('import repro; open("derived-values.json")', 'blocks path'),
])
def test_integrity_guard_intercepts_imports_and_open(script, message):
    result = subprocess.run([sys.executable, '-B', '-c', script],
                            env={**os.environ, 'REPRO_BLOCK_MODULES': 'locked_inputs',
                                 'REPRO_BLOCK_PATHS': 'derived-*.json'},
                            cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize('separate_data', [False, True])
def test_exclusions_follow_reader_feature_root(monkeypatch, tmp_path, separate_data):
    code = tmp_path / 'code'
    data = tmp_path / 'separate data' if separate_data else code / 'data'
    monkeypatch.setenv('P24_REPO_ROOT', str(code))
    monkeypatch.setenv('DATASET_ROOT', str(data))
    monkeypatch.setenv('MODEL_ROOT', str(tmp_path / 'models'))
    feature_root = Path(phase6_e2.resources()['cache'])
    assert feature_root == data / 'p24/features-v1'
    entries = excluded_cache_entries()
    assert len(entries) == 4611
    for entry in entries:
        path = feature_root.joinpath(*entry.parts)
        assert cache_path(feature_root, path) == path.with_name('phase5-tf4571-' + path.name)
    clean = feature_root / 'other-namespace/not-excluded.pt'
    assert cache_path(feature_root, clean) == clean
    with pytest.raises(ValueError, match='outside its configured root'):
        cache_path(feature_root, tmp_path / 'wrong-root/namespace/a.pt')
    with pytest.raises(ValueError, match='namespace and file'):
        cache_path(feature_root, feature_root / 'nested/namespace/a.pt')


def test_required_roots_and_nested_prereg_fields(monkeypatch, tmp_path):
    template = tmp_path / 'registration.yaml'
    template.write_text('id: example\npaper: P24\ngates: []\n'
                        'runtime:\n  command: use ${P24_REPO_ROOT}/bin/python\n'
                        '  logs: ${P24_RUN_ROOT}/logs\n', encoding='utf-8')
    monkeypatch.delenv('P24_REPO_ROOT', raising=False)
    with pytest.raises(RuntimeError, match='P24_REPO_ROOT'):
        load_prereg(template)
    monkeypatch.setenv('P24_REPO_ROOT', 'test repo')
    monkeypatch.setenv('P24_RUN_ROOT', 'test runs')
    assert load_prereg(template)['runtime'] == {'command': 'use test repo/bin/python', 'logs': 'test runs/logs'}
    monkeypatch.delenv('DATASET_ROOT', raising=False)
    with pytest.raises(RuntimeError, match='DATASET_ROOT'):
        phase6_e2.resources()
    monkeypatch.setenv('MODEL_ROOT', '')
    with pytest.raises(RuntimeError, match='MODEL_ROOT'):
        expand_config_paths('${MODEL_ROOT}/weights')


@pytest.mark.parametrize(('remote_revision', 'remote_sha', 'message'), [
    ('b' * 40, 'c' * 64, 'Resolved revision differs'),
    ('a' * 40, 'd' * 64, 'Published weights differ'),
])
def test_downloader_rejects_wrong_metadata_before_fetch(monkeypatch, tmp_path, remote_revision, remote_sha, message):
    import requests

    class Session:
        def __init__(self):
            self.proxies = {}

        def mount(self, *_):
            pass

        def get(self, url, **kwargs):
            assert url.endswith('/revision/' + 'a' * 40)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                'sha': remote_revision,
                'siblings': [{'rfilename': 'model.safetensors', 'lfs': {'sha256': remote_sha}}],
            })

    monkeypatch.setattr(requests, 'Session', Session)
    with pytest.raises(ValueError, match=message):
        resources.download_model('owner/model', tmp_path, revision='a' * 40, weights_sha256='c' * 64)


def test_resource_cli_passes_lock_values(monkeypatch, tmp_path):
    calls = []

    def download(repository, directory, **kwargs):
        calls.append((repository, kwargs))
        return {'status': 'downloaded'}

    monkeypatch.setattr(resources, 'download_model', download)
    monkeypatch.setattr(sys, 'argv', ['resources', '--models-root', str(tmp_path),
                                    '--inventory', str(tmp_path / 'inventory.json')])
    resources.main()
    lock = json.loads(Path(resources.__file__).with_name('resources.lock.json').read_text())['models']
    expected = {item['repository']: {'revision': item['revision'], 'weights_sha256': item['weights_sha256']}
                for item in lock.values()}
    assert dict(calls) == expected


def test_e1b_uses_relocated_paths_without_editing_historical_config(monkeypatch, tmp_path):
    historical = tmp_path / 'historical.json'
    original = {'sources': '/original/source.json', 'phase5_root': '/original/phase5', 'study': 'A1'}
    run = tmp_path / 'analysis'
    run.mkdir()
    config = run / 'config.json'
    config.write_text(json.dumps(original), encoding='utf-8')
    historical.write_text(json.dumps({'phase1_root': 'public default'}), encoding='utf-8')
    monkeypatch.setenv('P24_PHASE1_RUNS_ROOT', str(tmp_path / 'old phase 1 ledger'))
    monkeypatch.setattr(phase6_e1, 'find_run', lambda *args, **kwargs: (run, {}))

    def stop_after_check(current, seed, *, sources):
        assert seed == 42 and current['study'] == 'A1'
        assert current['phase5_root'] == str(tmp_path)
        assert sources['phase1_root'] == str(tmp_path / 'old phase 1 ledger')
        raise RuntimeError('reached source loader')

    monkeypatch.setattr(phase6_e1, 'study_sources', stop_after_check)
    prereg = {'sources': str(historical), 'phase5_root': str(tmp_path), 'seeds': [42],
              'E1b': {'studies': ['A1'], 'analysis_revision': 'v4'}}
    with pytest.raises(RuntimeError, match='reached source loader'):
        phase6_e1.auc_replay(prereg, None, None)
    assert json.loads(config.read_text(encoding='utf-8')) == original
