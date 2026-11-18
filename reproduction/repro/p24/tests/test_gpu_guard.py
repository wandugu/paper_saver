import pytest

from repro.p24.experiment import gpu_guard


def test_occupied_gpu_is_rejected_without_process_mutation(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('P24_FORMAL_HOSTNAME', 'formal-test-host')
    monkeypatch.setenv('P24_PILOT_HOSTNAME', 'pilot-test-host')
    monkeypatch.setattr('repro.p24.experiment.socket.gethostname', lambda: 'formal-test-host')
    monkeypatch.setattr('repro.p24.experiment.subprocess.check_output', lambda *args, **kwargs: '0, uuid, Tesla T4, 500, 40')
    with pytest.raises(RuntimeError, match='not eligible'):
        gpu_guard(False)


def test_local_other_users_gpus_are_forbidden(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    monkeypatch.setenv('P24_PILOT_HOSTNAME', 'pilot-test-host')
    monkeypatch.setattr('repro.p24.experiment.socket.gethostname', lambda: 'pilot-test-host')
    with pytest.raises(RuntimeError, match='forbidden'):
        gpu_guard(True)


def test_formal_host_and_single_explicit_device(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    with pytest.raises(RuntimeError, match='Exactly one'):
        gpu_guard(False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    monkeypatch.setenv('P24_PILOT_HOSTNAME', 'pilot-test-host')
    monkeypatch.setattr('repro.p24.experiment.socket.gethostname', lambda: 'pilot-test-host')
    monkeypatch.setenv('P24_FORMAL_HOSTNAME', 'formal-test-host')
    with pytest.raises(RuntimeError, match='configured formal hostname'):
        gpu_guard(False)


def test_idle_gpu_is_recorded(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    monkeypatch.setenv('P24_PILOT_HOSTNAME', 'pilot-test-host')
    monkeypatch.setattr('repro.p24.experiment.socket.gethostname', lambda: 'pilot-test-host')
    monkeypatch.setattr('repro.p24.experiment.subprocess.check_output', lambda *args, **kwargs: '2, uuid, Tesla T4, 3, 40')
    assert gpu_guard(True)['initial_memory_mib'] == 3


@pytest.mark.parametrize('variable,pilot', [('P24_PILOT_HOSTNAME', True), ('P24_FORMAL_HOSTNAME', False)])
@pytest.mark.parametrize('value', [None, ''])
def test_missing_gpu_host_is_rejected_before_hardware_probe(monkeypatch, variable, pilot, value):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    monkeypatch.setenv('P24_PILOT_HOSTNAME', 'pilot-test-host')
    if value is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, value)
    monkeypatch.setattr('repro.p24.experiment.subprocess.check_output',
                        lambda *args, **kwargs: pytest.fail('unexpected GPU probe'))
    with pytest.raises(RuntimeError, match=variable):
        gpu_guard(pilot)


def test_formal_hostname_must_match_exactly(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    monkeypatch.setenv('P24_PILOT_HOSTNAME', 'pilot-test-host')
    monkeypatch.setenv('P24_FORMAL_HOSTNAME', 'formal-test-host')
    monkeypatch.setattr('repro.p24.experiment.socket.gethostname', lambda: 'formal-test-host-other')
    with pytest.raises(RuntimeError, match='configured formal hostname'):
        gpu_guard(False)


def test_profile_requires_uuid_and_does_not_accept_a_prefix(monkeypatch):
    from types import SimpleNamespace

    from repro.p24 import profile_run

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('P24_FORMAL_HOSTNAME', 'formal-test-host')
    monkeypatch.setattr(profile_run.socket, 'gethostname', lambda: 'formal-test-host')
    monkeypatch.delenv('P24_PROFILE_GPU_UUID', raising=False)
    with pytest.raises(RuntimeError, match='P24_PROFILE_GPU_UUID'):
        profile_run._gpu_guard()
    monkeypatch.setenv('P24_PROFILE_GPU_UUID', 'test-device')
    monkeypatch.setattr(profile_run.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout='0, T4, test-device-other, 0'))
    with pytest.raises(RuntimeError, match='Physical GPU identity'):
        profile_run._gpu_guard()
