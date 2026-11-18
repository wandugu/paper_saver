"""Fixed-host phase2 queue that waits for phase1 lanes and profiles."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


from ..common.paths import require_env


def entries(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def gpu_status(gpu):
    values = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used,temperature.gpu,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    memory, temperature, utilization = map(int, values.strip().split(','))
    return {'memory_mib': memory, 'temperature': temperature, 'utilization': utilization}


def phase1_ready(plan, gpu, session_exists):
    root = Path(plan['phase1_root'])
    if not (root / 'queue' / f'lane{gpu}.done').is_file():
        return False
    prefix = require_env('P24_TMUX_PREFIX')
    if session_exists(f'{prefix}mre-s{42 + gpu}-gpu{gpu}'):
        return False
    if gpu == 0:
        completed = {row['tags'].get('parent_run_id') for row in entries(root / 'runs/ledger.jsonl')
                     if row.get('experiment') == 'profile_run' and row.get('status') == 'SUCCESS'}
        expected = {row['run_id'] for row in entries(root / 'runs/ledger.jsonl')
                    if row.get('experiment') == 'experiment' and row.get('status') == 'SUCCESS'
                    and row.get('tags', {}).get('stage') == 'formal'}
        if len(expected) != 21 or not expected <= completed:
            return False
    return True


def _session_exists(name):
    command = ['tmux', 'has-session', '-t', f'={name}']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'Cannot query tmux session {name!r}: {error}') from error
    if result.returncode == 0:
        return True
    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    if result.returncode == 1 and not stdout and stderr in (f"can't find session: ={name}", f"can't find session: {name}"):
        return False
    diagnostics = '\n'.join(part for part in (stderr, stdout) if part) or 'no diagnostics'
    raise RuntimeError(f'Cannot query tmux session {name!r} (exit {result.returncode}): {diagnostics}')


def matching_entries(job, root):
    result = []
    for row in entries(Path(root) / 'runs/ledger.jsonl'):
        if row['experiment'] != job['experiment'] or row['seed'] != job['seed']:
            continue
        directory = Path(root) / 'runs' / row['run_dir']
        source = json.loads((directory / 'config.json').read_text())
        if source == json.loads(Path(job['config']).read_text()):
            result.append(row)
    return result


def execute(job, root, deadline, *, prereg_sha256):
    queue = Path(root) / 'queue'
    log = queue / (job['name'] + '.log')
    telemetry = queue / (job['name'] + '.telemetry.jsonl')
    state = gpu_status(job['gpu'])
    if state['memory_mib'] >= 500 or state['temperature'] > 80:
        return {'status': 'WAITING', 'name': job['name'], **state}
    command = [sys.executable, '-m', 'repro.run', 'p24', job['experiment'], '--config', job['config'],
               '--seed', str(job['seed']), '--runs-root', str(Path(root) / 'runs'),
               '--prereg', 'repro/p24/prereg_p24_phase2.yaml', '--expect-prereg-sha256', prereg_sha256]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='' if job.get('cpu') else str(job['gpu']), TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    started = time.monotonic()
    with log.open('x') as stream, telemetry.open('x') as samples:
        process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        reason = None
        while process.poll() is None:
            snapshot = gpu_status(job['gpu'])
            samples.write(json.dumps({'at': datetime.now(timezone.utc).isoformat(), 'pid': process.pid,
                                      **snapshot}) + '\n')
            samples.flush()
            if snapshot['temperature'] > 80:
                reason = 'temperature_limit'
            elif snapshot['memory_mib'] > 14000:
                reason = 'memory_limit'
            elif time.monotonic() - started > job.get('max_seconds', 4 * 3600):
                reason = 'registered_time_limit'
            elif datetime.now(timezone.utc) >= deadline:
                reason = 'deadline'
            if reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(5)
    result = {'name': job['name'], 'command': command, 'gpu': job['gpu'], 'seed': job['seed'],
              'initial_gpu': state, 'seconds': time.monotonic() - started, 'exit_code': process.returncode,
              'reason': reason, 'status': 'FINISHED' if process.returncode == 0 else 'FAILED'}
    (queue / (job['name'] + '.exit.json')).write_text(json.dumps(result, indent=2) + '\n')
    return result


def work(plan, lane):
    if socket.gethostname().split('.')[0] != require_env('P24_FORMAL_HOSTNAME') or lane not in (0, 1, 2):
        raise RuntimeError('Phase2 workers only run on the configured formal hostname and GPU0/1/2')
    prefix = require_env('P24_TMUX_PREFIX')
    import fcntl

    root = Path(plan['root'])
    queue = root / 'queue'
    queue.mkdir(parents=True, exist_ok=True)
    deadline = datetime.fromisoformat(plan['deadline']).astimezone(timezone.utc)
    with (queue / f'lane{lane}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in plan['lanes'][lane]:
            while datetime.now(timezone.utc) < deadline:
                if job.get('parent_selector'):
                    candidates = [row for row in entries(root / 'runs/ledger.jsonl')
                                  if row['experiment'] == 'phase2_training' and row['status'] == 'SUCCESS'
                                  and row['seed'] == job['seed'] and row.get('tags', {}).get('stage') == 'formal'
                                  and all(row.get('tags', {}).get(key) == value
                                          for key, value in job['parent_selector'].items())]
                    if len(candidates) != 1:
                        raise RuntimeError('Prospective checkpoint source must be uniquely completed')
                    payload = {'parent_run': str(root / 'runs' / candidates[0]['run_dir'])}
                    path = Path(job['config'])
                    if path.exists() and json.loads(path.read_text()) != payload:
                        raise RuntimeError('Prospective parent config already exists with a different identity')
                    path.write_text(json.dumps(payload, indent=2) + '\n')
                previous = matching_entries(job, root)
                if previous:
                    if previous[-1]['status'] == 'SUCCESS' or (
                            previous[-1]['status'] == 'BLOCKED' and job['experiment'] == 'checkpoint_selection'):
                        break
                    raise RuntimeError(f'Prior unsuccessful run requires review: {previous[-1]}')
                if not phase1_ready(plan, lane, _session_exists) or _session_exists(
                        f'{prefix}phase2-audit-s{job["seed"]}-gpu{lane}'):
                    time.sleep(20)
                    continue
                result = execute(job, root, deadline, prereg_sha256=plan['prereg_sha256'])
                with (queue / f'lane{lane}.jsonl').open('a') as stream:
                    stream.write(json.dumps(result) + '\n')
                if result['status'] == 'WAITING':
                    time.sleep(20)
                    continue
                if result['status'] != 'FINISHED':
                    raise RuntimeError(f'Child failed; preserved without retry: {result}')
                break
            else:
                raise RuntimeError('Deadline reached before all queued jobs completed')
        (queue / f'lane{lane}.done').write_text(datetime.now(timezone.utc).isoformat() + '\n')
    subprocess.run([sys.executable, '-m', 'repro.make_tables', '--runs-root', str(root / 'runs'),
                    '--spec', 'repro/p24/tables.py', '--out-dir', str(root / 'tables')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--lane', type=int, required=True)
    arguments = parser.parse_args()
    work(json.loads(Path(arguments.plan).read_text()), arguments.lane)


if __name__ == '__main__':
    main()
