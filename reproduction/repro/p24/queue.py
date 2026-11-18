"""Fixed-host and fixed-GPU P24 workers; only own child processes are managed."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .protocol import read_records


def gpu_status(index):
    command = ['nvidia-smi', '-i', str(index), '--query-gpu=memory.used,temperature.gpu,utilization.gpu',
               '--format=csv,noheader,nounits']
    values = subprocess.check_output(command, text=True).strip().split(',')
    return {'memory_mib': int(values[0]), 'temperature': int(values[1]), 'utilization': int(values[2])}


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def execute_job(job, runs_root, log_root, deadline):
    gpu = job['gpu']
    state = gpu_status(gpu)
    if state['memory_mib'] >= 500 or state['temperature'] > 80:
        return {'name': job['name'], 'status': 'WAITING_FOR_IDLE', 'gpu': gpu, **state, 'time': _stamp()}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    command = [sys.executable, '-m', 'repro.run', 'p24', job.get('experiment', 'experiment'), '--config', job['config'],
               '--seed', str(job['seed']), '--runs-root', str(runs_root), '--prereg', 'repro/p24/prereg_p24.yaml']
    log_path = log_root / (job['name'] + '.log')
    if log_path.exists() and job.get('experiment') == 'profile_run':
        log_path = log_root / (job['name'] + f'.retry-{int(time.time())}.log')
    telemetry_path = log_root / (job['name'] + '.telemetry.jsonl')
    if telemetry_path.exists() and job.get('experiment') == 'profile_run':
        telemetry_path = log_path.with_suffix('.telemetry.jsonl')
    started = time.monotonic()
    with log_path.open('x') as stream, telemetry_path.open('x') as telemetry:
        process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            snapshot = gpu_status(gpu)
            telemetry.write(json.dumps({'time': _stamp(), 'child_pid': process.pid, **snapshot}) + '\n')
            telemetry.flush()
            reason = None
            if snapshot['temperature'] > 80:
                reason = 'temperature_limit'
            elif snapshot['memory_mib'] > 14000:
                reason = 'memory_limit'
            elif time.monotonic() - started > job.get('max_seconds', 4 * 3600):
                reason = 'registered_job_time_limit'
            elif datetime.now(timezone.utc) >= deadline:
                reason = 'deadline'
            if reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return {'name': job['name'], 'status': 'STOPPED', 'reason': reason, 'time': _stamp(), 'command': command}
            time.sleep(10)
    return {'name': job['name'], 'status': 'FINISHED' if process.returncode == 0 else 'FAILED',
            'exit_code': process.returncode, 'seconds': time.monotonic() - started, 'time': _stamp(), 'command': command}


def _profile_jobs(plan, runs_root, log_root):
    expected = {(job['seed'], job['task'], job['method']) for lane in plan['lanes'] for job in lane}
    ledger = read_records(runs_root / 'ledger.jsonl')
    successful_parents = {row.get('tags', {}).get('parent_run_id') for row in ledger
                          if row.get('paper') == 'P24' and row.get('experiment') == 'profile_run' and row.get('status') == 'SUCCESS'}
    for row in ledger:
        if row.get('paper') != 'P24' or row.get('experiment') != 'experiment' or row.get('status') != 'SUCCESS':
            continue
        directory = runs_root / row['run_dir']
        config = json.loads((directory / 'config.json').read_text())
        identity = (row['seed'], config.get('task'), config.get('method'))
        if identity not in expected:
            continue
        name = f'profile-{identity[1]}-{identity[2]}-s{identity[0]}'
        path = log_root / (name + '.yaml')
        if row.get('run_id') in successful_parents:
            continue
        payload = {'parent_run': str(directory), 'task': identity[1], 'method': identity[2], 'profile_gpu': 0}
        path.write_text(json.dumps(payload, indent=2) + '\n')
        yield {'name': name, 'task': identity[1], 'method': identity[2], 'seed': identity[0],
               'gpu': 0, 'experiment': 'profile_run', 'config': str(path), 'max_seconds': 1800}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--lane', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    runs_root, log_root = Path(plan['runs_root']), Path(plan['logs'])
    log_root.mkdir(parents=True, exist_ok=True)
    deadline = datetime.fromisoformat(plan['deadline'])
    status_path = log_root / f'lane{args.lane}.jsonl'
    lane = plan['lanes'][args.lane]
    for job in lane:
        if datetime.now(timezone.utc) >= deadline:
            break
        while True:
            result = execute_job(job, runs_root, log_root, deadline)
            with status_path.open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            print(json.dumps(result), flush=True)
            if result['status'] != 'WAITING_FOR_IDLE':
                break
            if datetime.now(timezone.utc) >= deadline:
                return
            time.sleep(30)
        if result['status'] != 'FINISHED':
            return
    (log_root / f'lane{args.lane}.done').write_text(_stamp() + '\n')
    if args.lane == 0:
        completed = set()
        while datetime.now(timezone.utc) < deadline:
            for job in _profile_jobs(plan, runs_root, log_root):
                if job['name'] in completed:
                    continue
                result = execute_job(job, runs_root, log_root, deadline)
                with status_path.open('a') as stream:
                    stream.write(json.dumps(result) + '\n')
                if result['status'] == 'WAITING_FOR_IDLE':
                    time.sleep(30)
                    break
                if result['status'] != 'FINISHED':
                    raise RuntimeError(f'Profile failed; artifacts preserved, no silent completion: {result}')
                completed.add(job['name'])
            if all((log_root / f'lane{index}.done').exists() for index in range(len(plan['lanes']))):
                command = [sys.executable, '-m', 'repro.make_tables', '--runs-root', str(runs_root),
                           '--spec', 'repro/p24/tables.py', '--out-dir', str(log_root / 'tables')]
                subprocess.run(command, check=True)
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
