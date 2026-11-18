"""Locked reduced-scale training, calibration and single-pass task evaluation."""

import json
import math
import os
import random
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ..common.paths import require_env
from .calibration import calibrate_threshold, cp_upper
from .encoders import FrozenEncoders
from .model import SaverCore, gate_floor
from .protocol import (FeatureStore, decoded_spans, deterministic_draws, digest,
                       labels_for, read_records, split_records, task_f1)


def _mapping(records, task):
    names = {record['label'] for record in records} if task == 'mre' else {
        span[2] for record in records for span in record['spans']}
    return {name: index for index, name in enumerate(['None'] + sorted(names - {'None'}))}


def _gradient(module):
    return math.sqrt(sum(float(parameter.grad.detach().square().sum())
                         for parameter in module.parameters() if parameter.grad is not None))


def gpu_guard(pilot):
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not physical.isdigit():
        raise RuntimeError('Exactly one explicit physical CUDA GPU index is required')
    hostname = socket.gethostname().split('.')[0]
    pilot_hostname = require_env('P24_PILOT_HOSTNAME')
    if not pilot and hostname != require_env('P24_FORMAL_HOSTNAME'):
        raise RuntimeError('All formal P24 training must use the configured formal hostname')
    if hostname == pilot_hostname and physical != '2':
        raise RuntimeError('Local GPU0/1 are forbidden')
    command = ['nvidia-smi', '-i', physical, '--query-gpu=index,uuid,name,memory.used,temperature.gpu',
               '--format=csv,noheader,nounits']
    fields = [value.strip() for value in subprocess.check_output(command, text=True).strip().split(',')]
    if int(fields[3]) >= 500 or int(fields[4]) > 80:
        raise RuntimeError(f'Assigned GPU not eligible; leaving all existing processes untouched: {fields}')
    return {'physical_gpu': int(physical), 'uuid': fields[1], 'name': fields[2],
            'initial_memory_mib': int(fields[3]), 'initial_temperature': int(fields[4])}


def verify_resources(config, task):
    lock = json.loads(Path('repro/p24/resources.lock.json').read_text())
    if task == 'mre':
        formal_host = socket.gethostname().split('.')[0] == require_env('P24_FORMAL_HOSTNAME')
        key = 'normalized_remote_sha256' if formal_host else 'normalized_local_sha256'
        if digest(config['manifest']) != lock['mnre'][key]:
            raise RuntimeError('Dataset manifest differs from committed source identity')
    for role, key in (('text', 'text_model'), ('vision', 'vision_model')):
        if digest(Path(config[key]) / 'model.safetensors') != lock['models'][role]['weights_sha256']:
            raise RuntimeError(f'{role} weights differ from preregistered resource lock')
    return {'models': lock['models'], 'torch': torch.__version__, 'max_length': 128,
            'patches': [64, 256], 'word_pooling': 'first_subtoken'}


def _random_rate(ctx, task):
    ledger = Path(ctx.runs_root) / 'ledger.jsonl'
    for row in reversed(read_records(ledger) if ledger.exists() else []):
        if row.get('paper') == 'P24' and row.get('seed') == ctx.seed and row.get('status') == 'SUCCESS':
            directory = Path(ctx.runs_root) / row['run_dir']
            config = json.loads((directory / 'config.json').read_text())
            if config.get('method') == 'saver' and config.get('task') == task and not config.get('pilot'):
                calibration = json.loads((directory / 'calibration.json').read_text())
                return calibration['coverage'], row['run_id']
    raise RuntimeError('Matching successful SAVER run must finish before random control')


def _forward(model, store, record, task, mapping, method, step, *, training=False,
             threshold=.5, random_rate=.35, seed=42, force_details=False, cache_detail=True):
    payload = store.text_global(record, task, method != 'text_only')
    units = store.units(payload)
    global_features = payload['global']
    if global_features is None:
        global_features = torch.empty((0, model.vision_dim), device=store.encoders.device)
    provider = lambda indices: {index: store.detail(record['images'][index], cache=cache_detail)
                                for index in indices}
    labels = labels_for(record, units, task, mapping).to(store.encoders.device)
    weights = None
    if task == 'mner':
        weights = torch.ones(len(mapping), device=store.encoders.device)
        weights[0] = .25
        if training:
            positive = (labels != 0).nonzero(as_tuple=False).flatten()
            negative = (labels == 0).nonzero(as_tuple=False).flatten()
            keep = max(3, 3 * len(positive))
            if len(negative) > keep:
                generator = torch.Generator().manual_seed(seed + step)
                discarded = negative[torch.randperm(len(negative), generator=generator)[keep:].to(negative.device)]
                labels = labels.clone()
                labels[discarded] = -100
    output = model(units, global_features, detail_provider=provider, method=method, step=step,
                   threshold=threshold, training=training, force_details=force_details,
                   random_draws=deterministic_draws(record, seed, len(labels), store.encoders.device),
                   random_rate=random_rate, labels=labels, class_weights=weights)
    return output, units, labels


@torch.no_grad()
def evaluate(model, store, records, task, mapping, method, step, *, threshold=.5,
             random_rate=.35, seed=42, soft=False, audit=False, test=False, ctx=None, raw_name=None):
    model.eval()
    rows = []
    for record in records:
        output, units, labels = _forward(model, store, record, task, mapping, method, step,
                                        training=soft, threshold=threshold, random_rate=random_rate,
                                        seed=seed, force_details=audit, cache_detail=not test)
        probabilities = output['logits'].softmax(-1)
        row = {'id': record['id'], 'prediction': probabilities.argmax(-1).tolist(),
               'gold': labels.tolist(), 'probabilities': probabilities.tolist(),
               'gate': output['gate'].tolist(), 'active': output['active'].tolist(),
               'eligible': output['image_available'].tolist(),
               'selected': output['selected'].tolist(), 'patchids': output['patchids'].tolist(),
               'text_prediction': output['text_logits'].argmax(-1).tolist(),
               'vision_prediction': output['vision_logits'].argmax(-1).tolist(),
               'detail_encodes': output['high_resolution_calls'],
               'screen_encodes': output['global_screen_calls']}
        if audit:
            row['forced_error'] = (output['vision_logits'].argmax(-1) != labels).tolist()
            row['harm'] = (output['vision_losses'] + .05 >= output['text_losses']).tolist()
        if task == 'mner':
            row['spans'] = units['spans'].tolist()
            row['decoded'] = sorted(decoded_spans(probabilities, units['spans']))
            row['gold_spans'] = [[start, end, mapping[label]] for start, end, label in record['spans']]
        rows.append(row)
        if ctx is not None and raw_name is not None:
            ctx.write_rows(raw_name, [row])
    return rows


def _calibrate(rows, task, seed):
    population = [(score, error, label) for row in rows
                  for score, error, label, valid in zip(row['gate'], row['forced_error'], row['gold'], row['eligible'])
                  if valid]
    raw_count = len(population)
    if task == 'mner' and population:
        groups = {}
        for unit in population:
            groups.setdefault(unit[2], []).append(unit)
        minimum = min(map(len, groups.values()))
        population = []
        generator = random.Random(seed)
        for label, group in sorted(groups.items()):
            generator.shuffle(group)
            population.extend(group[:minimum])
    result = calibrate_threshold([unit[0] for unit in population], [unit[1] for unit in population])
    result['raw_valid_units'] = raw_count
    result['threshold'] = result['threshold'] if math.isfinite(result['threshold']) else None
    return result


def run(ctx, config, pilot=False):
    started = time.perf_counter()
    ctx.save_json('gpu_initial.json', gpu_guard(pilot))
    task, method = config['task'], config['method']
    ctx.tags.update(task=task, method=method, stage='pilot' if pilot else 'formal')
    random.seed(ctx.seed)
    np.random.seed(ctx.seed)
    torch.manual_seed(ctx.seed)
    torch.cuda.manual_seed_all(ctx.seed)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    records = read_records(config['manifest'])
    feature_identity = verify_resources(config, task)
    mapping = _mapping([record for record in records if record['split'] == 'train'], task)
    splits = split_records(records, ctx.seed, pilot=pilot)
    if not all(splits[name] for name in ('fit', 'dev', 'calibration')):
        raise RuntimeError('Required source splits are empty')
    ctx.save_json('data_identity.json', {'sha256': digest(config['manifest']), 'labels': mapping,
                  'split_ids': {name: [record['id'] for record in values] for name, values in splits.items()}})
    ctx.save_json('resource_inventory.json', json.loads(Path(config['inventory']).read_text()))
    encoders = FrozenEncoders(config['text_model'], config['vision_model'])
    store = FeatureStore(config['cache'], encoders, identity=feature_identity)
    model = SaverCore(len(mapping), task=task, text_dim=encoders.text_hidden, vision_dim=encoders.vision_hidden,
                      forced_steps=4 if pilot else 500, anneal_steps=4 if pilot else 500).cuda()
    rate, matched_run = (.35, None) if method != 'random' else _random_rate(ctx, task)
    ctx.save_json('matched_random.json', {'rate': rate, 'saver_run': matched_run})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .999), weight_decay=.01)
    maximum_steps = 12 if pilot else math.ceil(len(splits['fit']) / 16) * 10
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min((step + 1) / max(1, .1 * maximum_steps), 1) *
        (.5 + .5 * math.cos(math.pi * max(0, step - .1 * maximum_steps) / (.9 * maximum_steps))))
    best, stale, step, accumulation = -1.0, 0, 0, 0
    diagnostics = {'gate_gradient_max': 0.0, 'fusion_gradient_max': 0.0, 'finite': True}
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(100 if pilot else 10):
        model.train()
        epoch_records = list(splits['fit'])
        random.Random(ctx.seed + epoch).shuffle(epoch_records)
        for position, record in enumerate(epoch_records):
            output, units, labels = _forward(model, store, record, task, mapping, method, step,
                                            training=True, seed=ctx.seed, random_rate=rate)
            if not torch.isfinite(output['loss']) or not torch.isfinite(output['gate']).all():
                raise RuntimeError('Nonfinite training loss; no hyperparameter fallback allowed')
            (output['loss'] / 16).backward()
            accumulation += 1
            if accumulation == 16 or position + 1 == len(epoch_records):
                diagnostics['gate_gradient_max'] = max(diagnostics['gate_gradient_max'], _gradient(model.gate_mlp))
                diagnostics['fusion_gradient_max'] = max(diagnostics['fusion_gradient_max'], _gradient(model.fusion_mlp))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                accumulation = 0
                gates = output['gate'].detach()
                progress = {'step': step, 'epoch': epoch, 'loss': float(output['loss']),
                            'gate_mean': float(gates.mean()), 'gate_min': float(gates.min()),
                            'gate_max': float(gates.max()), 'floor': gate_floor(step - 1, 4 if pilot else 500, 4 if pilot else 500),
                            'elapsed_seconds': time.perf_counter() - started,
                            'peak_gpu_mib': torch.cuda.max_memory_allocated() / 1024 ** 2, **diagnostics}
                ctx.write_rows('training', [progress])
                print(json.dumps(progress), flush=True)
                if step % 25 == 0 or pilot:
                    ctx.log_event(kind='training', **progress)
            if pilot and step >= maximum_steps:
                break
        dev_rows = evaluate(model, store, splits['dev'], task, mapping, method, 1000,
                            seed=ctx.seed, random_rate=rate, soft=method in ('saver', 'mean', 'topk2'))
        dev_f1 = task_f1(dev_rows, task)
        ctx.log_event(kind='development', epoch=epoch, step=step, f1=dev_f1)
        if dev_f1 > best:
            best, stale = dev_f1, 0
            torch.save(model.state_dict(), ctx.run_dir / 'best.pt')
            ctx.save_json('best.json', {'epoch': epoch, 'step': step, 'dev_f1': dev_f1})
        else:
            stale += 1
        if (pilot and step >= maximum_steps) or (not pilot and step >= 1000 and stale >= 3):
            break
    model.load_state_dict(torch.load(ctx.run_dir / 'best.pt', weights_only=True))
    calibration_rows = evaluate(model, store, splits['calibration'], task, mapping, method, 1000,
                                seed=ctx.seed, random_rate=rate, audit=True)
    ctx.write_rows('calibration', calibration_rows)
    calibration = _calibrate(calibration_rows, task, ctx.seed)
    if method == 'random':
        audited = [(error, active) for row in calibration_rows
                   for error, active, valid in zip(row['forced_error'], row['active'], row['eligible']) if valid]
        active_count = sum(active for error, active in audited)
        errors = sum(error and active for error, active in audited)
        calibration.update(coverage=active_count / len(audited) if audited else 0.0,
                           n=active_count, k=errors, n_total=len(audited),
                           risk=errors / active_count if active_count else None,
                           upper=cp_upper(errors, active_count), threshold=None, curve=[],
                           matching_run=matched_run, matching_probability=rate,
                           not_a_cp_selected_router=True)
    if method in ('always_vision', 'text_only'):
        calibration['not_a_cp_selected_router'] = True
    ctx.save_json('calibration.json', calibration)
    threshold = math.inf if calibration['threshold'] is None else calibration['threshold']
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    ctx.set_metrics({'development_f1': best, 'optimizer_steps': step, 'peak_gpu_mib': peak,
                     'calibration_coverage': calibration['coverage'],
                     'calibration_threshold': calibration['threshold'],
                     'calibration_risk': calibration['risk'], 'calibration_upper': calibration['upper'],
                     'calibration_n': calibration['n'], 'calibration_curve': calibration['curve'], **diagnostics})
    if pilot:
        passed = diagnostics['finite'] and diagnostics['gate_gradient_max'] > 0 and diagnostics['fusion_gradient_max'] > 0 and peak < 14000
        ctx.set_metrics({'pilot_pass': passed, 'pilot_seconds': time.perf_counter() - started})
        return
    ctx.save_json('METHOD_LOCK.json', {'seed': ctx.seed, 'method': method, 'threshold': calibration['threshold'],
                  'best_checkpoint_sha256': digest(ctx.run_dir / 'best.pt'), 'test_evaluations_allowed': 1})
    marker = ctx.run_dir / 'TEST_STARTED'
    with marker.open('x') as stream:
        stream.write('One test pass, no resume or replacement.\n')
    test_rows = evaluate(model, store, splits['test'], task, mapping, method, 1000,
                         threshold=threshold, seed=ctx.seed, random_rate=rate, test=True,
                         ctx=ctx, raw_name='test')
    unit_count = sum(len(row['active']) for row in test_rows)
    ctx.set_metrics({'micro_f1': task_f1(test_rows, task), 'test_samples': len(test_rows),
                     'activation_rate': sum(sum(row['active']) for row in test_rows) / unit_count,
                     'detail_encodes_per_sample': sum(row['detail_encodes'] for row in test_rows) / len(test_rows)})
    ctx.save_json('PROFILE_PENDING.json', {'gpu_host': os.environ.get('P24_FORMAL_HOST', 'formal-host'), 'physical_gpu': 0,
                  'reason': 'All methods measured sequentially on the same physical T4 in a separate ledger run.'})


def main(ctx, config):
    run(ctx, config)
