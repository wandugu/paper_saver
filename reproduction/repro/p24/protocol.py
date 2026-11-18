"""Deterministic splits, strict metrics and frozen-feature storage."""

import hashlib
import json
import os
import random
from pathlib import Path

import torch


METHODS = ('saver', 'text_only', 'always_vision', 'maxsim', 'random', 'mean', 'topk2')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_records(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def post_key(record):
    return json.dumps([record['tokens'], record['images']], ensure_ascii=False, sort_keys=True)


def split_records(records, seed, fit_cap=4096, dev_cap=512, pilot=False):
    train = [record for record in records if record['split'] == 'train']
    groups = sorted({post_key(record) for record in train})
    random.Random(seed).shuffle(groups)
    calibration_keys = set(groups[:max(1, round(len(groups) * .10))])
    fit = [record for record in train if post_key(record) not in calibration_keys]
    calibration = [record for record in train if post_key(record) in calibration_keys]
    dev = [record for record in records if record['split'] in ('dev', 'val', 'valid')]
    test = [record for record in records if record['split'] == 'test']
    for offset, population in enumerate((fit, dev, calibration)):
        random.Random(seed + offset).shuffle(population)
    if pilot:
        return {'fit': fit[:32], 'dev': dev[:16], 'calibration': calibration[:16], 'test': []}
    return {'fit': fit[:fit_cap], 'dev': dev[:dev_cap], 'calibration': calibration, 'test': test}


def labels_for(record, units, task, mapping):
    if task == 'mre':
        return torch.tensor([mapping[record['label']]], dtype=torch.long)
    gold = {(int(start), int(end)): mapping[label] for start, end, label in record['spans']}
    return torch.tensor([gold.get(tuple(span), 0) for span in units['spans'].tolist()], dtype=torch.long)


def decoded_spans(probabilities, spans):
    confidence, prediction = probabilities.max(-1)
    candidates = [(float(confidence[index]), int(start), int(end), int(prediction[index]))
                  for index, (start, end) in enumerate(spans.tolist()) if int(prediction[index]) != 0]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    occupied, accepted = set(), set()
    for score, start, end, label in candidates:
        positions = set(range(start, end))
        if not positions & occupied:
            accepted.add((start, end, label))
            occupied.update(positions)
    return accepted


def task_f1(rows, task):
    true_positive = predicted = gold = 0
    for row in rows:
        if task == 'mre':
            prediction, target = row['prediction'][0], row['gold'][0]
            predicted += prediction != 0
            gold += target != 0
            true_positive += prediction == target and target != 0
        else:
            prediction = {tuple(item) for item in row['decoded']}
            target = {tuple(item) for item in row['gold_spans']}
            true_positive += len(prediction & target)
            predicted += len(prediction)
            gold += len(target)
    denominator = predicted + gold
    return 2 * true_positive / denominator if denominator else 0.0


def cache_key(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


class FeatureStore:
    def __init__(self, root, encoders, identity=None):
        identity = identity if identity is not None else getattr(encoders, 'cache_identity', {})
        namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        self.root = Path(root) / namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.encoders = encoders

    def text_global(self, record, task, visual=True):
        path = self.root / (cache_key(record) + '.pt')
        if path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if not visual or payload.get('global') is not None:
                return payload
        units = self.encoders.encode_text(record, task, return_units=False)
        payload = {'units': {name: value.detach().cpu() for name, value in units.items()
                             if isinstance(value, torch.Tensor)},
                   'truncated': units['truncated'],
                   'global': self.encoders.encode_global(record['images']).cpu() if visual else None}
        temporary = path.with_suffix(f'.{os.getpid()}.partial')
        torch.save(payload, temporary)
        temporary.replace(path)
        return payload

    def units(self, payload):
        units = {name: value.to(self.encoders.device) for name, value in payload['units'].items()}
        for name in ('head', 'tail'):
            if name + '_span' in units:
                units[name] = self.encoders.span_features(units['word_states'], units[name + '_span'],
                                                        int(units['word_offsets'][0]))
        return units

    def detail(self, image, cache=True):
        key = hashlib.sha256(str(image).encode()).hexdigest()
        path = self.root / ('detail-' + key + '.pt')
        if cache and path.exists():
            return torch.load(path, map_location=self.encoders.device, weights_only=True)
        patches, mask = self.encoders.encode_detail(image, return_mask=True)
        patches = patches[mask.bool()]
        if cache:
            temporary = path.with_suffix(f'.{os.getpid()}.partial')
            torch.save(patches.cpu(), temporary)
            temporary.replace(path)
        return patches


def deterministic_draws(record, seed, count, device):
    value = int(hashlib.sha256(f'{record["id"]}:{seed}'.encode()).hexdigest()[:16], 16)
    generator = torch.Generator(device='cpu').manual_seed(value)
    return torch.rand(count, generator=generator).to(device)
