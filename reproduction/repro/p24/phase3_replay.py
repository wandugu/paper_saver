"""Predeclared frozen-policy replay, post metrics, and paired diagnostics."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from ..common.paths import expand_config_paths
from .calibration import cp_upper
from .protocol import post_key, read_records


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def json_read(path):
    return expand_config_paths(json.loads(Path(path).read_text()))


def parents(root, seed):
    root = Path(root)
    rows = read_records(root / 'ledger.jsonl')
    result = {}
    for method in ('saver', 'text_only', 'always_vision'):
        matches = [row for row in rows if row['experiment'] == 'experiment'
                   and row['seed'] == seed and row['status'] == 'SUCCESS'
                   and row.get('tags', {}).get('stage') == 'formal'
                   and row.get('tags', {}).get('method') == method]
        if len(matches) != 1:
            raise ValueError(f'Expected one parent for {seed}/{method}')
        result[method] = root / matches[0]['run_dir']
    return result


def post_id(record):
    return fingerprint(post_key(record))[:20]


def calibration_populations(records, identity, seed, fractions=(.02, .05, .1, .2)):
    train = [row for row in records if row['split'] == 'train']
    groups = sorted({post_key(row) for row in train})
    random.Random(seed).shuffle(groups)
    original = set(identity['split_ids']['calibration'])
    by_id = {row['id']: row for row in records}
    original_groups = {post_key(by_id[key]) for key in original}
    if original_groups != set(groups[:max(1, round(len(groups) * .1))]):
        raise ValueError('Historical held-out posts do not match original seed split')
    fitted = {post_key(by_id[key]) for key in identity['split_ids']['fit']}
    if fitted & original_groups:
        raise ValueError('Actual fit/calibration post leakage')
    extension = [key for key in groups if key not in original_groups and key not in fitted]
    ordered = [key for key in groups if key in original_groups] + extension
    result = {}
    for fraction in fractions:
        count = max(1, round(len(groups) * fraction))
        if len(ordered) < count:
            raise ValueError('Insufficient unused training posts for independent sensitivity')
        selected = set(ordered[:count])
        result[fraction] = [row for row in train if post_key(row) in selected]
    return result


def policy_family(penalties):
    return [{'id': f'e{index}_t{step:03d}', 'eta': float(eta), 'threshold': step / 100}
            for index, eta in enumerate(penalties) for step in range(101)] + [
                {'id': 'off', 'eta': 0., 'threshold': None}]


class Population:
    def __init__(self, rows, manifest, timings=None):
        self.rows = rows
        records = {row['id']: row for row in manifest}
        if not rows or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Population is empty or has repeated IDs')
        self.ids = [row['id'] for row in rows]
        self.gold = np.array([row['gold'][0] for row in rows])
        self.text = np.array([row['text_prediction'][0] for row in rows])
        self.vision = np.array([row['vision_prediction'][0] for row in rows])
        self.gate = np.array([row['gate'][0] for row in rows])
        self.eligible = np.array([row['eligible'][0] for row in rows], dtype=bool)
        if not np.isfinite(self.gate).all() or np.any((self.gate < 0) | (self.gate > 1)):
            raise ValueError('Invalid frozen gate scores')
        keys = [post_id(records[key]) for key in self.ids]
        self.post_ids = sorted(set(keys))
        lookup = {key: index for index, key in enumerate(self.post_ids)}
        self.groups = np.array([lookup[key] for key in keys])
        self.counts = np.bincount(self.groups, minlength=len(self.post_ids))
        self.images = [records[key]['images'] for key in self.ids]
        self.timings = timings
        self.cache = {}

    def active(self, policy):
        threshold = policy['threshold']
        if threshold is None:
            return np.zeros(len(self.rows), dtype=bool)
        score = self.gate - policy['eta'] * (1 - self.gate)
        return self.eligible & (score >= threshold)

    def evaluate(self, policy):
        active = self.active(policy)
        key = np.packbits(active).tobytes()
        if key in self.cache:
            return self.cache[key]
        prediction = np.where(active, self.vision, self.text)
        result = self.metrics(prediction, active)
        self.cache[key] = result
        return result

    def metrics(self, prediction, active=None):
        active = np.zeros(len(prediction), dtype=bool) if active is None else active
        correct = np.bincount(self.groups, weights=(prediction == self.gold), minlength=len(self.post_ids))
        text_correct = np.bincount(self.groups, weights=(self.text == self.gold), minlength=len(self.post_ids))
        gain = (correct - text_correct) / self.counts
        detail_sets = [set() for _ in self.post_ids]
        for index in np.flatnonzero(active):
            selected = self.rows[index].get('selected', [[0]])[0]
            selected = [value for value in selected if value >= 0] or [0]
            for attachment in selected:
                if attachment < len(self.images[index]):
                    detail_sets[self.groups[index]].add(self.images[index][attachment])
        details = np.array([len(values) for values in detail_sets])
        tp = int(np.sum((prediction == self.gold) & (self.gold != 0)))
        denominator = int(np.sum(prediction != 0) + np.sum(self.gold != 0))
        latency = None
        p90 = None
        if self.timings is not None:
            off, on = self.timings['off_ms'], self.timings['on_ms']
            latency_values = off + (on - off) * details
            latency, p90 = float(np.mean(latency_values)), float(np.quantile(latency_values, .9))
        return {'f1': 2 * tp / denominator if denominator else 0.,
                'harm': float(np.maximum(-gain, 0).mean()), 'harmful_posts': int(np.sum(gain < 0)),
                'positive_harm_rate': float(np.mean(gain < 0)), 'gain': float(gain.mean()),
                'detail': float(details.mean()), 'coverage': float(active.mean()),
                'latency_ms': latency, 'p90_ms': p90, 'n_posts': len(self.post_ids), 'n_units': len(self.rows)}

    def post_counts(self, prediction):
        return np.stack([np.bincount(self.groups, weights=value, minlength=len(self.post_ids))
                         for value in ((prediction == self.gold) & (self.gold != 0),
                                       prediction != 0, self.gold != 0)], axis=1)


def calibrate(population, policies, alpha=.1, delta=.05):
    family_size = len(policies) - 1
    result = []
    for policy in policies:
        metric = dict(population.evaluate(policy))
        upper = 0. if policy['threshold'] is None else cp_upper(
            metric['harmful_posts'], metric['n_posts'], 1 - delta / family_size)
        result.append({**policy, **metric, 'ucb': upper, 'feasible': upper <= alpha,
                       'ucb_slack': upper - metric['positive_harm_rate']})
    return result


def select(curve, development, selector, penalty=0.):
    feasible = [row for row in curve if row['feasible']]
    nonempty = [row for row in feasible if row['coverage'] > 0]
    fallback = not nonempty
    if fallback:
        return next(row for row in feasible if row['id'] == 'off'), True
    scored = []
    for row in feasible:
        metric = development.evaluate(row)
        threshold = 2. if row['threshold'] is None else row['threshold']
        cost = metric['latency_ms'] if selector == 'value_latency' else metric['detail']
        if selector == 'max_coverage':
            rank = (row['coverage'], -threshold, -row['eta'])
        elif selector == 'max_dev_f1':
            rank = (metric['f1'], -metric['detail'], -row['ucb'], threshold, -row['eta'])
        else:
            if cost is None:
                raise ValueError('Measured timing endpoints unavailable for latency selection')
            rank = (metric['f1'] - penalty * cost, -cost, -row['ucb'], threshold, -row['eta'])
        scored.append((rank, row['id'], row))
    return max(scored, key=lambda item: (item[0], item[1]))[2], fallback


def add_regret(selected, candidates):
    latency = selected['latency_ms']
    comparable = [row for row in candidates if row['f1'] >= selected['f1'] - 1e-12]
    return {**selected, 'f1_regret': max(row['f1'] for row in candidates) - selected['f1'],
            'latency_regret_ms': None if latency is None else latency - min(row['latency_ms'] for row in candidates),
            'f1_matched_latency_regret_ms': None if latency is None else latency - min(row['latency_ms'] for row in comparable)}


def pareto_flags(rows):
    values = np.array([[-row['f1'], row['harm'], row['detail'], row['latency_ms']] for row in rows])
    return [not bool(np.any(np.all(values <= value + 1e-12, axis=1) &
                               np.any(values < value - 1e-12, axis=1))) for value in values]


def auroc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    return float((rankdata(scores)[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def agreement(population):
    gains = np.bincount(population.groups, weights=(population.vision == population.gold).astype(int)
                        - (population.text == population.gold).astype(int)) / population.counts
    scores = np.bincount(population.groups, weights=population.gate) / population.counts
    comparable = concordant = tied = 0
    for index in range(len(gains)):
        differences = gains[index + 1:] - gains[index]
        ranked = scores[index + 1:] - scores[index]
        valid = differences != 0
        comparable += int(valid.sum())
        concordant += int(np.sum((differences * ranked > 0) & valid))
        tied += int(np.sum((ranked == 0) & valid))
    return {'post_auroc': auroc(scores, gains > 0),
            'unit_parent_post_auroc': auroc(population.gate, (gains > 0)[population.groups]),
            'rank_concordance': (concordant + .5 * tied) / comparable if comparable else None,
            'sign_agreement': float(np.mean((scores >= .5) == (gains > 0))),
            'positive_posts': int(np.sum(gains > 0)), 'negative_posts': int(np.sum(gains < 0)),
            'zero_posts': int(np.sum(gains == 0)), 'comparable_post_pairs': comparable,
            'score_ties': tied, 'n_posts': len(gains)}


def paired(population, first, second, seed, bootstrap=2000):
    first, second = np.asarray(first), np.asarray(second)
    changed = np.bincount(population.groups, weights=(first != second), minlength=len(population.post_ids)) > 0
    first_count, second_count = population.post_counts(first), population.post_counts(second)
    def f1(counts):
        total = counts.sum(axis=0)
        return 2 * total[0] / (total[1] + total[2]) if total[1] + total[2] else 0.
    generator = np.random.default_rng(20261113 + seed)
    deltas = []
    for _ in range(bootstrap):
        sample = generator.integers(0, len(changed), len(changed))
        deltas.append(f1(first_count[sample]) - f1(second_count[sample]))
    correct_delta = np.bincount(population.groups, weights=(first == population.gold).astype(int)
                                - (second == population.gold).astype(int), minlength=len(changed)) / population.counts
    return {'f1_delta': float(f1(first_count) - f1(second_count)),
            'ci_low': float(np.quantile(deltas, .025)), 'ci_high': float(np.quantile(deltas, .975)),
            'post_disagreement': float(changed.mean()), 'disagree_posts': int(changed.sum()),
            'mean_post_accuracy_delta': float(correct_delta.mean()),
            'positive_posts': int(np.sum(correct_delta > 0)), 'negative_posts': int(np.sum(correct_delta < 0)),
            'harm_vs_reference': float(np.maximum(-correct_delta, 0).mean()), 'n_posts': len(changed)}
