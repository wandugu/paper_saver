"""Paired post-resampling statistics for P24's fixed model comparisons."""

import math

import numpy as np
from scipy.stats import binomtest

from .phase3_replay import Population, post_id
from .phase5_analysis import index


def comparison(vision_rows, text_rows, manifest):
    vision = index(vision_rows)
    text = index(text_rows)
    if vision.keys() != text.keys():
        raise ValueError("Paired unit IDs differ")
    rows = [text[key] for key in sorted(text)]
    if any(row["gold"] != vision[row["id"]]["gold"] for row in rows):
        raise ValueError("Paired gold labels differ")
    population = Population(rows, manifest)
    return (population, np.array([vision[key]["prediction"][0] for key in population.ids]),
            np.array([text[key]["prediction"][0] for key in population.ids]))


def f1(counts):
    hits, predicted, gold = counts.sum(axis=0)
    return float(2 * hits / (predicted + gold)) if predicted + gold else 0.0


def interval(samples):
    if len(samples) < 2:
        raise ValueError("Insufficient valid bootstrap draws")
    low, high, upper = map(float, np.quantile(samples, [.025, .975, .95]))
    se = float(np.std(samples, ddof=1))
    return {"low": low, "high": high, "upper": upper, "se": se, "mde80": 2.80 * se,
            "draws": len(samples)}


def paired_f1(comparisons, draws, random_seed):
    posts = comparisons[0][0].post_ids
    if any(population.post_ids != posts for population, _, _ in comparisons):
        raise ValueError("Common bootstrap requires identical ordered post populations")
    counts = [(population.post_counts(vision), population.post_counts(text))
              for population, vision, text in comparisons]
    generator = np.random.default_rng(random_seed)
    samples = []
    for _ in range(draws):
        sample = generator.integers(0, len(posts), len(posts))
        samples.append(float(np.mean([f1(vision[sample]) - f1(text[sample]) for vision, text in counts])))
    points = [f1(vision) - f1(text) for vision, text in counts]
    return {"point": float(np.mean(points)), "posts": len(posts), **interval(samples)}


def seed_t(points):
    if len(points) != 3:
        raise ValueError("Seed t interval needs all three training seeds")
    mean = float(np.mean(points))
    sd = float(np.std(points, ddof=1))
    se = sd / math.sqrt(3)
    return {"point": mean, "sd": sd, "low": mean - 4.303 * se, "high": mean + 4.303 * se,
            "upper": mean + 2.920 * se, "n": 3}


def post_gain(population, vision, text):
    return np.bincount(population.groups, weights=(vision == population.gold).astype(int)
                       - (text == population.gold).astype(int)) / population.counts


def signs(fix, broken, zero):
    return {"fix": fix, "break": broken, "zero": zero,
            "p": float(binomtest(fix, fix + broken, .5, alternative="two-sided").pvalue) if fix + broken else 1.0}


def counts_by_sign(gains):
    return signs(int(np.sum(gains > 0)), int(np.sum(gains < 0)), int(np.sum(gains == 0)))
