"""E7: real batched, cache-free ModernBERT/NaFlex serving on the locked T4.

Run with ``python -m repro.run p24 batch_serving --config CONFIG --seed 42``.
CONFIG requires ``parent_run`` (completed phase1 SAVER); ``always_top1_run``
may identify the same-seed phase1 always-vision checkpoint. Without it, a
matching successful ledger run is used if available, otherwise the comparison
is explicitly labeled a forced-vision ablation of the SAVER checkpoint.
``sample_posts`` defaults to 32. No labels, calibration refits or test forwards
are used. Batch response latency is NOT divided by batch size; throughput is.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import math
import random
import subprocess
import threading
import time
from collections import Counter
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .encoders import FrozenEncoders
from .model import SaverCore
from ..common.paths import require_env
from .profile_run import (
    _gpu_guard,
    _json,
    _parent_artifacts,
    _state_and_model_config,
)
from .protocol import digest, post_key, read_records

BATCH_SIZES = (1, 8, 16, 32)
PASSES = 3
WARMUPS = 2


@dataclass(frozen=True)
class ServingPost:
    post_id: str
    records: tuple[dict, ...]

    @property
    def images(self):
        return self.records[0]["images"]


def group_posts(records):
    """Use exactly phase1's tokens+ordered-attachments post identity."""
    grouped = {}
    seen = set()
    for record in records:
        if record["split"] not in {"dev", "val", "valid"}:
            raise ValueError("E7 only accepts dev records; never test or training")
        identity = str(record["id"])
        if identity in seen:
            raise ValueError(f"Duplicate task-unit record ID: {identity}")
        seen.add(identity)
        grouped.setdefault(post_key(record), []).append(record)
    return [
        ServingPost(
            hashlib.sha256(key.encode()).hexdigest(),
            tuple(sorted(rows, key=lambda row: str(row["id"]))),
        )
        for key, rows in sorted(grouped.items())
    ]


def select_dev_posts(records, parent_dev_ids, sample_posts=32):
    """Anchor selection to parent dev IDs, then restore each complete dev post."""
    if type(sample_posts) is not int or sample_posts < 1:
        raise ValueError("sample_posts must be a positive integer")
    wanted = {str(record_id) for record_id in parent_dev_ids}
    if not wanted or len(wanted) != len(parent_dev_ids):
        raise ValueError("Parent dev IDs must be nonempty and unique")
    parent_records = [record for record in records if str(record["id"]) in wanted]
    if len(parent_records) != len(wanted):
        raise ValueError("Parent dev identities missing or duplicated in manifest")
    parent_posts = group_posts(parent_records)
    selected = parent_posts[:sample_posts]
    selected_keys = {post_key(post.records[0]) for post in selected}
    official_dev = [record for record in records if record["split"] in {"dev", "val", "valid"}]
    expanded = [record for record in official_dev if post_key(record) in selected_keys]
    posts = group_posts(expanded)
    selected_parent_ids = [str(record["id"]) for post in selected for record in post.records]
    added_ids = sorted(str(record["id"]) for record in expanded if str(record["id"]) not in wanted)
    return posts, {
        "sample_rule": "Select first sample_posts post identities among parent dev IDs, sorted by phase1 tokens+ordered-images post_key; expand selected identities to ALL official dev records of the same post. Never expand into train/calibration/test.",
        "full_official_dev_posts": True,
        "parent_dev_record_count": len(parent_records),
        "parent_dev_post_count": len(parent_posts),
        "selected_parent_dev_record_ids": selected_parent_ids,
        "selected_parent_dev_record_count": len(selected_parent_ids),
        "expanded_dev_record_count": len(expanded),
        "additional_official_dev_record_count": len(added_ids),
        "additional_official_dev_record_ids": added_ids,
        "official_dev_record_attachment_histogram": dict(
            Counter(len(record["images"]) for record in official_dev)
        ),
        "parent_dev_record_attachment_histogram": dict(
            Counter(len(record["images"]) for record in parent_records)
        ),
    }


def make_batches(posts, batch_size, bucketed):
    if type(batch_size) is not int or batch_size < 1 or not posts:
        raise ValueError("Need nonempty posts and a positive integer batch size")
    if not bucketed:
        return [posts[start : start + batch_size] for start in range(0, len(posts), batch_size)]
    buckets = {}
    for post in posts:
        buckets.setdefault(len(post.images), []).append(post)
    return [
        bucket[start : start + batch_size]
        for _, bucket in sorted(buckets.items())
        for start in range(0, len(bucket), batch_size)
    ]


@torch.no_grad()
def encode_text_batch(encoders, records, task):
    """One padded ModernBERT forward, preserving phase1 entity windows.

    Pairs in one post can require different entity-preserving windows. Equal
    aligned windows share one sequence, but distinct relation records remain
    distinct task units. No singleton encoder forwards are called here.
    """
    if not records or task not in {"mre", "mner"}:
        raise ValueError("Need nonempty records and a supported task")
    prepared, windows, window_index = [], [], {}
    for record in records:
        words = encoders._words(record)
        required = (
            [encoders._entity_span(record, name, len(words)) for name in ("head", "tail")]
            if task == "mre"
            else []
        )
        encoded, start, end = encoders._aligned_inputs(words, required)
        alignment = encoded.word_ids(batch_index=0)
        key = (tuple(encoded["input_ids"][0].tolist()), tuple(alignment))
        if key not in window_index:
            window_index[key] = len(windows)
            windows.append(encoded)
        prepared.append((window_index[key], alignment, start, end, required, len(words)))
    pad_id = encoders.tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Batched ModernBERT requires the pinned tokenizer's pad token")
    input_ids = pad_sequence(
        [encoded["input_ids"][0] for encoded in windows], batch_first=True, padding_value=pad_id
    ).to(encoders.device)
    attention_mask = pad_sequence(
        [encoded["attention_mask"][0] for encoded in windows], batch_first=True, padding_value=0
    ).to(encoders.device)
    states = encoders.text_model(
        input_ids=input_ids, attention_mask=attention_mask
    ).last_hidden_state
    if tuple(states.shape[:2]) != tuple(input_ids.shape):
        raise ValueError("Unexpected batched ModernBERT hidden-state shape")
    results = []
    for index, alignment, start, end, required, word_count in prepared:
        positions = []
        for position, word_id in enumerate(alignment):
            if word_id is not None and word_id == len(positions):
                positions.append(position)
        word_states = states[index, positions].detach()
        result = {"word_states": word_states, "truncated": start != 0 or end != word_count}
        spans_by_name = (
            dict(zip(("head", "tail"), required))
            if task == "mre"
            else {
                "head": [
                    (left, right)
                    for left in range(word_count)
                    for right in range(
                        left + 1, min(word_count, left + encoders.max_span_width) + 1
                    )
                ]
            }
        )
        for name, spans in spans_by_name.items():
            spans = [spans] if task == "mre" else spans
            span_tensor = torch.tensor(spans, device=encoders.device, dtype=torch.long)
            result[name] = encoders.span_features(word_states, span_tensor, start)
            result[name + "_width"] = span_tensor[:, 1] - span_tensor[:, 0]
            if task == "mner":
                result["spans"] = span_tensor
        results.append(result)
    return results, {
        "text_forward_calls": 1,
        "text_sequences": len(windows),
        "text_token_slots": input_ids.numel(),
        "text_valid_tokens": sum(int(encoded["attention_mask"].sum()) for encoded in windows),
    }


@torch.no_grad()
def encode_vision_batch(encoders, images, *, detail=False):
    """One actual NaFlex forward for all global or all acquired detail images."""
    patches = encoders.detail_patches if detail else encoders.global_patches
    if not encoders.is_naflex:
        raise ValueError("E7 requires the registered SigLIP2 NaFlex backbone")
    if not images:
        shape = (0, patches, encoders.vision_hidden) if detail else (0, encoders.vision_hidden)
        return torch.empty(shape, dtype=encoders.dtype, device=encoders.device), {
            "forward_calls": 0,
            "patch_slots": 0,
            "valid_patches": 0,
        }
    with encoders._rgb_images(images) as sources:
        inputs = encoders._vision_inputs(sources, patches)
        output = encoders._vision_forward(inputs)
    mask = inputs["pixel_attention_mask"].bool()
    features = output.last_hidden_state
    if tuple(features.shape) != (len(images), patches, encoders.vision_hidden):
        raise ValueError("Unexpected batched NaFlex hidden-state shape")
    if detail:
        result = features.masked_fill(~mask[..., None], 0)
    else:
        result = output.pooler_output
        if result is None:
            result = features.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)
        if tuple(result.shape) != (len(images), encoders.vision_hidden):
            raise ValueError("Unexpected batched NaFlex global-pooling shape")
    encoded = (result.detach(), mask) if detail else result.detach()
    return encoded, {
        "forward_calls": 1,
        "patch_slots": mask.numel(),
        "valid_patches": int(mask.sum()),
    }


@torch.no_grad()
def forward_batch(model, encoders, posts, image_bank, method, threshold):
    """Batch both backbones and reuse SaverCore's learned layers/patch pooling."""
    if method not in {"saver_k1", "always_top1"}:
        raise ValueError("E7 supports saver_k1 and always_top1 only")
    if model.training:
        raise ValueError("Serving requires an eval-mode task model")
    if math.isnan(threshold) or threshold < 0:
        raise ValueError("Invalid locked routing threshold")
    if any(len(image_bank[post.post_id]) != len(post.images) for post in posts):
        raise ValueError("Decoded attachment bank differs from post identity")
    records = [record for post in posts for record in post.records]
    unit_parts, text_counts = encode_text_batch(encoders, records, model.task)
    names = (
        ("head", "head_width", "tail", "tail_width")
        if model.task == "mre"
        else ("head", "head_width")
    )
    units = {name: torch.cat([part[name] for part in unit_parts]) for name in names}
    mask = torch.ones(len(units["head"]), dtype=torch.bool, device=encoders.device)
    head = model._span(units, "head", mask)
    if model.task == "mre":
        tail = model._span(units, "tail", mask)
        text_state = model.pair_projection(
            torch.cat((head, tail, head * tail, (head - tail).abs()), -1)
        )
    else:
        text_state = head
    queries = F.normalize(model.query_projection(text_state), dim=-1)
    sources = [image for post in posts for image in image_bank[post.post_id]]
    globals_raw, global_counts = encode_vision_batch(encoders, sources)
    globals_projected = F.normalize(
        model.global_projection(globals_raw.to(model.global_projection.weight)), dim=-1
    )
    interactions, unit_ranges, best_indices = [], [], []
    unit_offset, image_offset, record_offset = 0, 0, 0
    for post in posts:
        count = sum(
            len(part["head"])
            for part in unit_parts[record_offset : record_offset + len(post.records)]
        )
        unit_ranges.append((unit_offset, unit_offset + count))
        if post.images:
            scores = (
                queries[unit_offset : unit_offset + count]
                @ globals_projected[image_offset : image_offset + len(post.images)].T
            )
            ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)
            best = ranked[:, 0] + image_offset
            maximum = scores.max(-1).values
            statistics = torch.stack(
                (
                    maximum,
                    scores.mean(-1),
                    scores.std(-1, unbiased=False),
                    scores.sort(dim=-1, descending=True).values[:, :2].mean(-1),
                ),
                -1,
            )
            query = queries[unit_offset : unit_offset + count]
            image = globals_projected[best]
            interactions.append(
                torch.cat((query, image, query * image, (query - image).abs(), statistics), -1)
            )
            best_indices.extend(best.tolist())
        else:
            interactions.append(queries.new_zeros((count, 4 * model.query_dim + 4)))
            best_indices.extend([-1] * count)
        unit_offset += count
        image_offset += len(post.images)
        record_offset += len(post.records)
    retrieved = torch.tensor(best_indices, device=encoders.device, dtype=torch.long)
    eligible = retrieved >= 0
    gate = (
        model.gate_mlp(torch.cat(interactions)).squeeze(-1).sigmoid()
        if method == "saver_k1"
        else queries.new_ones(len(retrieved))
    ).masked_fill(~eligible, 0)
    active = eligible & ((gate >= threshold) if method == "saver_k1" else True)
    selected = retrieved.masked_fill(~active, -1)[:, None]
    required = sorted(set(selected[selected >= 0].tolist()))
    encoded_details, detail_counts = encode_vision_batch(
        encoders, [sources[index] for index in required], detail=True
    )
    if required:
        patches, patch_mask = encoded_details
        details = {index: patches[local][patch_mask[local]] for local, index in enumerate(required)}
    else:
        details = {}
    projected, _ = model._details(required, details, None)
    evidence, _, available = model._pool(
        queries, globals_projected, selected, projected, model.pooling
    )
    vision_state = text_state
    if available.any():
        indices = available.nonzero(as_tuple=False).flatten()
        base = text_state[indices]
        projected = model.evidence_projection(evidence[indices])
        fused = model.fusion_mlp(
            torch.cat((base, projected, base * projected, (base - projected).abs()), -1)
        )
        vision_state = text_state.index_copy(0, indices, fused)
    text_logits = model.classifier(text_state)
    logits = torch.where(active[:, None], model.classifier(vision_state), text_logits)
    counts = {
        **text_counts,
        "global_forward_calls": global_counts["forward_calls"],
        "global_image_encodes": len(sources),
        "global_patch_slots": global_counts["patch_slots"],
        "global_valid_patches": global_counts["valid_patches"],
        "detail_forward_calls": detail_counts["forward_calls"],
        "detail_image_encodes": len(required),
        "detail_patch_slots": detail_counts["patch_slots"],
        "detail_valid_patches": detail_counts["valid_patches"],
        "attachment_slots": len(posts) * max(len(post.images) for post in posts),
        "valid_attachments": len(sources),
        "actual_attachment_padding": 0,
        "unit_padding": 0,
        "task_units": len(mask),
        "active_task_units": int(active.sum()),
        "active_posts": sum(bool(active[start:end].any()) for start, end in unit_ranges),
    }
    return {
        "logits": logits,
        "text_logits": text_logits,
        "gate": gate,
        "active": active,
        "retrieved": retrieved,
        "selected": selected,
        "unit_ranges": unit_ranges,
        "units_per_record": [len(part["head"]) for part in unit_parts],
        "counts": counts,
    }


class GPUUtilization:
    """Read-only telemetry for the same physical T4, never process control."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.samples = []
        self.error = None
        self.stopped = threading.Event()

    def sample(self):
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                require_env("P24_PROFILE_GPU_UUID"),
                "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        rows = list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != 3:
            raise RuntimeError("Invalid GPU telemetry response")
        utilization, memory, temperature = map(float, rows[0])
        self.samples.append(
            {
                "time_unix": time.time(),
                "monotonic_seconds": time.perf_counter(),
                "gpu_utilization": utilization,
                "memory_used_mib": memory,
                "temperature_c": temperature,
            }
        )

    def check(self):
        if self.error is not None:
            raise RuntimeError("GPU telemetry failed") from self.error
        if any(sample["temperature_c"] > 80 for sample in self.samples):
            raise RuntimeError("T4 temperature exceeds the preregistered 80 C limit")
        if any(sample["memory_used_mib"] > 14000 for sample in self.samples):
            raise RuntimeError("T4 memory exceeds the preregistered 14000 MiB limit")

    def _poll(self):
        try:
            while not self.stopped.wait(self.interval):
                self.sample()
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
            self.error = error

    def __enter__(self):
        self.sample()
        self.check()
        self.worker = threading.Thread(target=self._poll, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stopped.set()
        self.worker.join()
        if exc_type is None:
            self.sample()
            self.check()


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_timings(rows):
    if not rows or any(row["status"] != "OK" or row["latency_ms"] <= 0 for row in rows):
        raise ValueError("Need positive successful timing rows")
    elapsed = sum(row["latency_ms"] for row in rows)
    posts = sum(row["posts"] for row in rows)
    units = sum(row["task_units"] for row in rows)
    responses = [row["latency_ms"] for row in rows for _ in range(row["posts"])]
    slots = sum(
        row["text_token_slots"] + row["global_patch_slots"] + row["detail_patch_slots"]
        for row in rows
    )
    valid = sum(
        row["text_valid_tokens"] + row["global_valid_patches"] + row["detail_valid_patches"]
        for row in rows
    )
    result = {
        "posts_per_second": posts * 1000 / elapsed,
        "units_per_second": units * 1000 / elapsed,
        "p50_ms": _percentile(responses, 0.5),
        "p90_ms": _percentile(responses, 0.9),
        "amortized_ms_per_post": elapsed / posts,
        "padding_waste": (slots - valid) / slots,
        "timed_posts": posts,
        "timed_task_units": units,
        "timed_batches": len(rows),
        "timed_seconds": elapsed / 1000,
    }
    for prefix, slot_name, valid_name in (
        ("text", "text_token_slots", "text_valid_tokens"),
        ("global", "global_patch_slots", "global_valid_patches"),
        ("detail", "detail_patch_slots", "detail_valid_patches"),
        ("attachment_layout", "attachment_slots", "valid_attachments"),
    ):
        total = sum(row[slot_name] for row in rows)
        used = sum(row[valid_name] for row in rows)
        result[prefix + "_padding_waste"] = (total - used) / total if total else 0.0
    result["actual_attachment_padding"] = sum(row["actual_attachment_padding"] for row in rows)
    if all("active_task_units" in row for row in rows):
        activated = sum(row["active_task_units"] for row in rows)
        result["detail_coverage"] = activated / units
        result["post_detail_coverage"] = sum(row["active_posts"] for row in rows) / posts
        result["all_off"] = activated == 0
        result["routing_regime"] = (
            "all_off" if activated == 0 else ("all_on" if activated == units else "conditional")
        )
    return result


def _prediction_rows(posts, result, setting, repeat, batch_index):
    rows, record_offset, image_offset = [], 0, 0
    for post, (start, end) in zip(posts, result["unit_ranges"]):
        unit_ids = []
        for record in post.records:
            count = result["units_per_record"][record_offset]
            unit_ids.extend(
                [{"record_id": record["id"], "unit_index": index} for index in range(count)]
            )
            record_offset += 1
        rows.append(
            {
                **setting,
                "pass": repeat,
                "batch_index": batch_index,
                "post_id": post.post_id,
                "record_ids": [record["id"] for record in post.records],
                "units": unit_ids,
                "logits": result["logits"][start:end].cpu().tolist(),
                "text_logits": result["text_logits"][start:end].cpu().tolist(),
                "gate": result["gate"][start:end].cpu().tolist(),
                "active": result["active"][start:end].cpu().tolist(),
                "retrieved": [
                    (index - image_offset if index >= 0 else -1)
                    for index in result["retrieved"][start:end].cpu().tolist()
                ],
                "selected": [
                    (index[0] - image_offset if index[0] >= 0 else -1)
                    for index in result["selected"][start:end].cpu().tolist()
                ],
            }
        )
        image_offset += len(post.images)
    return rows


def profile_batches(
    ctx,
    models,
    encoders,
    posts,
    image_bank,
    threshold,
    *,
    batch_sizes=BATCH_SIZES,
    passes=PASSES,
    warmups=WARMUPS,
    clock=time.perf_counter,
    monitor_factory=GPUUtilization,
):
    """CPU support is exclusively for fake-encoder tests; main always guards CUDA."""
    if not posts or passes < 1 or warmups < 0:
        raise ValueError("Invalid serving profiling population or repeats")
    device = torch.device(encoders.device)
    summaries = []
    for method in ("always_top1", "saver_k1"):
        model = models[method].eval()
        for batch_size in batch_sizes:
            for bucketed in (False, True):
                batches = make_batches(posts, batch_size, bucketed)
                setting = {"method": method, "batch_size": batch_size, "bucketed": bucketed}
                rows = []
                monitor = monitor_factory() if device.type == "cuda" else None
                stage, repeat, batch_index = "warmup", -1, -1
                failure = None
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                try:
                    for batch_index in range(warmups):
                        forward_batch(
                            model,
                            encoders,
                            batches[batch_index % len(batches)],
                            image_bank,
                            method,
                            threshold,
                        )
                        _synchronize(device)
                    stage = "measured"
                    with monitor if monitor is not None else nullcontext():
                        for repeat in range(passes):
                            for batch_index, batch in enumerate(batches):
                                if monitor is not None:
                                    monitor.check()
                                _synchronize(device)
                                started = clock()
                                result = forward_batch(
                                    model, encoders, batch, image_bank, method, threshold
                                )
                                _synchronize(device)
                                ended = clock()
                                elapsed_ms = (ended - started) * 1000
                                row = {
                                    **setting,
                                    "status": "OK",
                                    "pass": repeat,
                                    "batch_index": batch_index,
                                    "post_ids": [post.post_id for post in batch],
                                    "posts": len(batch),
                                    "started_monotonic_seconds": started,
                                    "ended_monotonic_seconds": ended,
                                    "latency_ms": elapsed_ms,
                                    **result["counts"],
                                }
                                rows.append(row)
                                ctx.write_rows("batch_timings", [row])
                                ctx.write_rows(
                                    "batch_predictions",
                                    _prediction_rows(batch, result, setting, repeat, batch_index),
                                )
                                del result
                            ctx.log_event(kind="batch_serving_pass", **setting, repeat=repeat)
                except torch.cuda.OutOfMemoryError as error:
                    failure = {
                        **setting,
                        "status": "OOM",
                        "stage": stage,
                        "pass": repeat,
                        "batch_index": batch_index,
                        "error": str(error),
                        "completed_batches": len(rows),
                        "latency_ms": None,
                    }
                    ctx.write_rows("batch_timings", [failure])
                peak = (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0
                )
                summary = {
                    **setting,
                    "status": "OOM" if failure else "OK",
                    "unique_posts": len(posts),
                    "passes": passes,
                    "warmup_batches": warmups,
                    "peak_gpu_mib": peak,
                    "posts_per_second": None,
                    "p50_ms": None,
                    "p90_ms": None,
                    "units_per_second": None,
                    "padding_waste": None,
                }
                if failure:
                    summary.update(failure)
                else:
                    summary.update(summarize_timings(rows))
                if monitor is not None:
                    ctx.write_rows(
                        "batch_gpu_telemetry", [{**setting, **sample} for sample in monitor.samples]
                    )
                samples = monitor.samples if monitor is not None else []
                summary["gpu_utilization"] = (
                    sum(sample["gpu_utilization"] for sample in samples) / len(samples)
                    if samples
                    else None
                )
                summary["gpu_utilization_samples"] = len(samples)
                summary["gpu_fraction_ge90"] = (
                    sum(sample["gpu_utilization"] >= 90 for sample in samples) / len(samples)
                    if samples
                    else None
                )
                ctx.write_rows("batch_serving_summary", [summary])
                summaries.append(summary)
                if device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()
    return summaries


def _counterpart_run(ctx, config, parent_config):
    if config.get("always_top1_run"):
        return Path(config["always_top1_run"]).expanduser().resolve()
    ledger = Path(ctx.runs_root) / "ledger.jsonl"
    candidates = []
    for entry in read_records(ledger) if ledger.is_file() else []:
        if (
            entry.get("paper") != "P24"
            or entry.get("experiment") != "experiment"
            or entry.get("status") != "SUCCESS"
            or entry.get("seed") != ctx.seed
        ):
            continue
        run = (ledger.parent / entry["run_dir"]).resolve()
        candidate = _json(run / "config.json")
        if (
            all(
                candidate.get(key) == parent_config.get(key)
                for key in ("task", "manifest", "text_model", "vision_model")
            )
            and candidate.get("method") == "always_vision"
        ):
            candidates.append(run)
    if len(set(candidates)) > 1:
        raise ValueError("Multiple always-vision checkpoints; specify always_top1_run")
    return candidates[0] if candidates else None


def main(ctx, config):
    for key, expected in (
        ("batch_sizes", list(BATCH_SIZES)),
        ("profiling_passes", PASSES),
        ("warmup_batches", WARMUPS),
    ):
        if key in config and config[key] != expected:
            raise ValueError(f"E7 {key} is fixed by phase2 preregistration")
    parent_config, mapping, _, provenance, checkpoint = _parent_artifacts(ctx, config)
    if provenance["method"] != "saver" or provenance["task"] != "mre":
        raise ValueError("E7 requires the same-seed formal phase1 MRE SAVER parent")
    identity = _json(Path(config["parent_run"]).expanduser() / "data_identity.json")
    posts, population_metadata = select_dev_posts(
        read_records(parent_config["manifest"]),
        identity["split_ids"]["dev"],
        config.get("sample_posts", 32),
    )
    counterpart = _counterpart_run(ctx, config, parent_config)
    counterpart_config, counterpart_checkpoint = parent_config, checkpoint
    counterpart_provenance = {
        **provenance,
        "comparison_role": "forced_top1_same_saver_checkpoint_not_independent_baseline",
    }
    if counterpart is not None:
        (
            counterpart_config,
            counterpart_mapping,
            _,
            counterpart_provenance,
            counterpart_checkpoint,
        ) = _parent_artifacts(ctx, {"parent_run": str(counterpart)})
        if counterpart_provenance["method"] != "always_vision" or counterpart_mapping != mapping:
            raise ValueError("Counterpart must be the same-seed phase1 always-vision checkpoint")
        if any(
            counterpart_config.get(key) != parent_config.get(key)
            for key in ("task", "text_model", "vision_model")
        ):
            raise ValueError("Counterpart must use the same task and encoders")
        if counterpart_provenance["manifest_sha256"] != provenance["manifest_sha256"]:
            raise ValueError("Counterpart manifest differs from SAVER")
        counterpart_identity = _json(counterpart / "data_identity.json")
        if counterpart_identity["split_ids"]["dev"] != identity["split_ids"]["dev"]:
            raise ValueError("Counterpart dev population differs from SAVER")
        counterpart_provenance["comparison_role"] = "independently_trained_phase1_always_vision"
    ctx.tags.update(
        task="mre",
        stage="batch_serving",
        method="paired",
        parent_run_id=provenance["parent_run_id"],
    )
    ctx.save_json(
        "batch_parent_identity.json",
        {"saver_k1": provenance, "always_top1": counterpart_provenance},
    )
    population_counts = Counter(len(post.images) for post in posts)
    ctx.save_json(
        "batch_population.json",
        {
            **population_metadata,
            "manifest_sha256": provenance["manifest_sha256"],
            "attachment_histogram": dict(population_counts),
            "attachment_bucketing_degenerate": len(population_counts) == 1,
            "single_attachment_population": set(population_counts) == {1},
            "posts": [
                {
                    "post_id": post.post_id,
                    "record_ids": [record["id"] for record in post.records],
                    "images": [{"path": path, "sha256": digest(path)} for path in post.images],
                }
                for post in posts
            ],
        },
    )
    lock_path = Path(ctx.runs_root) / ".p24-profile-gpu0.lock"
    import fcntl

    with lock_path.open("a") as profile_lock:
        try:
            fcntl.flock(profile_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Phase1/phase2 T4 profiler already holds the shared lock") from error
        hardware = _gpu_guard()
        ctx.save_json("batch_hardware.json", hardware)
        random.seed(ctx.seed)
        torch.manual_seed(ctx.seed)
        torch.cuda.manual_seed_all(ctx.seed)
        torch.set_num_threads(4)
        encoders = FrozenEncoders(
            parent_config["text_model"], parent_config["vision_model"], "cuda:0"
        )
        if not encoders.is_naflex or encoders.text_model.config.model_type != "modernbert":
            raise ValueError("E7 requires the registered ModernBERT and SigLIP2 NaFlex backbones")
        models = {}
        for method, model_checkpoint, source in (
            ("saver_k1", checkpoint, parent_config),
            ("always_top1", counterpart_checkpoint, counterpart_config),
        ):
            state, kwargs = _state_and_model_config(model_checkpoint, source, mapping)
            models[method] = (
                SaverCore(
                    len(mapping),
                    task="mre",
                    text_dim=encoders.text_hidden,
                    vision_dim=encoders.vision_hidden,
                    **kwargs,
                )
                .to(encoders.device)
                .eval()
            )
            models[method].load_state_dict(state, strict=True)
        threshold = math.inf if provenance["threshold"] is None else provenance["threshold"]
        ctx.save_json(
            "batch_scope.json",
            {
                "batch_sizes": list(BATCH_SIZES),
                "bucketing": [False, True],
                "passes": PASSES,
                "warmup_batches": WARMUPS,
                "test_evaluations": 0,
                "calibration_refits": 0,
                "timing_includes": [
                    "tokenization",
                    "image_preprocessing",
                    "H2D",
                    "batched_text",
                    "batched_all_attachment_screen",
                    "routing",
                    "batched_conditional_detail",
                    "task_head",
                    "synchronization",
                    "padding_accounting",
                ],
                "timing_excludes": [
                    "model_loading",
                    "disk_IO",
                    "RGB_decode",
                    "raw_output_serialization",
                    "interbatch_queue_delay",
                ],
                "latency_definition": "Post response time = synchronized batch wall time, weighted by actual posts; never divide it by batch size for P50/P90",
                "padding_definition": "Actual masked text tokens and NaFlex patch slots divided by allocated token/patch slots; combined ratio is a slot diagnostic, not FLOP waste. Attachments and task units are packed, so no actual attachment/unit padding; attachment_layout_padding_waste is hypothetical rectangular layout waste.",
                "always_top1_definition": "Acquire union of each task unit's Top-1 attachment; identical to phase1 always_vision on the single-attachment manifest, not all-attachment detail on hypothetical multi-image posts",
                "visual_feature_cache": False,
                "saver_threshold": provenance["threshold"],
                "saver_threshold_source": "Original phase1 METHOD_LOCK.json/calibration.json; never the E1 post-harm threshold",
                "all_off_interpretation": "Report a text-plus-global-screen serving operating point, not evidence of conditional-utility routing advantage",
                "gpu_uuid": hardware["gpu_uuid"],
                "telemetry_interval_seconds": 0.2,
                "gpu_utilization_scope": "Measured pass wall interval, includes interbatch raw logging; nvidia-smi device sample estimates, not exact kernel-active time",
            },
        )
        with ExitStack() as stack:
            image_bank = {
                post.post_id: stack.enter_context(encoders._rgb_images(post.images))
                for post in posts
            }
            summaries = profile_batches(ctx, models, encoders, posts, image_bank, threshold)
        ctx.set_metrics(
            {
                "batch_serving": summaries,
                "settings": {
                    f'{row["method"]}_b{row["batch_size"]}_bucket{int(row["bucketed"])}': row
                    for row in summaries
                },
                "test_evaluations": 0,
                "parent_run_id": provenance["parent_run_id"],
                "profile_gpu_uuid": hardware["gpu_uuid"],
                "attachment_bucketing_degenerate": len(population_counts) == 1,
                "oom_configurations": sum(row["status"] == "OOM" for row in summaries),
            }
        )
