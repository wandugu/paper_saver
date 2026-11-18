"""Single-T4, batch-one wall time and explicitly scoped FLOP accounting."""

import statistics
import time

import torch
from PIL import Image

from .protocol import deterministic_draws


def _images(record):
    images = []
    for path in record["images"]:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _dense_encoder_flops(config, length, modernbert=False):
    hidden, intermediate, layers = (
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
    )
    if modernbert:
        intervals = getattr(config, "global_attn_every_n_layers", 3)
        local = min(length, getattr(config, "local_attention", 128))
        attention = sum(length if index % intervals == 0 else local for index in range(layers))
        return (
            8 * layers * length * hidden**2
            + 6 * layers * length * hidden * intermediate
            + 4 * length * hidden * attention
        )
    return layers * (
        8 * length * hidden**2 + 4 * length * hidden * intermediate + 4 * length**2 * hidden
    )


def _text_length(encoders, record, task):
    words = encoders._words(record)
    required = (
        [encoders._entity_span(record, name, len(words)) for name in ("head", "tail")]
        if task == "mre"
        else []
    )
    encoded, _, _ = encoders._aligned_inputs(words, required)
    return int(encoded["input_ids"].shape[1])


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _flop_summary(events):
    supported = sum(int(getattr(event, "flops", 0) or 0) for event in events)
    unsupported = sorted(
        {
            event.key
            for event in events
            if not getattr(event, "flops", 0)
            and any(name in event.key.lower() for name in ("attention", "matmul", "bmm", "einsum"))
        }
    )
    return supported, unsupported


@torch.no_grad()
def profile(
    ctx,
    model,
    encoders,
    store,
    records,
    task,
    mapping,
    method,
    threshold,
    rate,
    *,
    warmups=10,
    repeats=3,
):
    """No labels or persistent feature-store access; CPU is for synthetic tests only."""
    if not records or warmups < 0 or repeats < 1:
        raise ValueError("Profile requires nonempty records, warmups >= 0 and repeats >= 1")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Profile record IDs must be unique")
    model.eval()
    device = torch.device(encoders.device)
    raw_images = {
        record["id"]: _images(record) if method != "text_only" else [] for record in records
    }
    text_lengths = {record["id"]: _text_length(encoders, record, task) for record in records}
    warm_banks = {}
    for record in records:
        if method != "text_only":
            warm_banks[record["id"]] = (
                encoders.encode_global(raw_images[record["id"]]),
                {
                    index: encoders.encode_detail(image, return_mask=True)
                    for index, image in enumerate(raw_images[record["id"]])
                },
            )

    def execute(record, hot):
        units = encoders.encode_text(record, task)
        sources = raw_images[record["id"]]
        counts = {
            "actual_text_forwards": 1,
            "actual_global_forwards": 0,
            "actual_global_image_encodes": 0,
            "actual_detail_forwards": 0,
        }
        if method == "text_only":
            globals_ = torch.empty((0, encoders.vision_hidden), device=device, dtype=encoders.dtype)
            provider = None
        elif hot:
            globals_, bank = warm_banks[record["id"]]
            provider = lambda indices: {
                index: bank[index][0][bank[index][1].bool()] for index in indices
            }
        else:
            globals_ = encoders.encode_global(sources)
            counts["actual_global_forwards"] = int(bool(sources))
            counts["actual_global_image_encodes"] = len(sources)
            detail_bank = {}

            def provider(indices):
                for index in indices:
                    if index not in detail_bank:
                        patches, mask = encoders.encode_detail(sources[index], return_mask=True)
                        detail_bank[index] = patches[mask.bool()]
                        counts["actual_detail_forwards"] += 1
                return {index: detail_bank[index] for index in indices}

        output = model(
            units,
            globals_,
            detail_provider=provider,
            method=method,
            step=1000,
            threshold=threshold,
            random_rate=rate,
            random_draws=deterministic_draws(record, ctx.seed, len(units["head"]), device),
            training=False,
        )
        if not hot and (
            counts["actual_detail_forwards"] != output["high_resolution_calls"]
            or counts["actual_global_image_encodes"] != output["global_screen_calls"]
        ):
            raise RuntimeError("Actual encoder forwards disagree with routing counts")
        output["_profile_counts"] = counts
        return output

    results = {}
    timing_rows = []
    cold_counts = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for hot in (False, True):
        for index in range(warmups):
            execute(records[index % len(records)], hot)
        samples = []
        for repeat in range(repeats):
            for record in records:
                _synchronize(device)
                started = time.perf_counter()
                output = execute(record, hot)
                _synchronize(device)
                elapsed = 1000 * (time.perf_counter() - started)
                samples.append(elapsed)
                timing_rows.append(
                    {
                        "id": record["id"],
                        "cache": "visual_features_hot" if hot else "visual_features_cold",
                        "repeat": repeat,
                        "latency_ms": elapsed,
                        "logical_detail_encodes": output["high_resolution_calls"],
                        "logical_global_image_encodes": output["global_screen_calls"],
                        **output["_profile_counts"],
                    }
                )
                if not hot:
                    previous = cold_counts.setdefault(record["id"], output["_profile_counts"])
                    if previous != output["_profile_counts"]:
                        raise RuntimeError(
                            "Hard routing changed across deterministic timing repeats"
                        )
            ctx.log_event(kind="profile_timing", cache="hot" if hot else "cold", repeat=repeat)
        name = "warm" if hot else "cold"
        results[name + "_latency_ms"] = statistics.mean(samples)
        results[name + "_p50_ms"] = statistics.median(samples)
        results[name + "_p90_ms"] = sorted(samples)[int(0.9 * (len(samples) - 1))]
    ctx.write_rows("latency", timing_rows)
    results["peak_gpu_mib"] = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    )
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities, with_flops=True, record_shapes=True
    ) as profiler:
        for record in records:
            profiled_output = execute(record, False)
            if profiled_output["_profile_counts"] != cold_counts[record["id"]]:
                raise RuntimeError("Profiler routing differs from timed cold routing")
        _synchronize(device)
    events = profiler.key_averages()
    lower_bound, unsupported = _flop_summary(events)
    dense_rows = []
    for record in records:
        counts = cold_counts[record["id"]]
        text_length = text_lengths[record["id"]]
        text = _dense_encoder_flops(encoders.text_model.config, text_length, modernbert=True)
        screen = counts["actual_global_image_encodes"] * _dense_encoder_flops(
            encoders.vision_model.config, 64
        )
        detail = counts["actual_detail_forwards"] * _dense_encoder_flops(
            encoders.vision_model.config, 256
        )
        dense_rows.append(
            {
                "id": record["id"],
                "text_dense_flops": text,
                "global_dense_flops": screen,
                "detail_dense_flops": detail,
                "encoder_dense_flops": text + screen + detail,
                "text_subtokens": text_length,
                **counts,
            }
        )
    ctx.write_rows("flops", dense_rows)
    ctx.save_json(
        "profile_scope.json",
        {
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU test only",
            "device": str(device),
            "batch": 1,
            "warmups": warmups,
            "repeats": repeats,
            "samples": len(records),
            "cold_excludes": ["disk_IO", "model_loading", "feature_cache_population"],
            "cold_includes": [
                "preprocessing",
                "H2D",
                "text_forward",
                "all_attachment_screen",
                "conditional_detail",
                "task_head",
            ],
            "warm_scope": "Visual feature-cache hits, text reencoded; no claim of allocator-cold or disk-cold latency.",
            "detail_padding": "NaFlex validity mask applied before task patch attention/pooling.",
            "profiler_flops_lower_bound": lower_bound,
            "profiler_samples": len(records),
            "profiler_flops_is_lower_bound": True,
            "unsupported_attention_operators": unsupported,
            "profiler_lower_bound_reason": "with_flops counts registered operators only; fused attention, norms, activations and unsupported ops may be absent. Wrapper events may have counted child matmuls, so the unsupported list is not additive.",
            "dense_estimate_scope": "Transformer-block attention/projection/MLP matmuls; multiply-add=2 FLOPs; actual aligned text length, padded visual budgets64/256. All global image encodes included even with the detail gate closed. Excludes patch projection, vision pooling head, norms/activation/task-head; not exact full FLOPs.",
            "global_forward_definition": "One encoder invocation batches N attachments; image encodes count N.",
            "profiled_global_image_encodes": sum(
                row["actual_global_image_encodes"] for row in dense_rows
            ),
            "profiled_global_forward_calls": sum(
                row["actual_global_forwards"] for row in dense_rows
            ),
            "profiled_detail_forward_calls": sum(
                row["actual_detail_forwards"] for row in dense_rows
            ),
            "labels_or_test_evaluation": False,
        },
    )
    results["encoder_dense_gflops"] = (
        statistics.mean(row["encoder_dense_flops"] for row in dense_rows) / 1e9
    )
    results["global_dense_gflops"] = (
        statistics.mean(row["global_dense_flops"] for row in dense_rows) / 1e9
    )
    results["profiler_supported_gflops_lower_bound"] = lower_bound / len(records) / 1e9
    results["profiler_flops_is_lower_bound"] = True
    results["screening_included_in_flops"] = True
    results["profile_samples"] = len(records)
    for images in raw_images.values():
        for image in images:
            image.close()
    return results
