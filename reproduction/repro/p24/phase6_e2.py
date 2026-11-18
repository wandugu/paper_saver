"""Fine-tune ModernBERT with the two fixed phase5 task-head controls."""

import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import transformers
from torch import nn
from transformers import AutoModel, AutoTokenizer

from repro.common.paths import expand_config_paths, require_env

from .encoders import FrozenEncoders
from .experiment import _mapping, verify_resources
from .phase6_readonly import ReadOnlyStore
from .model import SaverCore
from .protocol import digest, labels_for, post_key, read_records, split_records, task_f1


ARMS = {"FT-text": "text_only", "FT-vision": "always_vision", "FT-shuffled": "always_vision"}
SEEDS = (42, 43, 44)
BATCH = 16
EPOCHS = 5
SMOKE_STEPS = 100
PREREG = Path(__file__).with_name("prereg_p24_phase6_e2.yaml")
NUM_LABELS = 23
MAX_RUN_BYTES = 50_000_000
MIN_FREE_BYTES = 10_000_000_000


class NonFinite(RuntimeError):
    pass


def resources():
    data = Path(expand_config_paths("${DATASET_ROOT}/p24"))
    models = Path(require_env("MODEL_ROOT"))
    return {"manifest": str(data / "mnre_remote.jsonl"), "cache": str(data / "features-v1"),
            "inventory": str(data / "resource_inventory.json"), "text_model": str(models / "ModernBERT-base"),
            "vision_model": str(models / "siglip2-base-patch16-naflex"), "num_labels": NUM_LABELS}


def runtime(cpu=False):
    if (torch.__version__, transformers.__version__) != ("2.6.0+cu118", "4.57.1"):
        raise RuntimeError("P24 requires torch 2.6.0+cu118 and transformers 4.57.1")
    if cpu and (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available()):
        raise RuntimeError("CPU validation requires an empty CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)


def data_for(seed):
    records = read_records(resources()["manifest"])
    splits = split_records(records, seed, fit_cap=None, dev_cap=None)
    mapping = _mapping([record for record in records if record["split"] == "train"], "mre")
    if len(mapping) != NUM_LABELS or mapping["None"] != 0:
        raise ValueError("Pinned MRE label mapping differs")
    return splits, mapping


def batches(records, batch_size=BATCH):
    iterator = iter(records)
    while batch := list(islice(iterator, batch_size)):
        yield batch


class BatchedModernBERT(nn.Module):
    _words = staticmethod(FrozenEncoders._words)
    _entity_span = staticmethod(FrozenEncoders._entity_span)
    _tokenize = FrozenEncoders._tokenize
    _aligned_inputs = FrozenEncoders._aligned_inputs
    span_features = staticmethod(FrozenEncoders.span_features.__wrapped__)

    def __init__(self, device, dtype=torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.max_length = 128
        path = resources()["text_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for entity alignment")
        kwargs = {"local_files_only": True, "torch_dtype": dtype, "attn_implementation": "sdpa"}
        try:
            self.text_model = AutoModel.from_pretrained(path, reference_compile=False, **kwargs)
        except TypeError as error:
            if "reference_compile" not in str(error):
                raise
            self.text_model = AutoModel.from_pretrained(path, **kwargs)
        if self.text_model.config.model_type != "modernbert":
            raise ValueError("Expected the pinned ModernBERT backbone")
        self.text_model.to(self.device).requires_grad_(True)

    def prepare(self, record):
        words = self._words(record)
        spans = [self._entity_span(record, name, len(words)) for name in ("head", "tail")]
        encoded, start, end = self._aligned_inputs(words, spans)
        first = []
        for position, word_id in enumerate(encoded.word_ids(batch_index=0)):
            if word_id is not None and word_id == len(first):
                first.append(position)
        inputs = {name: encoded[name][0].tolist() for name in ("input_ids", "attention_mask")}
        return inputs, first, (start, end), spans

    def forward(self, records):
        prepared = [self.prepare(record) for record in records]
        pad_id = self.tokenizer.pad_token_id
        if self.tokenizer.padding_side != "right" or pad_id is None:
            raise ValueError("Pinned ModernBERT requires right padding")
        length = max(len(inputs["input_ids"]) for inputs, *_ in prepared)
        inputs = {name: torch.tensor([item[name] + [fill] * (length - len(item[name])) for item, *_ in prepared],
                                     dtype=torch.long, device=self.device)
                  for name, fill in (("input_ids", pad_id), ("attention_mask", 0))}
        states = self.text_model(**inputs).last_hidden_state
        results = []
        for hidden, (item, first, bounds, spans) in zip(states, prepared):
            words = hidden.index_select(0, torch.tensor(first, device=self.device))
            units = {"word_states": words, "word_offsets": torch.tensor(bounds, device=self.device),
                     "token_count": len(item["input_ids"])}
            for name, span in zip(("head", "tail"), spans):
                positions = torch.tensor([span], device=self.device)
                units[f"{name}_span"] = positions
                units[f"{name}_width"] = positions[:, 1] - positions[:, 0]
                units[name] = self.span_features(words, positions, bounds[0])
            results.append(units)
        return results


def shuffled_donors(records):
    representatives = {}
    for record in sorted(records, key=lambda row: row["id"]):
        representatives.setdefault(post_key(record), record)
    keys = sorted(representatives)
    if len(keys) < 2:
        raise ValueError("Cannot derange a one-post split")
    rng = random.Random(20261118)
    donors = list(keys)
    for _ in range(10000):
        rng.shuffle(donors)
        if all(representatives[target]["images"] != representatives[source]["images"]
               for target, source in zip(keys, donors)):
            return {target: representatives[source] for target, source in zip(keys, donors)}
    raise ValueError("Failed to find an image-changing within-split derangement")


class FineTuneHead(SaverCore):
    def _span(self, units, name, mask):
        raw = units[name].to(self.span_projection.weight)
        if raw.ndim != 2 or raw.shape != (mask.numel(), 3 * self.text_dim):
            raise ValueError(f"{name} must have shape [U, {3 * self.text_dim}]")
        widths = torch.as_tensor(units[f"{name}_width"], device=raw.device)
        if widths.shape != mask.shape or widths.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name}_width must be an integer [U] tensor")
        if ((widths[mask] < 1) | (widths[mask] > self.max_span_width)).any():
            raise ValueError(f"{name}_width is outside [1, {self.max_span_width}]")
        raw = raw.masked_fill(~mask[:, None], 0)
        widths = widths.long().masked_fill(~mask, 0)
        return self.span_projection(torch.cat((raw, self.width_embedding(widths)), dim=-1))


class FineTune(nn.Module):
    def __init__(self, encoder, mapping, seed, store):
        super().__init__()
        self.encoder = encoder
        self.mapping = mapping
        self.store = store
        self.head = FineTuneHead(NUM_LABELS, text_dim=encoder.text_model.config.hidden_size).to(encoder.device)
        self.head.gate_mlp.requires_grad_(False)

    def forward(self, records, method, *, training, step=1000, donors=None):
        units_batch = self.encoder(records)
        outputs = []
        for record, units in zip(records, units_batch):
            labels = labels_for(record, units, "mre", self.mapping).to(self.encoder.device)
            if method == "always_vision":
                donor = donors[post_key(record)] if donors is not None else record
                globals_ = self.store.text_global(donor, "mre", visual=True)["global"]
                details = {index: self.store.detail(image) for index, image in enumerate(donor["images"])}
            else:
                globals_ = torch.empty((0, self.head.vision_dim), device=self.encoder.device)
                details = None
            output = self.head(units, globals_, details=details, labels=labels, method=method, training=training,
                               step=step, threshold=.5, force_details=method == "always_vision")
            outputs.append((output, labels))
        return outputs


def autocast(enabled):
    return torch.autocast("cuda", dtype=torch.float16) if enabled else nullcontext()


@torch.inference_mode()
def evaluate(model, records, method, mixed, donors=None, retain_logits=False):
    model.eval()
    rows = []
    for batch in batches(records):
        with autocast(mixed):
            outputs = model(batch, method, training=False, donors=donors)
        for record, (output, labels) in zip(batch, outputs):
            if not all(torch.isfinite(output[key]).all() for key in ("logits", "text_logits", "vision_logits", "loss")):
                raise NonFinite(f"Nonfinite evaluation output at {record['id']}")
            eligible = output["image_available"].tolist()
            row = {"id": record["id"], "gold": labels.tolist(), "prediction": output["logits"].argmax(-1).tolist(),
                   "text_prediction": output["text_logits"].argmax(-1).tolist(),
                   "vision_prediction": output["vision_logits"].argmax(-1).tolist(),
                   "text_probabilities": output["text_logits"].softmax(-1).tolist(),
                   "gate": output["gate"].tolist(), "eligible": eligible,
                   "selected": [[0] if valid else [] for valid in eligible], "utility_target": output["utility_targets"].tolist(),
                   "text_ce": output["text_losses"].tolist(), "vision_ce": output["vision_losses"].tolist()}
            if retain_logits:
                row.update({key: output[key].tolist() for key in ("logits", "text_logits", "vision_logits")})
            rows.append(row)
    return rows


def disk_check(ctx):
    usage = shutil.disk_usage("/")
    size = sum(path.stat().st_size for path in ctx.run_dir.rglob("*") if path.is_file())
    if usage.free < MIN_FREE_BYTES or size >= MAX_RUN_BYTES:
        raise RuntimeError(f"Disk stop: {usage.free} free bytes, {size} run bytes")
    ctx.log_event(kind="disk", filesystem_bytes=usage.total, used_bytes=usage.used, free_bytes=usage.free, run_bytes=size)
    return size


def train(ctx, arm, smoke, mixed):
    device = torch.device("cuda:0")
    set_seed(ctx.seed, device)
    splits, mapping = data_for(ctx.seed)
    identity = verify_resources(resources(), "mre")
    encoder = BatchedModernBERT(device)
    store = ReadOnlyStore(resources()["cache"], encoder, identity) if arm != "FT-text" else None
    model = FineTune(encoder, mapping, ctx.seed, store)
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 2e-5},
                                  {"params": [value for value in model.head.parameters() if value.requires_grad], "lr": 1e-4}],
                                 betas=(.9, .999), weight_decay=.01)
    total = math.ceil(len(splits["fit"]) / BATCH) * EPOCHS
    warm = max(1, int(.1 * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warm if step < warm
                                                  else max(0., (total - step - 1) / (total - warm)))
    scaler = torch.cuda.amp.GradScaler(enabled=mixed)
    method = ARMS[arm]
    donors = {split: shuffled_donors(rows) for split, rows in splits.items()} if arm == "FT-shuffled" else {}
    if donors:
        ctx.save_json("shuffle.json", {split: [lookup[post_key(row)]["id"] for row in splits[split]]
                                       for split, lookup in donors.items()})
    ctx.save_json("data_identity.json", {"sha256": digest(resources()["manifest"]), "labels": mapping,
                  "split_ids": {name: [row["id"] for row in rows] for name, rows in splits.items()}})
    torch.cuda.reset_peak_memory_stats()
    disk_check(ctx)
    best, stale, step, snapshot, best_epoch = -1., 0, 0, None, None
    step_times, losses, dev_times = [], [], []
    for epoch in range(EPOCHS):
        model.train()
        ordered = list(splits["fit"])
        random.Random(ctx.seed + epoch).shuffle(ordered)
        for batch in batches(ordered):
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with autocast(mixed):
                outputs = model(batch, method, training=True, step=step, donors=donors.get("fit"))
                loss = torch.stack([output["loss"] for output, _ in outputs]).mean()
            if not torch.isfinite(loss):
                raise NonFinite(f"Nonfinite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise NonFinite(f"Nonfinite gradient at step {step}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()
            step += 1
            elapsed = time.monotonic() - started
            step_times.append(elapsed)
            losses.append(float(loss.detach()))
            if smoke or step % 25 == 0:
                ctx.write_rows("training", [{"epoch": epoch, "step": step, "loss": losses[-1], "seconds": elapsed}])
            if smoke and step == SMOKE_STEPS:
                break
        disk_check(ctx)
        started = time.monotonic()
        dev = evaluate(model, splits["dev"], method, mixed, donors.get("dev"), retain_logits=True)
        dev_times.append(time.monotonic() - started)
        score = task_f1(dev, "mre")
        disk_check(ctx)
        ctx.write_rows(f"dev_epoch_{epoch:02d}", dev)
        ctx.log_event(kind="development", epoch=epoch, step=step, dev_f1=score, seconds=dev_times[-1])
        if smoke:
            ctx.set_metrics({"development_f1": score, "optimizer_steps": step, "seconds_per_step": float(np.mean(step_times)),
                             "dev_seconds": dev_times[-1], "loss_decreased": float(np.mean(losses[-10:])) < float(np.mean(losses[:10])),
                             "first10_loss": float(np.mean(losses[:10])), "last10_loss": float(np.mean(losses[-10:])),
                             "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2, "nonfinite": False,
                             "run_bytes": disk_check(ctx)})
            return
        if score > best:
            best, stale, best_epoch = score, 0, epoch
            snapshot = None
            snapshot = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 2:
            break
    model.load_state_dict(snapshot)
    scores, eval_times = {}, {}
    for split in ("dev", "calibration", "test"):
        disk_check(ctx)
        started = time.monotonic()
        rows = evaluate(model, splits[split], method, mixed, donors.get(split), retain_logits=split == "dev")
        eval_times[split] = time.monotonic() - started
        ctx.write_rows(split, rows)
        scores[split] = task_f1(rows, "mre")
    ctx.set_metrics({"development_f1": scores["dev"], "calibration_f1": scores["calibration"], "test_f1": scores["test"],
                     "selected_epoch": best_epoch, "selected_dev_f1": best, "dev_used_for_early_stopping": True,
                     "optimizer_steps": step, "seconds_per_step": float(np.mean(step_times)), "final_eval_seconds": eval_times,
                     "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2, "nonfinite": False,
                     "run_bytes": disk_check(ctx)})


def main(ctx, config):
    from repro.common.prereg import require_fixed_prereg

    runtime()
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit() or not torch.cuda.is_available():
        raise RuntimeError("E2 training is for the authorized phase7 GPU queue only")
    require_fixed_prereg(ctx, PREREG)
    if set(config) != {"arm", "smoke", "precision", "queue_id"}:
        raise ValueError("E2 requires the preregistration and exact queue config schema")
    arm, smoke, precision = (config[key] for key in ("arm", "smoke", "precision"))
    if (arm not in ARMS or ctx.seed not in SEEDS or precision not in ("amp", "fp32") or type(smoke) is not bool
            or (smoke and (ctx.seed != 42 or arm == "FT-shuffled"))
            or require_env("P24_PHASE7_QUEUE_ID") != config["queue_id"]):
        raise ValueError("Invalid phase7 arm, mode or queue binding")
    ctx.tags.update(stage="phase7", study="E2", arm=arm, smoke=smoke, precision=precision, queue_id=config["queue_id"])
    try:
        train(ctx, arm, smoke, precision == "amp")
    except NonFinite as error:
        ctx.set_metrics({"nonfinite": True})
        ctx.save_json("nonfinite.json", {"error": str(error)})
        raise
