"""E3/E4 training under ``python -m repro.run p24 phase2_training``.

Use the phase1 resource config plus ``router=linear|utility_mlp|saver`` and
``teacher=instantaneous|ema|frozen``. Frozen teachers require ``parent_run``
(a phase1 SAVER run directory or ledger run ID). ``checkpoint_selection`` is
``hard_dev`` by default; ``soft_dev`` and ``last_epoch`` are also supported.
Formal runs always train ten complete epochs with effective batch sixteen.
For a separate, test-free smoke set ``pilot: true, max_steps: 2``; optional
``pilot_epochs`` defaults to two. No CLI flag bypasses the phase1 GPU guard.

``epochs.json`` retains both dev policies and every student checkpoint for
prospective E6. ``raw/utility_epochs.jsonl`` and ``raw/utility_final.jsonl``
contain one row per common diagnostic unit; target is the supervision teacher's
label, student_target is the student's own detached forced-vision improvement.
The diagnostic population never depends on the training seed or its dev order.
Calibration deliberately reuses phase1 pair-level forced-error CP, not E1's
post-harm CP; an infeasible audit disables vision without relaxing its bound.
"""

import copy
from collections.abc import Mapping
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import experiment as phase1
from .model import SaverCore, gate_floor, utility_targets
from .protocol import digest, post_key, read_records, split_records, task_f1


def _router_name(name):
    name = str(name).lower().replace("-", "_")
    return {"utilitymlp_k1": "utility_mlp", "utility_mlp_k1": "utility_mlp"}.get(name, name)


class Phase2Saver(SaverCore):
    """Keep SAVER forward/routing intact, overriding only utility supervision."""

    def __init__(self, num_labels, *, router="saver", **kwargs):
        super().__init__(num_labels, **kwargs)
        self.router = _router_name(router)
        if self.router not in {"linear", "utility_mlp", "saver"}:
            raise ValueError("router must be linear, utility_mlp or saver")
        if self.router == "linear":
            with torch.random.fork_rng(devices=[]):
                self.gate_mlp = nn.Linear(4 * self.query_dim + 4, 1)
        elif self.router == "utility_mlp":
            self.gate_mlp = copy.deepcopy(self.gate_mlp)
        object.__setattr__(self, "_utility_teacher", None)
        self._teacher_output = None
        self.teacher_mode = "instantaneous"

    def set_teacher(self, teacher, mode):
        if mode not in {"instantaneous", "ema", "frozen"}:
            raise ValueError("Unknown teacher mode")
        if (teacher is None) != (mode == "instantaneous") or teacher is self:
            raise ValueError("EMA/frozen supervision needs a separate teacher")
        if teacher is not None:
            teacher.requires_grad_(False)
            teacher.eval()
        object.__setattr__(self, "_utility_teacher", teacher)
        self.teacher_mode = mode

    def forward(self, units, global_features, details=None, **kwargs):
        teacher = self._utility_teacher
        supervised = kwargs.get("labels") is not None
        audit = kwargs.get("force_details", False)
        training = kwargs.get("training")
        training = self.training if training is None else training
        if teacher is None or not supervised or not (training or audit):
            return super().forward(units, global_features, details, **kwargs)
        bank = {} if details is None else dict(details)
        provider = kwargs.get("detail_provider")

        def shared_provider(indices):
            missing = [index for index in indices if index not in bank]
            if missing:
                if provider is None:
                    raise ValueError("Missing shared high-resolution features")
                supplied = provider(missing)
                if isinstance(supplied, Mapping):
                    bank.update(supplied)
                elif isinstance(supplied, torch.Tensor) and supplied.ndim == 2 and len(missing) == 1:
                    bank[missing[0]] = supplied
                else:
                    bank.update(zip(missing, supplied))
            return {index: bank[index] for index in indices}

        teacher_kwargs = dict(kwargs, detail_provider=shared_provider, training=False, force_details=True)
        teacher.eval()
        with torch.no_grad():
            teacher_output = teacher(units, global_features, bank, **teacher_kwargs)
        self._teacher_output = teacher_output
        try:
            return super().forward(units, global_features, bank,
                                   **dict(kwargs, detail_provider=shared_provider))
        finally:
            self._teacher_output = None

    def compute_losses(self, output, labels, *, class_weights=None):
        losses = super().compute_losses(output, labels, class_weights=class_weights)
        losses["student_utility_targets"] = losses["utility_targets"]
        teacher = self._teacher_output
        if teacher is None:
            losses["teacher_text_losses"] = losses["text_losses"].detach()
            losses["teacher_vision_losses"] = losses["vision_losses"].detach()
            return losses
        targets = utility_targets(teacher["text_losses"], teacher["vision_losses"], self.utility_margin)
        valid = output["unit_mask"] & (torch.as_tensor(labels, device=targets.device) != -100)
        visual = valid & output["vision_available"]
        utility_loss = output["logits"].sum() * 0
        if visual.any():
            utility_loss = F.binary_cross_entropy(output["gate"][visual], targets[visual])
        losses.update(utility_targets=targets, utility_loss=utility_loss,
                      teacher_text_losses=teacher["text_losses"].detach(),
                      teacher_vision_losses=teacher["vision_losses"].detach())
        losses["loss"] = (losses["task_loss"] + .25 * utility_loss
                          + .05 * losses["budget_loss"] + .05 * losses["consistency_loss"])
        return losses


@torch.no_grad()
def update_ema(teacher, student, decay=.99):
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must lie in [0, 1)")
    teacher_parameters = dict(teacher.named_parameters())
    student_parameters = dict(student.named_parameters())
    if teacher_parameters.keys() != student_parameters.keys():
        raise ValueError("EMA teacher and student parameter names differ")
    for name, parameter in teacher_parameters.items():
        parameter.mul_(decay).add_(student_parameters[name].detach(), alpha=1 - decay)
    student_buffers = dict(student.named_buffers())
    for name, buffer in teacher.named_buffers():
        buffer.copy_(student_buffers[name])
    teacher.eval()


def diagnostic_records(records, cap=512):
    population = sorted((record for record in records if record["split"] in {"dev", "val", "valid"}),
                        key=lambda record: str(record["id"]))
    identities = [str(record["id"]) for record in population]
    if len(identities) != len(set(identities)):
        raise ValueError("Diagnostic dev identities must be unique")
    return population[:cap]


@torch.no_grad()
def diagnostic_rows(model, store, records, task, mapping, *, seed, epoch, step, stage):
    model.eval()
    rows = []
    for record in records:
        output, units, labels = phase1._forward(model, store, record, task, mapping, "saver", step,
                                               seed=seed, force_details=True)
        for index, gold in enumerate(labels.tolist()):
            eligible = bool(output["unit_mask"][index] and output["vision_available"][index] and gold != -100)
            row = {"id": record["id"], "unit_id": f'{record["id"]}:{index}',
                   "unit_index": index, "epoch": epoch, "seed": seed,
                   "router": model.router, "teacher": model.teacher_mode, "stage": stage,
                   "target": int(output["utility_targets"][index]) if eligible else None,
                   "student_target": int(output["student_utility_targets"][index]) if eligible else None,
                   "gate": float(output["gate"][index]), "eligible": eligible, "gold": gold,
                   "text_loss": float(output["text_losses"][index]),
                   "vision_loss": float(output["vision_losses"][index]),
                   "teacher_text_loss": float(output["teacher_text_losses"][index]),
                   "teacher_vision_loss": float(output["teacher_vision_losses"][index])}
            if "spans" in units:
                row["span"] = units["spans"][index].tolist()
            rows.append(row)
    return rows


def utility_metrics(rows):
    population = [row for row in rows if row["eligible"] and row["student_target"] is not None]
    labels = np.asarray([row["student_target"] for row in population], dtype=int)
    scores = np.asarray([row["gate"] for row in population], dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise RuntimeError("Invalid diagnostic gate probabilities")
    metrics = {"utility_n": len(population), "utility_auroc": None, "utility_brier": None,
               "utility_positive_rate": float(labels.mean()) if len(labels) else None}
    if len(labels):
        metrics["utility_brier"] = float(np.square(scores - labels).mean())
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives and negatives:
        order = np.argsort(scores, kind="stable")
        sorted_scores, sorted_labels = scores[order], labels[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
        ends = np.r_[starts[1:], len(labels)]
        rank_sum = sum(float(sorted_labels[start:end].sum()) * (start + end + 1) / 2
                       for start, end in zip(starts, ends))
        metrics["utility_auroc"] = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    else:
        metrics["utility_auroc_reason"] = "empty_population" if not len(labels) else "single_class"
    return metrics


def flip_metrics(histories):
    flips = transitions = student_flips = 0
    previous = None
    for rows in histories:
        current = {row["unit_id"]: row for row in rows if row["eligible"]}
        if len(current) != sum(bool(row["eligible"]) for row in rows):
            raise ValueError("Duplicate utility diagnostic units")
        if previous is not None:
            if previous.keys() != current.keys():
                raise ValueError("Diagnostic population changed between epochs")
            for identity, row in current.items():
                flips += row["target"] != previous[identity]["target"]
                student_flips += row["student_target"] != previous[identity]["student_target"]
                transitions += 1
        previous = current
    return {"epoch_flip_rate": flips / transitions if transitions else None,
            "student_epoch_flip_rate": student_flips / transitions if transitions else None,
            "utility_flips": flips, "utility_transitions": transitions}


def _parent_teacher(ctx, config, mapping, splits):
    parent = config.get("parent_run")
    if not parent:
        raise RuntimeError("Frozen teacher BLOCKED: parent_run is required")
    root = Path(ctx.runs_root)
    candidates = [Path(parent), root / parent, root / "P24" / parent]
    directory = next((path.resolve() for path in candidates if path.is_dir()), None)
    if directory is None:
        raise RuntimeError("Frozen teacher BLOCKED: parent run does not exist")
    entries = []
    ledger = None
    for ledger_root in dict.fromkeys((directory.parent.parent, directory.parent, root.resolve())):
        candidate = ledger_root / "ledger.jsonl"
        if not candidate.is_file():
            continue
        matches = [row for row in read_records(candidate)
                   if row.get("run_dir") and (ledger_root / row["run_dir"]).resolve() == directory]
        if matches:
            entries, ledger = matches, candidate
            break
    if not entries or entries[-1].get("status") != "SUCCESS":
        raise RuntimeError("Frozen teacher BLOCKED: parent lacks successful ledger provenance")
    entry = entries[-1]
    if entry.get("paper") != "P24" or entry.get("experiment") != "experiment" or entry.get("seed") != ctx.seed:
        raise RuntimeError("Frozen teacher BLOCKED: expected same-seed phase1 SAVER")
    if not (directory / "_SUCCESS").exists() or any((directory / name).exists() for name in
                                                    ("FAILED.json", "BLOCKED.json", "SUPERSEDED.json")):
        raise RuntimeError("Frozen teacher BLOCKED: parent is incomplete or superseded")
    parent_config = json.loads((directory / "config.json").read_text())
    if parent_config.get("method") != "saver" or parent_config.get("task") != config["task"] or parent_config.get("pilot"):
        raise RuntimeError("Frozen teacher BLOCKED: parent is not a formal SAVER run")
    identity = json.loads((directory / "data_identity.json").read_text())
    expected_splits = {name: [record["id"] for record in values] for name, values in splits.items()}
    if identity.get("sha256") != digest(config["manifest"]) or identity.get("labels") != mapping:
        raise RuntimeError("Frozen teacher BLOCKED: dataset or label mapping differs")
    if not config.get("pilot") and identity.get("split_ids") != expected_splits:
        raise RuntimeError("Frozen teacher BLOCKED: seed-specific split differs")
    checkpoint = directory / "best.pt"
    if not checkpoint.is_file() or not (directory / "best.json").is_file():
        raise RuntimeError("Frozen teacher BLOCKED: no completed best-dev checkpoint")
    return checkpoint, {"run_id": entry["run_id"], "run_dir": str(directory),
                        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                        "ledger": str(ledger),
                        "seed": ctx.seed, "best": json.loads((directory / "best.json").read_text())}


def _options(ctx, config):
    pilot = bool(config.get("pilot", False))
    router = _router_name(config.get("router", "saver"))
    teacher = str(config.get("teacher", "instantaneous")).lower()
    selection = config.get("checkpoint_selection", "hard_dev")
    if router not in {"linear", "utility_mlp", "saver"} or teacher not in {"instantaneous", "ema", "frozen"}:
        raise ValueError("Invalid phase2 router/teacher")
    if teacher != "instantaneous" and router != "saver":
        raise ValueError("Preregistered teacher ablations use only the SAVER router")
    if selection not in {"hard_dev", "soft_dev", "last_epoch"}:
        raise ValueError("checkpoint_selection must be hard_dev, soft_dev or last_epoch")
    if config.get("task") != "mre" or config.get("method", "saver") != "saver":
        raise ValueError("Phase2 E3/E4 are preregistered for MRE SAVER Top-1")
    fixed = {"epochs": 10, "effective_batch": 16, "learning_rate": 1e-4,
             "ema_decay": .99, "fit_cap": 4096, "dev_cap": 512}
    for key, value in fixed.items():
        if key in config and config[key] != value:
            raise ValueError(f"Preregistered {key} must be {value}; use pilot-specific flags for smoke runs")
    if not pilot and (ctx.seed not in {42, 43, 44} or any(key in config for key in
                     ("max_steps", "pilot_epochs", "pilot_fit_cap", "pilot_dev_cap", "pilot_diagnostic_cap"))):
        raise ValueError("Formal runs require registered seeds and ten COMPLETE epochs without step caps")
    if pilot and (int(config.get("max_steps", 2)) < 1 or int(config.get("pilot_epochs", 2)) < 1):
        raise ValueError("Pilot step and epoch limits must be positive")
    if pilot and not 1 <= int(config.get("pilot_diagnostic_cap", 16)) <= 16:
        raise ValueError("pilot_diagnostic_cap must lie in [1, 16]")
    return pilot, router, teacher, selection


def _runtime_guard(started, device):
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.0
    if elapsed > 4 * 3600 or peak >= 14000:
        raise RuntimeError("Phase2 runtime or GPU memory stop-loss reached; no retry/tuning")
    return elapsed, peak


def _post_harm(rows, records):
    identities = {record["id"]: post_key(record) for record in records}
    groups = {}
    for row in rows:
        differences = groups.setdefault(identities[row["id"]], [])
        differences.extend(int(prediction != gold) - int(text != gold)
                           for prediction, text, gold in zip(row["prediction"], row["text_prediction"], row["gold"]))
    return sum(max(0, sum(values)) / len(values) for values in groups.values()) / len(groups) if groups else None


def main(ctx, config):
    pilot, router, teacher_mode, selection = _options(ctx, config)
    if (ctx.run_dir / "TEST_STARTED").exists():
        raise RuntimeError("This run already locked its single test evaluation; no resume/replacement")
    started = time.perf_counter()
    ctx.tags.update(task=config["task"], method="saver", stage="pilot" if pilot else "formal",
                    phase=2, router=router, teacher=teacher_mode, checkpoint_selection=selection)
    initial = phase1.gpu_guard(pilot)
    ctx.save_json("gpu_initial.json", initial)
    physical = initial.get("physical_gpu")
    if physical is not None and int(physical) not in {0, 1, 2}:
        raise RuntimeError("Phase2 is restricted to physical GPU0/1/2")
    random.seed(ctx.seed)
    np.random.seed(ctx.seed)
    torch.manual_seed(ctx.seed)
    torch.cuda.manual_seed_all(ctx.seed)
    torch.set_num_threads(4)
    records = read_records(config["manifest"])
    identity = phase1.verify_resources(config, config["task"])
    mapping = phase1._mapping([record for record in records if record["split"] == "train"], config["task"])
    splits = split_records(records, ctx.seed, pilot=pilot)
    if pilot:
        for name, key, default in (("fit", "pilot_fit_cap", 32), ("dev", "pilot_dev_cap", 16)):
            cap = int(config.get(key, default))
            if not 1 <= cap <= default:
                raise ValueError(f"{key} must lie in [1, {default}]")
            splits[name] = splits[name][:cap]
    if not all(splits[name] for name in ("fit", "dev", "calibration")) or (not pilot and not splits["test"]):
        raise RuntimeError("Required source split is empty")
    common = diagnostic_records(records, int(config.get("pilot_diagnostic_cap", 16)) if pilot else 512)
    if not common:
        raise RuntimeError("Common official dev diagnostic population is empty")
    ids = [record["id"] for record in common]
    ids_sha256 = hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest()
    ctx.save_json("diagnostic_ids.json", {"ids": ids, "sha256": ids_sha256,
                  "selection": "Official dev sorted by string ID, first 512; no seed or class balancing",
                  "pilot": pilot})
    ctx.save_json("data_identity.json", {"sha256": digest(config["manifest"]), "labels": mapping,
                  "split_ids": {name: [record["id"] for record in values] for name, values in splits.items()}})
    ctx.save_json("resource_inventory.json", json.loads(Path(config["inventory"]).read_text()))
    parent_checkpoint = parent_metadata = None
    if teacher_mode == "frozen":
        parent_checkpoint, parent_metadata = _parent_teacher(ctx, config, mapping, splits)
    encoders = phase1.FrozenEncoders(config["text_model"], config["vision_model"])
    device = torch.device(encoders.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    store = phase1.FeatureStore(config["cache"], encoders, identity=identity)
    model = Phase2Saver(len(mapping), task=config["task"], router=router,
                       text_dim=encoders.text_hidden, vision_dim=encoders.vision_hidden,
                       forced_steps=4 if pilot else 500, anneal_steps=4 if pilot else 500).to(device)
    teacher = None
    if teacher_mode != "instantaneous":
        teacher = copy.deepcopy(model)
        if parent_checkpoint is not None:
            teacher.load_state_dict(torch.load(parent_checkpoint, map_location=device, weights_only=True))
        model.set_teacher(teacher, teacher_mode)
    parameter_counts = {"router_parameters": sum(parameter.numel() for parameter in model.gate_mlp.parameters()),
                        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                        "teacher_parameters": sum(parameter.numel() for parameter in teacher.parameters()) if teacher else 0,
                        "router_input_dim": 4 * model.query_dim + 4,
                        "router_hidden_dim": model.query_dim if router != "linear" else None}
    ctx.save_json("phase2_protocol.json", {"router": router, "teacher": teacher_mode, "pilot": pilot,
                  "epochs": 10, "effective_batch": 16, "learning_rate": 1e-4, "ema_decay": .99,
                  "hyperparameter_configurations": 1, "seed": ctx.seed, "parent": parent_metadata,
                  "checkpoint_selection": selection, "hard_dev_threshold": .5,
                  "calibration_policy": "phase1_pair_forced_error_CP_not_post_harm",
                  "utility_margin": .05, "diagnostic_ids_sha256": ids_sha256, **parameter_counts})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .999), weight_decay=.01)
    maximum_steps = int(config.get("max_steps", 2)) if pilot else math.ceil(len(splits["fit"]) / 16) * 10
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min((step + 1) / max(1, .1 * maximum_steps), 1) *
        (.5 + .5 * math.cos(math.pi * max(0, step - .1 * maximum_steps) / (.9 * maximum_steps))))
    step = 0
    epochs, histories = [], []
    best_index = 0
    gradients = {"gate_gradient_max": 0.0, "fusion_gradient_max": 0.0, "finite": True}
    optimizer.zero_grad(set_to_none=True)
    eval_step = model.forced_steps + model.anneal_steps
    for epoch in range(int(config.get("pilot_epochs", 2)) if pilot else 10):
        model.train()
        epoch_records = list(splits["fit"])
        random.Random(ctx.seed + epoch).shuffle(epoch_records)
        processed = 0
        for position, record in enumerate(epoch_records):
            output, _, _ = phase1._forward(model, store, record, config["task"], mapping, "saver", step,
                                           training=True, seed=ctx.seed)
            if not torch.isfinite(output["loss"]) or not torch.isfinite(output["gate"]).all():
                raise RuntimeError("Nonfinite phase2 training; no hyperparameter fallback")
            (output["loss"] / 16).backward()
            processed += 1
            if processed % 16 == 0 or position + 1 == len(epoch_records):
                gradients["gate_gradient_max"] = max(gradients["gate_gradient_max"], phase1._gradient(model.gate_mlp))
                gradients["fusion_gradient_max"] = max(gradients["fusion_gradient_max"], phase1._gradient(model.fusion_mlp))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if teacher_mode == "ema":
                    update_ema(teacher, model, .99)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                elapsed, peak = _runtime_guard(started, device)
                progress = {"epoch": epoch, "step": step, "samples_seen_epoch": processed,
                            "loss": float(output["loss"].detach()), "gate_mean": float(output["gate"].detach().mean()),
                            "floor": gate_floor(step - 1, model.forced_steps, model.anneal_steps),
                            "elapsed_seconds": elapsed, "peak_gpu_mib": peak, **gradients}
                if device.type == "cuda" and (step % 25 == 0 or pilot):
                    temperature = float(subprocess.check_output(["nvidia-smi", "-i", str(physical),
                                        "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True).strip())
                    progress["temperature_c"] = temperature
                    if temperature > 80:
                        raise RuntimeError("Phase2 temperature stop-loss reached")
                ctx.write_rows("training", [progress])
                print(json.dumps(progress), flush=True)
                if step % 25 == 0 or pilot:
                    ctx.log_event(kind="training", **progress)
                if pilot and step >= maximum_steps:
                    break
        checkpoint_name = f"epoch_{epoch:02d}.pt"
        torch.save(model.state_dict(), ctx.run_dir / checkpoint_name)
        if teacher_mode == "ema":
            torch.save(teacher.state_dict(), ctx.run_dir / f"teacher_epoch_{epoch:02d}.pt")
        soft_rows = phase1.evaluate(model, store, splits["dev"], config["task"], mapping, "saver", eval_step,
                                    seed=ctx.seed, soft=True)
        hard_rows = phase1.evaluate(model, store, splits["dev"], config["task"], mapping, "saver", eval_step,
                                    seed=ctx.seed, threshold=.5)
        ctx.write_rows("dev_soft", [dict(row, epoch=epoch) for row in soft_rows])
        ctx.write_rows("dev_hard", [dict(row, epoch=epoch) for row in hard_rows])
        entry = {"epoch": epoch, "step": step, "checkpoint": checkpoint_name,
                 "soft_dev_f1": task_f1(soft_rows, config["task"]),
                 "hard_dev_f1": task_f1(hard_rows, config["task"]),
                 "complete": processed == len(epoch_records), "examples": processed}
        epochs.append(entry)
        ctx.save_json("epochs.json", {"epochs": epochs})
        if selection == "last_epoch" or (len(epochs) > 1 and
                entry[f'{selection.split("_")[0]}_dev_f1'] > epochs[best_index][f'{selection.split("_")[0]}_dev_f1']):
            best_index = len(epochs) - 1
        selected = epochs[best_index]
        ctx.save_json("best.json", {"epoch": selected["epoch"], "step": selected["step"],
                      "dev_f1": selected["hard_dev_f1"] if selection == "hard_dev" else selected["soft_dev_f1"],
                      "checkpoint": selected["checkpoint"], "selection": selection,
                      "soft_dev_f1": selected["soft_dev_f1"], "hard_dev_f1": selected["hard_dev_f1"]})
        rows = diagnostic_rows(model, store, common, config["task"], mapping, seed=ctx.seed,
                               epoch=epoch, step=eval_step, stage="epoch")
        ctx.write_rows("utility_epochs", rows)
        histories.append(rows)
        ctx.log_event(kind="development", **entry, **utility_metrics(rows))
        _runtime_guard(started, device)
        if pilot and step >= maximum_steps:
            break
    if not pilot and (len(epochs) != 10 or not all(entry["complete"] for entry in epochs)):
        raise RuntimeError("Formal phase2 run did not complete all ten epochs")
    selected = epochs[best_index]
    model.load_state_dict(torch.load(ctx.run_dir / selected["checkpoint"], map_location=device, weights_only=True))
    if teacher_mode == "ema":
        teacher.load_state_dict(torch.load(ctx.run_dir / f'teacher_epoch_{selected["epoch"]:02d}.pt',
                                          map_location=device, weights_only=True))
    torch.save(model.state_dict(), ctx.run_dir / "best.pt")
    final_rows = diagnostic_rows(model, store, common, config["task"], mapping, seed=ctx.seed,
                                 epoch=selected["epoch"], step=eval_step, stage="selected")
    ctx.write_rows("utility_final", final_rows)
    calibration_rows = phase1.evaluate(model, store, splits["calibration"], config["task"], mapping,
                                       "saver", eval_step, seed=ctx.seed, audit=True)
    ctx.write_rows("calibration", calibration_rows)
    calibration = phase1._calibrate(calibration_rows, config["task"], ctx.seed)
    calibration["policy"] = "phase1_pair_forced_error_CP_not_post_harm"
    ctx.save_json("calibration.json", calibration)
    elapsed, peak = _runtime_guard(started, device)
    metrics = {**parameter_counts, **gradients, **utility_metrics(final_rows), **flip_metrics(histories),
               "development_f1": selected["hard_dev_f1"] if selection == "hard_dev" else selected["soft_dev_f1"],
               "soft_dev_f1": selected["soft_dev_f1"], "hard_dev_f1": selected["hard_dev_f1"],
               "selected_epoch": selected["epoch"], "optimizer_steps": step, "completed_epochs": len(epochs),
               "calibration_threshold": calibration["threshold"], "calibration_coverage": calibration["coverage"],
               "calibration_upper": calibration["upper"], "calibration_risk": calibration["risk"],
               "calibration_n": calibration["n"], "calibration_curve": calibration["curve"],
               "peak_gpu_mib": peak, "training_gpu_hours": elapsed / 3600,
               "diagnostic_ids_sha256": ids_sha256, "test_evaluations": 0}
    ctx.set_metrics(metrics)
    if pilot:
        passed = gradients["gate_gradient_max"] > 0 and gradients["fusion_gradient_max"] > 0 and peak < 14000
        ctx.set_metrics({"pilot_pass": passed, "pilot_seconds": elapsed})
        ctx.save_json("PILOT_ONLY.json", {"test_access": False, "max_steps": maximum_steps,
                      "pilot_epochs": len(epochs), "complete_epochs": sum(entry["complete"] for entry in epochs)})
        ctx.save_json("final_eval.json", {**metrics, "pilot": True, "pilot_pass": passed})
        if not passed:
            raise RuntimeError("Phase2 pilot failed; stop dispatch without tuning")
        return
    ctx.save_json("METHOD_LOCK.json", {"seed": ctx.seed, "method": "saver", "router": router,
                  "teacher": teacher_mode, "parent": parent_metadata, "checkpoint_selection": selection,
                  "best_checkpoint_sha256": digest(ctx.run_dir / "best.pt"), "selected_epoch": selected["epoch"],
                  "threshold": calibration["threshold"], "calibration_policy": calibration["policy"],
                  "test_evaluations_allowed": 1})
    with (ctx.run_dir / "TEST_STARTED").open("x") as stream:
        stream.write("One locked test pass; no resume or replacement.\n")
    threshold = math.inf if calibration["threshold"] is None else calibration["threshold"]
    test_rows = phase1.evaluate(model, store, splits["test"], config["task"], mapping, "saver", eval_step,
                                threshold=threshold, seed=ctx.seed, test=True, ctx=ctx, raw_name="test")
    unit_count = sum(len(row["active"]) for row in test_rows)
    coverage = sum(sum(row["active"]) for row in test_rows) / unit_count if unit_count else 0.0
    elapsed, peak = _runtime_guard(started, device)
    final_metrics = {"micro_f1": task_f1(test_rows, config["task"]), "test_samples": len(test_rows),
                     "test_post_harm": _post_harm(test_rows, splits["test"]),
                     "activation_rate": coverage, "detail_coverage": coverage,
                     "detail_encodes_per_sample": sum(row["detail_encodes"] for row in test_rows) / len(test_rows),
                     "gpu_hours": elapsed / 3600, "peak_gpu_mib": peak, "test_evaluations": 1}
    ctx.set_metrics(final_metrics)
    ctx.save_json("final_eval.json", {**metrics, **final_metrics})
