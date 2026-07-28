"""Training, calibration, evaluation, and artifact writing for SAVER."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from models.saver import RiskControlledCalibrator, SaverOutput
from processor.saver_dataset import model_inputs_from_batch, move_batch_to_device


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate_text: float = 2e-5
    learning_rate_vision: float = 1e-5
    learning_rate_head: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gate_anneal_steps: int = 500
    seed: int = 42
    bf16: bool = True
    target_risk: float = 0.10
    calibration_confidence: float = 0.95
    calibration_grid_size: int = 101
    output_dir: str = "result/saver"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrainingConfig":
        from dataclasses import fields

        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                "unknown TrainingConfig keys: {}".format(", ".join(unknown))
            )
        result = cls(**dict(values))
        if result.epochs < 1:
            raise ValueError("epochs must be positive")
        if result.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if result.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if any(
            value <= 0
            for value in (
                result.learning_rate_text,
                result.learning_rate_vision,
                result.learning_rate_head,
            )
        ):
            raise ValueError("learning rates must be positive")
        if result.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if result.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if result.gate_anneal_steps < 0:
            raise ValueError("gate_anneal_steps must be nonnegative")
        if not 0.0 < result.target_risk < 1.0:
            raise ValueError("target_risk must be in (0, 1)")
        if not 0.0 < result.calibration_confidence < 1.0:
            raise ValueError("calibration_confidence must be in (0, 1)")
        if result.calibration_grid_size < 2:
            raise ValueError("calibration_grid_size must be at least two")
        return result


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_value(arguments: Sequence[str]) -> Optional[str]:
    repository = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def collect_run_metadata(
    task: str,
    config: TrainingConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Collect reproducibility metadata without recording credentials."""

    metadata: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "seed": config.seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": {
            name: _package_version(name)
            for name in (
                "numpy",
                "Pillow",
                "PyYAML",
                "scipy",
                "tokenizers",
                "torch",
                "transformers",
            )
        },
        "torch_cuda_build": torch.version.cuda,
        "cudnn": (
            None
            if not torch.backends.cudnn.is_available()
            else torch.backends.cudnn.version()
        ),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_branch": _git_value(["branch", "--show-current"]),
    }
    tracked_changes = _git_value(["status", "--porcelain", "--untracked-files=no"])
    metadata["git_tracked_dirty"] = (
        None if tracked_changes is None else bool(tracked_changes)
    )

    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata["gpu"] = {
            "index": index,
            "name": properties.name,
            "total_memory_mib": properties.total_memory / (1024.0**2),
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
        }
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,name,memory.total",
                    "--format=csv,noheader,nounits",
                    "--id={}".format(index),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            metadata["nvidia_smi"] = query.stdout.strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            metadata["nvidia_smi"] = None
    else:
        metadata["gpu"] = None
    return metadata


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def micro_f1(
    predictions: Sequence[int],
    labels: Sequence[int],
    ignored_label: Optional[int] = None,
) -> float:
    """Compute single-label micro-F1 without a scikit-learn dependency."""

    true_positive = 0
    false_positive = 0
    false_negative = 0
    classes = sorted(set(predictions) | set(labels))
    if ignored_label is not None:
        classes = [label for label in classes if label != ignored_label]
    for label in classes:
        true_positive += sum(
            prediction == label and target == label
            for prediction, target in zip(predictions, labels)
        )
        false_positive += sum(
            prediction == label and target != label
            for prediction, target in zip(predictions, labels)
        )
        false_negative += sum(
            prediction != label and target == label
            for prediction, target in zip(predictions, labels)
        )
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def strict_span_f1(
    predicted: Sequence[Sequence[Mapping[str, Any]]],
    gold: Sequence[Sequence[Mapping[str, Any]]],
) -> float:
    """Strict entity-level micro-F1 on ``(start, end, label)`` tuples."""

    true_positive = 0
    predicted_count = 0
    gold_count = 0
    for predicted_sample, gold_sample in zip(predicted, gold):
        predicted_set = {
            (int(item["start"]), int(item["end"]), int(item["label"]))
            for item in predicted_sample
        }
        gold_set = {
            (int(item["start"]), int(item["end"]), int(item["label"]))
            for item in gold_sample
        }
        true_positive += len(predicted_set & gold_set)
        predicted_count += len(predicted_set)
        gold_count += len(gold_set)
    denominator = predicted_count + gold_count
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


class SaverTrainer:
    """Single-GPU trainer with an explicit anti-collapse routing schedule."""

    def __init__(
        self,
        model: nn.Module,
        task: str,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.task = task.lower()
        if self.task not in {"mre", "mner"}:
            raise ValueError("task must be 'mre' or 'mner'")
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.global_step = 0
        self.threshold = float(model.config.gate_threshold)
        self.optimizer = torch.optim.AdamW(
            self._optimizer_groups(),
            weight_decay=config.weight_decay,
        )
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_metadata = collect_run_metadata(self.task, config, self.device)
        run_metadata["training_config"] = asdict(config)
        run_metadata["model_config"] = self.model.config.to_dict()
        write_json(self.output_dir / "run_metadata.json", run_metadata)

    def _optimizer_groups(self) -> List[Dict[str, Any]]:
        groups: Dict[str, List[nn.Parameter]] = {
            "text": [],
            "vision": [],
            "head": [],
        }
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("text_encoder."):
                groups["text"].append(parameter)
            elif name.startswith("vision_encoder."):
                groups["vision"].append(parameter)
            else:
                groups["head"].append(parameter)
        return [
            {
                "params": groups["text"],
                "lr": self.config.learning_rate_text,
            },
            {
                "params": groups["vision"],
                "lr": self.config.learning_rate_vision,
            },
            {
                "params": groups["head"],
                "lr": self.config.learning_rate_head,
            },
        ]

    def _route_schedule(self) -> Tuple[str, float]:
        warmup = int(self.model.config.gate_warmup_steps)
        if self.global_step < warmup:
            return "vision", 1.0
        anneal = int(self.config.gate_anneal_steps)
        if anneal <= 0:
            return "soft", 0.0
        elapsed = self.global_step - warmup
        if elapsed < anneal:
            return "soft", 1.0 - elapsed / anneal
        return "soft", 0.0

    def _autocast(self):
        enabled = (
            self.config.bf16
            and self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=enabled,
        )

    def train_epoch(self, loader: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals: Dict[str, float] = {}
        batches = 0
        accumulation = self.config.gradient_accumulation_steps
        for batch_index, host_batch in enumerate(loader):
            batch = move_batch_to_device(host_batch, self.device)
            model_inputs = model_inputs_from_batch(batch, self.task)
            routing_mode, gate_floor = self._route_schedule()
            with self._autocast():
                output: SaverOutput = self.model(
                    **model_inputs,
                    routing_mode=routing_mode,
                    gate_floor=gate_floor,
                )
                if output.loss is None:
                    raise RuntimeError("training batch did not produce a loss")
                scaled_loss = output.loss / accumulation
            scaled_loss.backward()
            if (batch_index + 1) % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            for name, value in output.losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1

        if batches and batches % accumulation:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
        return {name: value / max(1, batches) for name, value in totals.items()}

    @staticmethod
    def _gold_mner(
        labels: torch.Tensor,
        word_spans: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> List[List[Dict[str, int]]]:
        result: List[List[Dict[str, int]]] = []
        for batch_index in range(labels.size(0)):
            sample: List[Dict[str, int]] = []
            for span_index in range(labels.size(1)):
                if not bool(span_mask[batch_index, span_index]):
                    continue
                label = int(labels[batch_index, span_index])
                if label <= 0:
                    continue
                start, end = [
                    int(value) for value in word_spans[batch_index, span_index].tolist()
                ]
                sample.append({"start": start, "end": end, "label": label})
            result.append(sample)
        return result

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable[Mapping[str, Any]],
        routing_mode: str = "hard",
        artifact_name: Optional[str] = None,
    ) -> Dict[str, float]:
        self.model.eval()
        losses: List[float] = []
        gate_scores: List[torch.Tensor] = []
        gate_active: List[torch.Tensor] = []
        unit_errors: List[torch.Tensor] = []
        all_off: List[bool] = []
        all_on: List[bool] = []
        records: List[Dict[str, Any]] = []
        if self.task == "mre":
            predictions: List[int] = []
            labels: List[int] = []
        else:
            predicted_spans: List[List[Dict[str, Any]]] = []
            gold_spans: List[List[Dict[str, Any]]] = []

        for host_batch in loader:
            batch = move_batch_to_device(host_batch, self.device)
            inputs = model_inputs_from_batch(batch, self.task)
            with self._autocast():
                output: SaverOutput = self.model(
                    **inputs,
                    routing_mode=routing_mode,
                    threshold=self.threshold,
                )
            if output.loss is not None:
                losses.append(float(output.loss))
            if self.task == "mre":
                batch_predictions = output.logits.argmax(-1)
                batch_labels = batch["labels"]
                valid = torch.ones_like(batch_labels, dtype=torch.bool)
                predictions.extend(batch_predictions.cpu().tolist())
                labels.extend(batch_labels.cpu().tolist())
            else:
                batch_predictions = output.logits.argmax(-1)
                batch_labels = batch["labels"]
                valid = batch["span_mask"].bool() & batch_labels.ne(-100)
                batch_decoded = self.model.decode_non_overlapping(
                    output.logits,
                    batch["word_spans"],
                    batch["span_mask"],
                )
                batch_gold = self._gold_mner(
                    batch_labels,
                    batch["word_spans"],
                    batch["span_mask"],
                )
                predicted_spans.extend(batch_decoded)
                gold_spans.extend(batch_gold)

            scores = output.gate_probability.detach()
            active = output.gate_active.detach().bool()
            gate_scores.append(scores[valid].float().cpu())
            gate_active.append(active[valid].cpu())
            unit_errors.append(batch_predictions.ne(batch_labels)[valid].cpu())
            for sample_index in range(valid.size(0)):
                sample_valid = valid[sample_index].reshape(-1)
                sample_active = active[sample_index].reshape(-1)[sample_valid]
                all_off.append(not bool(sample_active.any()))
                all_on.append(bool(sample_active.all()))

            if artifact_name is not None:
                for sample_index, sample_id in enumerate(batch["sample_ids"]):
                    base_record: Dict[str, Any] = {
                        "sample_id": str(sample_id),
                        "image_paths": list(batch["image_paths"][sample_index]),
                    }
                    if self.task == "mre":
                        base_record.update(
                            {
                                "gold_label": int(batch_labels[sample_index]),
                                "predicted_label": int(batch_predictions[sample_index]),
                                "logits": output.logits[sample_index]
                                .detach()
                                .float()
                                .cpu()
                                .tolist(),
                                "text_logits": output.text_logits[sample_index]
                                .detach()
                                .float()
                                .cpu()
                                .tolist(),
                                "visual_logits": output.visual_logits[sample_index]
                                .detach()
                                .float()
                                .cpu()
                                .tolist(),
                                "gate_score": float(scores[sample_index]),
                                "gate_active": bool(active[sample_index]),
                                "selected_attachments": (
                                    output.selected_indices[sample_index]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "attachment_relevance": (
                                    output.attachment_relevance[sample_index]
                                    .detach()
                                    .float()
                                    .cpu()
                                    .tolist()
                                ),
                                "selected_regions": (
                                    output.selected_regions[sample_index]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ),
                            }
                        )
                    else:
                        span_records: List[Dict[str, Any]] = []
                        valid_indices = (
                            valid[sample_index].nonzero(as_tuple=False).squeeze(-1)
                        )
                        for span_index in valid_indices.tolist():
                            span_records.append(
                                {
                                    "word_span": (
                                        batch["word_spans"][
                                            sample_index,
                                            span_index,
                                        ]
                                        .detach()
                                        .cpu()
                                        .tolist()
                                    ),
                                    "gold_label": int(
                                        batch_labels[
                                            sample_index,
                                            span_index,
                                        ]
                                    ),
                                    "predicted_label": int(
                                        batch_predictions[
                                            sample_index,
                                            span_index,
                                        ]
                                    ),
                                    "logits": (
                                        output.logits[
                                            sample_index,
                                            span_index,
                                        ]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .tolist()
                                    ),
                                    "text_logits": (
                                        output.text_logits[
                                            sample_index,
                                            span_index,
                                        ]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .tolist()
                                    ),
                                    "visual_logits": (
                                        output.visual_logits[
                                            sample_index,
                                            span_index,
                                        ]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .tolist()
                                    ),
                                    "gate_score": float(
                                        scores[
                                            sample_index,
                                            span_index,
                                        ]
                                    ),
                                    "gate_active": bool(
                                        active[
                                            sample_index,
                                            span_index,
                                        ]
                                    ),
                                    "selected_attachments": (
                                        output.selected_indices[
                                            sample_index,
                                            span_index,
                                        ]
                                        .detach()
                                        .cpu()
                                        .tolist()
                                    ),
                                    "attachment_relevance": (
                                        output.attachment_relevance[
                                            sample_index,
                                            span_index,
                                        ]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .tolist()
                                    ),
                                    "selected_regions": (
                                        output.selected_regions[
                                            sample_index,
                                            span_index,
                                        ]
                                        .detach()
                                        .cpu()
                                        .tolist()
                                    ),
                                }
                            )
                        base_record.update(
                            {
                                "predicted_entities": batch_decoded[sample_index],
                                "gold_entities": batch_gold[sample_index],
                                "spans": span_records,
                            }
                        )
                    records.append(base_record)

        if not gate_scores:
            raise ValueError("evaluation loader produced no batches")
        flat_active = torch.cat(gate_active)
        flat_scores = torch.cat(gate_scores)
        flat_errors = torch.cat(unit_errors)
        activated_errors = flat_errors[flat_active]
        metrics: Dict[str, float] = {
            "loss": sum(losses) / max(1, len(losses)),
            "activation_coverage": float(flat_active.float().mean()),
            "mean_gate_score": float(flat_scores.mean()),
            "std_gate_score": float(flat_scores.std(unbiased=False)),
            "fraction_gate_below_0_01": float(flat_scores.lt(0.01).float().mean()),
            "fraction_gate_above_0_99": float(flat_scores.gt(0.99).float().mean()),
            "all_off_rate": sum(all_off) / max(1, len(all_off)),
            "all_on_rate": sum(all_on) / max(1, len(all_on)),
            "activated_error": (
                0.0
                if activated_errors.numel() == 0
                else float(activated_errors.float().mean())
            ),
            "activated_count": float(flat_active.sum()),
            "valid_unit_count": float(flat_active.numel()),
            "threshold": self.threshold,
        }
        if self.task == "mre":
            metrics["micro_f1"] = micro_f1(predictions, labels)
            metrics["accuracy"] = sum(
                a == b for a, b in zip(predictions, labels)
            ) / max(1, len(labels))
        else:
            metrics["strict_entity_f1"] = strict_span_f1(
                predicted_spans,
                gold_spans,
            )
        if artifact_name is not None:
            prediction_path = self.output_dir / "{}_predictions.jsonl".format(
                artifact_name
            )
            with prediction_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True))
                    handle.write("\n")
            write_json(
                self.output_dir / "{}_metrics.json".format(artifact_name),
                metrics,
            )
        return metrics

    @torch.no_grad()
    def calibrate(
        self,
        loader: Iterable[Mapping[str, Any]],
    ) -> Dict[str, float]:
        """Calibrate on forced-vision errors using the frozen learned scores."""

        self.model.eval()
        score_chunks: List[torch.Tensor] = []
        harmful_chunks: List[torch.Tensor] = []
        valid_chunks: List[torch.Tensor] = []
        for host_batch in loader:
            batch = move_batch_to_device(host_batch, self.device)
            inputs = model_inputs_from_batch(batch, self.task)
            with self._autocast():
                output: SaverOutput = self.model(
                    **inputs,
                    routing_mode="vision",
                )
            if self.task == "mre":
                harmful = output.visual_logits.argmax(-1).ne(batch["labels"])
                valid = torch.ones_like(harmful, dtype=torch.bool)
            else:
                predicted = output.visual_logits.argmax(-1)
                harmful = predicted.ne(batch["labels"])
                valid = batch["span_mask"].bool() & batch["labels"].ne(-100)
            score_chunks.append(output.gate_probability.detach().reshape(-1).cpu())
            harmful_chunks.append(harmful.detach().reshape(-1).cpu())
            valid_chunks.append(valid.detach().reshape(-1).cpu())

        calibrator = RiskControlledCalibrator(
            target_risk=self.config.target_risk,
            confidence=self.config.calibration_confidence,
            grid_size=self.config.calibration_grid_size,
        )
        result = calibrator.calibrate(
            torch.cat(score_chunks),
            torch.cat(harmful_chunks),
            torch.cat(valid_chunks),
        )
        self.threshold = float(result["threshold"])
        with (self.output_dir / "calibration.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return result

    def save_checkpoint(
        self,
        name: str,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        path = self.output_dir / "{}.pt".format(name)
        payload: Dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "model_config": self.model.config.to_dict(),
            "training_config": asdict(self.config),
            "task": self.task,
            "threshold": self.threshold,
            "global_step": self.global_step,
        }
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, path)
        return path

    def restore_checkpoint(
        self,
        path: Path,
        restore_optimizer: bool = True,
    ) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu")
        self.model.load_state_dict(payload["model_state"], strict=True)
        if restore_optimizer and "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        self.global_step = int(payload.get("global_step", self.global_step))
        self.threshold = float(
            payload.get("threshold", self.model.config.gate_threshold)
        )
        return payload

    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        dev_loader: Iterable[Mapping[str, Any]],
        calibration_loader: Iterable[Mapping[str, Any]],
        test_loader: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        set_reproducible_seed(self.config.seed)
        history: List[Dict[str, Any]] = []
        best_score = float("-inf")
        metric_name = "micro_f1" if self.task == "mre" else "strict_entity_f1"
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            dev_metrics = self.evaluate(dev_loader, routing_mode="soft")
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "dev": dev_metrics,
            }
            history.append(row)
            score = float(dev_metrics[metric_name])
            if not math.isfinite(score):
                raise RuntimeError(
                    "non-finite {} at epoch {}".format(metric_name, epoch)
                )
            if score > best_score:
                best_score = score
                self.save_checkpoint(
                    "best",
                    {"epoch": epoch, "dev": dev_metrics},
                )

        self.save_checkpoint("last", {"history": history})
        best_payload = self.restore_checkpoint(
            self.output_dir / "best.pt",
            restore_optimizer=True,
        )
        calibration = self.calibrate(calibration_loader)
        calibrated_dev = self.evaluate(dev_loader, routing_mode="hard")
        result = {
            "history": history,
            "calibration": calibration,
            "calibrated_dev": calibrated_dev,
            "best_dev_score": best_score,
            "selected_epoch": int(best_payload.get("extra", {}).get("epoch", -1)),
        }
        if test_loader is not None:
            result["test"] = self.evaluate(
                test_loader,
                routing_mode="hard",
                artifact_name="test",
            )
        write_json(self.output_dir / "metrics.json", result)
        self.save_checkpoint(
            "best",
            {
                "epoch": result["selected_epoch"],
                "calibration": calibration,
                "calibrated_dev": calibrated_dev,
            },
        )
        self.save_checkpoint("final", result)
        return result


@torch.no_grad()
def profile_end_to_end_latency(
    model: nn.Module,
    task: str,
    cached_examples: Sequence[Mapping[str, Any]],
    collator: Any,
    device: torch.device,
    threshold: float,
    warmup: int = 100,
    repeats: int = 300,
    passes: int = 3,
) -> Dict[str, Any]:
    """Profile preprocessing + H2D + model while excluding disk I/O.

    Images must already be materialized in ``cached_examples`` by
    ``SaverCollator.preload_images``.  Each timed iteration still performs
    token/image preprocessing, host-to-device transfer, global screening,
    routing, conditional detail encoding, and task prediction.
    """

    model.eval()
    if passes < 1:
        raise ValueError("passes must be positive")

    def one_iteration() -> None:
        host_batch = collator(cached_examples)
        device_batch = move_batch_to_device(host_batch, device)
        inputs = model_inputs_from_batch(device_batch, task)
        model(**inputs, routing_mode="hard", threshold=threshold)

    all_timings: List[float] = []
    pass_summaries: List[Dict[str, float]] = []
    peak_vram = 0.0
    for pass_index in range(passes):
        for _ in range(warmup):
            one_iteration()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings: List[float] = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            one_iteration()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start_time) * 1000.0)
        values = torch.tensor(timings)
        pass_summaries.append(
            {
                "pass": float(pass_index + 1),
                "mean_ms": float(values.mean()),
                "p50_ms": float(torch.quantile(values, 0.50)),
                "p90_ms": float(torch.quantile(values, 0.90)),
            }
        )
        all_timings.extend(timings)
        if device.type == "cuda":
            peak_vram = max(
                peak_vram,
                torch.cuda.max_memory_allocated(device) / (1024.0**2),
            )

    values = torch.tensor(all_timings)
    result = {
        "mean_ms": float(values.mean()),
        "p50_ms": float(torch.quantile(values, 0.50)),
        "p90_ms": float(torch.quantile(values, 0.90)),
        "warmup": float(warmup),
        "repeats": float(repeats),
        "passes": float(passes),
        "batch_size": float(len(cached_examples)),
        "includes": [
            "text_and_image_preprocessing",
            "host_to_device_transfer",
            "modernbert",
            "siglip2_global_screen",
            "gate_and_top1",
            "conditional_siglip2_detail",
            "task_head",
        ],
        "excludes": ["disk_io", "model_loading"],
        "pass_summaries": pass_summaries,
        "timings_ms": all_timings,
    }
    if device.type == "cuda":
        result["peak_vram_mib"] = peak_vram
    return result
