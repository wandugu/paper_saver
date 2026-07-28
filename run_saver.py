"""Command-line entry point for the ModernBERT/SigLIP-2 SAVER path."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
import yaml

from models.saver import SaverConfig, build_saver_model
from modules.saver_trainer import (
    SaverTrainer,
    TrainingConfig,
    profile_end_to_end_latency,
    set_reproducible_seed,
    write_json,
)
from processor.saver_dataset import (
    SaverCollator,
    SaverJsonlDataset,
    load_label_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "profile", "validate-data"],
        default="train",
    )
    parser.add_argument("--checkpoint", help="checkpoint for eval/profile")
    parser.add_argument("--seed", type=int, help="override training seed")
    parser.add_argument("--output-dir", help="override artifact directory")
    return parser.parse_args()


def read_yaml(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, Mapping):
        raise ValueError("configuration root must be a mapping")
    return dict(values)


def build_dataset(
    data_config: Mapping[str, Any],
    split: str,
    task: str,
    label_to_id: Mapping[str, int],
) -> Optional[SaverJsonlDataset]:
    key = "{}_jsonl".format(split)
    path = data_config.get(key)
    if path in {None, "", "to_fill"}:
        return None
    return SaverJsonlDataset(
        str(path),
        task,
        label_to_id,
        image_root=data_config.get("image_root"),
    )


def split_train_calibration(
    dataset: SaverJsonlDataset,
    fraction: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    calibration_size = max(1, round(len(dataset) * fraction))
    train_size = len(dataset) - calibration_size
    if train_size < 1:
        raise ValueError("dataset is too small for a calibration split")
    generator = torch.Generator().manual_seed(seed)
    train, calibration = random_split(
        dataset,
        [train_size, calibration_size],
        generator=generator,
    )
    return train, calibration


def make_loader(
    dataset,
    collator: SaverCollator,
    training: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collator,
    )


def load_checkpoint(model: torch.nn.Module, path: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint


def write_resolved_config(
    raw: Mapping[str, Any],
    model_config: SaverConfig,
    training_config: TrainingConfig,
) -> Path:
    resolved = dict(raw)
    resolved["model"] = model_config.to_dict()
    resolved["training"] = asdict(training_config)
    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resolved_config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            resolved,
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    return path


def main() -> None:
    args = parse_args()
    raw = read_yaml(args.config)
    task = str(raw["task"]).lower()
    model_config = SaverConfig.from_mapping(raw.get("model", {}))
    training_config = TrainingConfig.from_mapping(raw.get("training", {}))
    if args.seed is not None:
        training_config = replace(training_config, seed=args.seed)
    if args.output_dir is not None:
        training_config = replace(
            training_config,
            output_dir=args.output_dir,
        )
    set_reproducible_seed(training_config.seed)

    data_config = dict(raw["data"])
    label_to_id = load_label_map(str(data_config["label_map"]), task)
    train_dataset = build_dataset(
        data_config,
        "train",
        task,
        label_to_id,
    )
    dev_dataset = build_dataset(data_config, "dev", task, label_to_id)
    test_dataset = build_dataset(data_config, "test", task, label_to_id)
    calibration_dataset = build_dataset(
        data_config,
        "calibration",
        task,
        label_to_id,
    )
    if args.mode == "validate-data":
        sizes = {
            "train": 0 if train_dataset is None else len(train_dataset),
            "dev": 0 if dev_dataset is None else len(dev_dataset),
            "test": 0 if test_dataset is None else len(test_dataset),
            "calibration": (
                0 if calibration_dataset is None else len(calibration_dataset)
            ),
        }
        print(json.dumps({"task": task, "sizes": sizes}, indent=2))
        return
    if args.mode in {"eval", "profile"} and not args.checkpoint:
        raise ValueError("{} mode requires --checkpoint".format(args.mode))
    write_resolved_config(raw, model_config, training_config)

    try:
        from transformers import AutoImageProcessor, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install requirements-saver.txt before running SAVER"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_config.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(model_config.vision_model_name)
    collator = SaverCollator(
        tokenizer,
        image_processor,
        task,
        max_length=int(data_config.get("max_length", 128)),
        max_span_width=model_config.max_span_width,
        screen_max_num_patches=model_config.screen_max_num_patches,
        detail_max_num_patches=model_config.detail_max_num_patches,
    )
    num_workers = int(data_config.get("num_workers", 4))

    if calibration_dataset is None and train_dataset is not None:
        train_data, calibration_data = split_train_calibration(
            train_dataset,
            float(data_config.get("calibration_fraction", 0.10)),
            training_config.seed,
        )
    else:
        train_data = train_dataset
        calibration_data = calibration_dataset

    pretrained_kwargs: Dict[str, Any] = {}
    if torch.cuda.is_available() and training_config.bf16:
        pretrained_kwargs["torch_dtype"] = torch.bfloat16
    model = build_saver_model(
        task,
        model_config,
        num_labels=len(label_to_id),
        **pretrained_kwargs,
    )
    checkpoint = None
    if args.checkpoint:
        checkpoint = load_checkpoint(model, args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = SaverTrainer(model, task, training_config, device=device)
    if checkpoint is not None:
        trainer.threshold = float(
            checkpoint.get("threshold", model_config.gate_threshold)
        )

    batch_size = training_config.batch_size
    if args.mode == "train":
        if train_data is None or dev_dataset is None or calibration_data is None:
            raise ValueError("train mode requires train, dev, and calibration data")
        result = trainer.fit(
            make_loader(
                train_data,
                collator,
                True,
                batch_size,
                num_workers,
            ),
            make_loader(
                dev_dataset,
                collator,
                False,
                batch_size,
                num_workers,
            ),
            make_loader(
                calibration_data,
                collator,
                False,
                batch_size,
                num_workers,
            ),
            (
                None
                if test_dataset is None
                else make_loader(
                    test_dataset,
                    collator,
                    False,
                    batch_size,
                    num_workers,
                )
            ),
        )
    elif args.mode == "eval":
        if test_dataset is None:
            raise ValueError("eval mode requires test_jsonl")
        if calibration_data is not None and args.checkpoint:
            trainer.calibrate(
                make_loader(
                    calibration_data,
                    collator,
                    False,
                    batch_size,
                    num_workers,
                )
            )
        result = trainer.evaluate(
            make_loader(
                test_dataset,
                collator,
                False,
                batch_size,
                num_workers,
            ),
            artifact_name="test",
        )
    else:
        profile_dataset = test_dataset or dev_dataset
        if profile_dataset is None:
            raise ValueError("profile mode requires test_jsonl or dev_jsonl")
        cached_examples = collator.preload_images([profile_dataset[0]])
        runtime = dict(raw.get("runtime", {}))
        result = profile_end_to_end_latency(
            model,
            task,
            cached_examples,
            collator,
            device,
            trainer.threshold,
            warmup=int(runtime.get("warmup", 100)),
            repeats=int(runtime.get("repeats", 300)),
            passes=int(runtime.get("passes", 3)),
        )
        write_json(trainer.output_dir / "latency.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
