"""Data contract and batching for the ModernBERT/SigLIP-2 SAVER path.

The loader intentionally uses a small JSONL interchange format so MNRE,
MRE-MI, Twitter-2015/2017, MNER-MI, and MNER-MI-Plus can share the same model
code.  Offsets are zero-based and half-open at word level.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EntityAnnotation:
    start: int
    end: int
    label: str
    label_id: int


class SaverJsonlDataset(Dataset):
    """Validated JSONL dataset for MRE or span-level MNER."""

    def __init__(
        self,
        path: str,
        task: str,
        label_to_id: Mapping[str, int],
        image_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.path = Path(path).resolve()
        self.task = task.lower()
        if self.task not in {"mre", "mner"}:
            raise ValueError("task must be 'mre' or 'mner'")
        self.label_to_id = dict(label_to_id)
        if self.task == "mner" and self.label_to_id.get("NONE") != 0:
            raise ValueError("MNER label_to_id must reserve NONE=0")
        self.image_root = (
            Path(image_root).resolve() if image_root is not None else self.path.parent
        )
        self.examples = self._load_examples()

    def _load_examples(self) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "{}:{} is not valid JSON".format(self.path, line_number)
                    ) from exc
                examples.append(self._validate_example(example, line_number))
        if not examples:
            raise ValueError("{} contains no examples".format(self.path))
        return examples

    @staticmethod
    def _span(
        value: Any,
        name: str,
        token_count: int,
        line_number: int,
    ) -> Tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                "line {}: {} must be [start, end]".format(line_number, name)
            )
        start, end = int(value[0]), int(value[1])
        if not 0 <= start < end <= token_count:
            raise ValueError(
                "line {}: invalid {} span [{}, {}) for {} words".format(
                    line_number,
                    name,
                    start,
                    end,
                    token_count,
                )
            )
        return start, end

    def _validate_example(
        self,
        example: Mapping[str, Any],
        line_number: int,
    ) -> Dict[str, Any]:
        sample_id = str(example.get("id", line_number - 1))
        tokens = example.get("tokens")
        if (
            not isinstance(tokens, list)
            or not tokens
            or not all(isinstance(token, str) and token for token in tokens)
        ):
            raise ValueError(
                "line {}: tokens must be a nonempty string list".format(line_number)
            )
        images = example.get("images")
        if (
            not isinstance(images, list)
            or not images
            or not all(isinstance(path, str) and path for path in images)
        ):
            raise ValueError(
                "line {}: images must be a nonempty path list".format(line_number)
            )
        resolved_images = []
        for image_path in images:
            candidate = Path(image_path)
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (self.image_root / candidate).resolve()
            )
            resolved_images.append(str(resolved))

        normalized: Dict[str, Any] = {
            "id": sample_id,
            "tokens": list(tokens),
            "images": resolved_images,
        }
        if self.task == "mre":
            relation = str(example.get("relation"))
            if relation not in self.label_to_id:
                raise ValueError(
                    "line {}: unknown relation {!r}".format(line_number, relation)
                )
            normalized.update(
                {
                    "head": self._span(
                        example.get("head"),
                        "head",
                        len(tokens),
                        line_number,
                    ),
                    "tail": self._span(
                        example.get("tail"),
                        "tail",
                        len(tokens),
                        line_number,
                    ),
                    "label": self.label_to_id[relation],
                    "label_name": relation,
                }
            )
        else:
            raw_entities = example.get("entities", [])
            if not isinstance(raw_entities, list):
                raise ValueError("line {}: entities must be a list".format(line_number))
            entities: List[EntityAnnotation] = []
            occupied: Dict[Tuple[int, int], str] = {}
            for index, entity in enumerate(raw_entities):
                if not isinstance(entity, Mapping):
                    raise ValueError(
                        "line {}: entity {} must be an object".format(
                            line_number,
                            index,
                        )
                    )
                label = str(entity.get("type"))
                if label == "NONE" or label not in self.label_to_id:
                    raise ValueError(
                        "line {}: invalid entity type {!r}".format(
                            line_number,
                            label,
                        )
                    )
                start, end = self._span(
                    [entity.get("start"), entity.get("end")],
                    "entity",
                    len(tokens),
                    line_number,
                )
                existing = occupied.get((start, end))
                if existing is not None:
                    raise ValueError(
                        "line {}: duplicate entity span [{}, {})".format(
                            line_number,
                            start,
                            end,
                        )
                    )
                if any(start < other.end and other.start < end for other in entities):
                    raise ValueError(
                        "line {}: overlapping entities are incompatible with "
                        "non-overlapping decoding".format(line_number)
                    )
                occupied[(start, end)] = label
                entities.append(
                    EntityAnnotation(
                        start,
                        end,
                        label,
                        self.label_to_id[label],
                    )
                )
            normalized["entities"] = entities
        return normalized

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]


def load_label_map(path: str, task: str) -> Dict[str, int]:
    """Load a JSON object mapping string labels to contiguous integer ids."""

    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, Mapping):
        raise ValueError("label map must be a JSON object")
    result = {str(label): int(index) for label, index in values.items()}
    expected = list(range(len(result)))
    if sorted(result.values()) != expected:
        raise ValueError("label ids must be contiguous from zero")
    if task.lower() == "mner" and result.get("NONE") != 0:
        raise ValueError("MNER label map must contain NONE: 0")
    return result


class SaverCollator:
    """Tokenize text and produce low/high-patch SigLIP-2 image batches."""

    def __init__(
        self,
        tokenizer: Any,
        image_processor: Any,
        task: str,
        max_length: int = 128,
        max_span_width: int = 10,
        screen_max_num_patches: int = 64,
        detail_max_num_patches: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.task = task.lower()
        if self.task not in {"mre", "mner"}:
            raise ValueError("task must be 'mre' or 'mner'")
        self.max_length = max_length
        self.max_span_width = max_span_width
        self.screen_max_num_patches = screen_max_num_patches
        self.detail_max_num_patches = detail_max_num_patches

    @staticmethod
    def _open_images(paths: Iterable[str]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for image_path in paths:
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError("attached image not found: {}".format(path))
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        return images

    @staticmethod
    def _word_to_token_spans(
        word_ids: Sequence[Optional[int]],
    ) -> Dict[int, Tuple[int, int]]:
        result: Dict[int, List[int]] = {}
        for token_index, word_index in enumerate(word_ids):
            if word_index is None:
                continue
            if word_index not in result:
                result[word_index] = [token_index, token_index + 1]
            else:
                result[word_index][1] = token_index + 1
        return {
            word_index: (bounds[0], bounds[1]) for word_index, bounds in result.items()
        }

    @staticmethod
    def _map_word_span(
        word_span: Tuple[int, int],
        mapping: Mapping[int, Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        start, end = word_span
        if start not in mapping or end - 1 not in mapping:
            return None
        return mapping[start][0], mapping[end - 1][1]

    def _tokenize(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[int, Tuple[int, int]]]]:
        encoded = self.tokenizer(
            [example["tokens"] for example in examples],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        mappings = [
            self._word_to_token_spans(encoded.word_ids(batch_index=index))
            for index in range(len(examples))
        ]
        tensor_values = {
            key: value
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        return tensor_values, mappings

    def _mre_targets(
        self,
        examples: Sequence[Mapping[str, Any]],
        mappings: Sequence[Mapping[int, Tuple[int, int]]],
    ) -> Dict[str, torch.Tensor]:
        heads: List[Tuple[int, int]] = []
        tails: List[Tuple[int, int]] = []
        labels: List[int] = []
        for example, mapping in zip(examples, mappings):
            head = self._map_word_span(example["head"], mapping)
            tail = self._map_word_span(example["tail"], mapping)
            if head is None or tail is None:
                raise ValueError(
                    "entity marker was truncated for sample {}; increase max_length".format(
                        example["id"]
                    )
                )
            heads.append(head)
            tails.append(tail)
            labels.append(int(example["label"]))
        return {
            "head_spans": torch.tensor(heads, dtype=torch.long),
            "tail_spans": torch.tensor(tails, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _mner_targets(
        self,
        examples: Sequence[Mapping[str, Any]],
        mappings: Sequence[Mapping[int, Tuple[int, int]]],
    ) -> Dict[str, torch.Tensor]:
        per_example: List[List[Tuple[Tuple[int, int], Tuple[int, int], int, int]]] = []
        for example, mapping in zip(examples, mappings):
            gold = {
                (entity.start, entity.end): entity.label_id
                for entity in example["entities"]
            }
            candidates: List[Tuple[Tuple[int, int], Tuple[int, int], int, int]] = []
            visible_words = sorted(mapping)
            visible_count = visible_words[-1] + 1 if visible_words else 0
            for start in range(visible_count):
                if start not in mapping:
                    continue
                upper = min(visible_count, start + self.max_span_width)
                for end in range(start + 1, upper + 1):
                    token_span = self._map_word_span((start, end), mapping)
                    if token_span is None:
                        continue
                    label_id = int(gold.get((start, end), 0))
                    candidates.append((token_span, (start, end), end - start, label_id))
            represented = {item[1] for item in candidates}
            missing_gold = sorted(set(gold) - represented)
            if missing_gold:
                raise ValueError(
                    "gold MNER spans {} are truncated or wider than "
                    "max_span_width={} for sample {}".format(
                        missing_gold,
                        self.max_span_width,
                        example["id"],
                    )
                )
            per_example.append(candidates)

        max_spans = max((len(candidates) for candidates in per_example), default=1)
        batch_size = len(examples)
        spans = torch.zeros(batch_size, max_spans, 2, dtype=torch.long)
        word_spans = torch.zeros(batch_size, max_spans, 2, dtype=torch.long)
        widths = torch.ones(batch_size, max_spans, dtype=torch.long)
        mask = torch.zeros(batch_size, max_spans, dtype=torch.bool)
        labels = torch.full((batch_size, max_spans), -100, dtype=torch.long)
        for batch_index, candidates in enumerate(per_example):
            for span_index, (token_span, word_span, width, label) in enumerate(
                candidates
            ):
                spans[batch_index, span_index] = torch.tensor(token_span)
                word_spans[batch_index, span_index] = torch.tensor(word_span)
                widths[batch_index, span_index] = width
                mask[batch_index, span_index] = True
                labels[batch_index, span_index] = label
        return {
            "spans": spans,
            "word_spans": word_spans,
            "span_widths": widths,
            "span_mask": mask,
            "labels": labels,
        }

    @staticmethod
    def _pack_image_features(
        encoded: Mapping[str, torch.Tensor],
        image_counts: Sequence[int],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        required = {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
        missing = sorted(required - set(encoded))
        if missing:
            raise ValueError(
                "SigLIP-2 NaFlex processor omitted: {}".format(", ".join(missing))
            )
        batch_size = len(image_counts)
        max_images = max(image_counts)
        total_images = sum(image_counts)
        if encoded["pixel_values"].size(0) != total_images:
            raise ValueError("processor image count mismatch")
        packed: Dict[str, torch.Tensor] = {}
        for key in required:
            value = encoded[key]
            destination = value.new_zeros(
                batch_size,
                max_images,
                *value.shape[1:],
            )
            cursor = 0
            for batch_index, count in enumerate(image_counts):
                destination[batch_index, :count] = value[cursor : cursor + count]
                cursor += count
            packed[key] = destination
        image_mask = torch.zeros(batch_size, max_images, dtype=torch.bool)
        for batch_index, count in enumerate(image_counts):
            image_mask[batch_index, :count] = True
        return packed, image_mask

    def _images(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        image_counts = [len(example["images"]) for example in examples]
        images: List[Image.Image] = []
        for example in examples:
            cached = example.get("_preloaded_images")
            if cached is None:
                images.extend(self._open_images(example["images"]))
            else:
                if len(cached) != len(example["images"]):
                    raise ValueError("preloaded image count does not match paths")
                images.extend(image.copy() for image in cached)
        screen = self.image_processor(
            images=images,
            max_num_patches=self.screen_max_num_patches,
            return_tensors="pt",
        )
        detail = self.image_processor(
            images=images,
            max_num_patches=self.detail_max_num_patches,
            return_tensors="pt",
        )
        screen_batch, image_mask = self._pack_image_features(
            screen,
            image_counts,
        )
        detail_batch, detail_mask = self._pack_image_features(
            detail,
            image_counts,
        )
        if not torch.equal(image_mask, detail_mask):
            raise RuntimeError("screen/detail image masks disagree")
        return screen_batch, detail_batch, image_mask

    def preload_images(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Load image bytes once so latency timing can exclude disk I/O."""

        cached_examples: List[Dict[str, Any]] = []
        for example in examples:
            copied = dict(example)
            copied["_preloaded_images"] = self._open_images(example["images"])
            cached_examples.append(copied)
        return cached_examples

    def __call__(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        text, mappings = self._tokenize(examples)
        global_inputs, detail_inputs, image_mask = self._images(examples)
        result: Dict[str, Any] = {
            **text,
            "global_inputs": global_inputs,
            "detail_inputs": detail_inputs,
            "image_mask": image_mask,
            "sample_ids": [str(example["id"]) for example in examples],
            "image_paths": [list(example["images"]) for example in examples],
        }
        if self.task == "mre":
            result.update(self._mre_targets(examples, mappings))
        else:
            result.update(self._mner_targets(examples, mappings))
        return result


def move_batch_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    """Recursively move tensors while leaving ids and paths on the host."""

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_batch_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_batch_to_device(item, device) for item in value]
    return value


def model_inputs_from_batch(
    batch: Mapping[str, Any],
    task: str,
) -> Dict[str, Any]:
    """Drop host-only metadata and retain arguments accepted by a task head."""

    common = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "global_inputs": batch["global_inputs"],
        "detail_inputs": batch["detail_inputs"],
        "image_mask": batch["image_mask"],
        "labels": batch.get("labels"),
    }
    if "token_type_ids" in batch:
        common["token_type_ids"] = batch["token_type_ids"]
    if task.lower() == "mre":
        common.update(
            {
                "head_spans": batch["head_spans"],
                "tail_spans": batch["tail_spans"],
            }
        )
    elif task.lower() == "mner":
        common.update(
            {
                "spans": batch["spans"],
                "span_mask": batch["span_mask"],
                "span_widths": batch["span_widths"],
            }
        )
    else:
        raise ValueError("task must be 'mre' or 'mner'")
    return common
