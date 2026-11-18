"""Frozen, local-only ModernBERT and SigLIP encoders for the P24 runs.

Word spans are half-open in the original record's ``tokens`` (or ``token``).
The task head owns width embeddings, projections, labels, and normalization.
NaFlex detail padding is zeroed; attention consumers should request its mask.
There is deliberately no feature, image, or checkpoint cache in this module.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import pairwise
from os import PathLike
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    Siglip2VisionModel,
    SiglipVisionModel,
)

ImageSource = str | PathLike[str] | Image.Image


class TextAlignmentError(ValueError):
    """A record must be explicitly excluded rather than encoded incorrectly."""


class FrozenEncoders:
    """Load only pinned local pretrained weights, never an old task checkpoint.

    ``encode_text`` returns ``head: [U, 3 * text_hidden]`` and ``head_width``.
    MRE also has ``tail`` and ``tail_width`` with U=1. MNER has ``spans`` in
    original word coordinates. All task features use first-subtoken word
    states. Width embeddings are intentionally left to the trainable head.
    Set ``return_units=False`` to cache compact ``word_states`` and aggregate
    spans later with ``span_features``, avoiding the expanded MNER cache.

    Vision returns the pretrained pooling head for the global screen, or a
    correctly masked mean only when that head is absent. Detail returns
    last-hidden-state patches, not global embeddings. No output is normalized
    or transferred to CPU implicitly; callers may cache detached CPU float16.
    """

    global_patches = 64
    detail_patches = 256
    max_span_width = 10

    def __init__(
        self,
        text_path: str | PathLike[str],
        vision_path: str | PathLike[str],
        device: str | torch.device = "cuda",
        *,
        max_length: int = 128,
    ) -> None:
        self.device = torch.device(device)
        self.max_length = int(max_length)
        if self.max_length < 3:
            raise ValueError("max_length must accommodate words and special tokens")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(text_path), use_fast=True, local_files_only=True
        )
        if not self.tokenizer.is_fast:
            raise ValueError("ModernBERT requires a fast tokenizer with word alignment")
        text_kwargs = {
            "local_files_only": True,
            "torch_dtype": self.dtype,
            "attn_implementation": "sdpa",
        }
        try:
            self.text_model = AutoModel.from_pretrained(str(text_path), reference_compile=False, **text_kwargs)
        except TypeError as error:
            if "reference_compile" not in str(error):
                raise
            self.text_model = AutoModel.from_pretrained(str(text_path), **text_kwargs)
        if self.text_model.config.model_type != "modernbert":
            raise ValueError("The text backbone must be ModernBERT, not an old checkpoint")
        self.text_hidden = int(self.text_model.config.hidden_size)
        self.text_model.to(self.device).eval().requires_grad_(False)
        vision_config = AutoConfig.from_pretrained(str(vision_path), local_files_only=True)
        self.is_naflex = vision_config.model_type in {"siglip2", "siglip2_vision_model"}
        if self.is_naflex:
            vision_class = Siglip2VisionModel
        elif vision_config.model_type in {"siglip", "siglip_vision_model"}:
            vision_class = SiglipVisionModel
        else:
            raise ValueError(f"Unsupported SigLIP backbone: {vision_config.model_type}")
        self.vision_model = vision_class.from_pretrained(
            str(vision_path),
            local_files_only=True,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",
        )
        self.vision_model.to(self.device).eval().requires_grad_(False)
        self.vision_hidden = int(self.vision_model.config.hidden_size)
        self.vision_patch_size = int(self.vision_model.config.patch_size)
        if not self.is_naflex and self.vision_patch_size != 16:
            raise ValueError("The fixed-resolution substitute must use 16-pixel patches")
        self.image_processor = AutoImageProcessor.from_pretrained(
            str(vision_path), local_files_only=True, use_fast=False
        )

    @staticmethod
    def _words(record: Mapping[str, Any]) -> list[str]:
        words = record.get("tokens", record.get("token", record.get("words")))
        if isinstance(words, (str, bytes)) or not isinstance(words, Sequence):
            raise TextAlignmentError("Record needs a pre-tokenized word sequence")
        if not words or any(not isinstance(word, str) or not word.strip() for word in words):
            raise TextAlignmentError("Empty records or empty/non-string words cannot align")
        return list(words)

    @staticmethod
    def _entity_span(record: Mapping[str, Any], name: str, word_count: int) -> tuple[int, int]:
        value = record.get(f"{name}_span", record.get(name, record.get(name[0])))
        if isinstance(value, Mapping):
            value = value.get("pos", value.get("span"))
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise TextAlignmentError(f"Missing half-open {name} word span")
        start, end = value
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= word_count
        ):
            raise TextAlignmentError(f"Invalid half-open {name} word span: {value!r}")
        return start, end

    def _tokenize(self, words: list[str]) -> Any:
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        word_ids = encoded.word_ids(batch_index=0)
        present = {word_id for word_id in word_ids if word_id is not None}
        if present != set(range(len(words))):
            raise TextAlignmentError("Tokenizer omitted words; refusing incomplete span alignment")
        aligned_ids = [word_id for word_id in word_ids if word_id is not None]
        if any(previous > following for previous, following in pairwise(aligned_ids)):
            raise TextAlignmentError("Tokenizer word alignment must be monotone")
        return encoded

    def _aligned_inputs(
        self, words: list[str], required_spans: list[tuple[int, int]]
    ) -> tuple[Any, int, int]:
        encoded = self._tokenize(words)
        if encoded["input_ids"].shape[1] <= self.max_length:
            return encoded, 0, len(words)
        if not required_spans:
            raise TextAlignmentError(
                f"MNER exceeds max_length={self.max_length}; exclude record explicitly, "
                "rather than silently discarding candidate or gold spans"
            )
        word_ids = encoded.word_ids(batch_index=0)
        counts = [0] * len(words)
        for word_id in word_ids:
            if word_id is not None:
                counts[word_id] += 1
        budget = self.max_length - sum(word_id is None for word_id in word_ids)
        start = min(span[0] for span in required_spans)
        end = max(span[1] for span in required_spans)
        used = sum(counts[start:end])
        if used > budget:
            raise TextAlignmentError(
                f"Head and tail cannot both fit max_length={self.max_length}; "
                "exclude record explicitly"
            )
        while True:
            changed = False
            if start > 0 and used + counts[start - 1] <= budget:
                start -= 1
                used += counts[start]
                changed = True
            if end < len(words) and used + counts[end] <= budget:
                used += counts[end]
                end += 1
                changed = True
            if not changed:
                break
        encoded = self._tokenize(words[start:end])
        if encoded["input_ids"].shape[1] > self.max_length:
            raise TextAlignmentError("Retokenized entity-preserving window exceeds max_length")
        return encoded, start, end

    @staticmethod
    @torch.no_grad()
    def span_features(
        word_states: torch.Tensor, spans: torch.Tensor, word_offset: int = 0
    ) -> torch.Tensor:
        """Concatenate first/last/max word states without trainable embeddings."""
        if word_states.ndim != 2 or spans.ndim != 2 or spans.shape[1] != 2:
            raise ValueError("Expected word_states [T,H] and spans [U,2]")
        if spans.dtype not in (torch.int32, torch.int64):
            raise ValueError("Span boundaries must be integers")
        spans = spans.to(device=word_states.device, dtype=torch.long) - int(word_offset)
        if spans.shape[0] == 0:
            return word_states.new_empty((0, 3 * word_states.shape[1]))
        starts, ends = spans.unbind(dim=1)
        widths = ends - starts
        if bool(((starts < 0) | (ends > word_states.shape[0]) | (widths <= 0)).any()):
            raise TextAlignmentError("Span falls outside fully aligned word states")
        pooled = word_states.new_empty((spans.shape[0], word_states.shape[1]))
        for width in widths.unique().tolist():
            selected = widths == width
            windows = word_states.unfold(0, width, 1).amax(dim=-1)
            pooled[selected] = windows.index_select(0, starts[selected])
        return torch.cat(
            (word_states.index_select(0, starts), word_states.index_select(0, ends - 1), pooled),
            dim=1,
        )

    @torch.no_grad()
    def encode_text(
        self,
        record: Mapping[str, Any],
        task: str | None = None,
        *,
        return_units: bool = True,
    ) -> dict[str, Any]:
        words = self._words(record)
        if task is None:
            task = record.get("task") or (
                "mre" if any(key in record for key in ("h", "head", "head_span")) else "mner"
            )
        task = task.lower()
        if task not in {"mre", "mner"}:
            raise ValueError(f"Unknown task: {task!r}")
        required = []
        if task == "mre":
            required = [self._entity_span(record, name, len(words)) for name in ("head", "tail")]
        encoded, word_start, word_end = self._aligned_inputs(words, required)
        word_ids = encoded.word_ids(batch_index=0)
        inputs = {
            name: encoded[name].to(self.device)
            for name in ("input_ids", "attention_mask")
            if name in encoded
        }
        states = self.text_model(**inputs).last_hidden_state[0]
        first_positions = []
        for position, word_id in enumerate(word_ids):
            if word_id is not None and word_id == len(first_positions):
                first_positions.append(position)
        word_states = states.index_select(
            0, torch.tensor(first_positions, dtype=torch.long, device=self.device)
        ).detach()
        result: dict[str, Any] = {
            "word_states": word_states,
            "word_offsets": torch.tensor(
                [word_start, word_end], dtype=torch.long, device=self.device
            ),
            "word_pooling": "first_subtoken",
            "truncated": word_start != 0 or word_end != len(words),
        }
        if task == "mre":
            for name, span in zip(("head", "tail"), required):
                span_tensor = torch.tensor([span], dtype=torch.long, device=self.device)
                result[f"{name}_span"] = span_tensor
                result[f"{name}_width"] = span_tensor[:, 1] - span_tensor[:, 0]
                if return_units:
                    result[name] = self.span_features(word_states, span_tensor, word_start)
        else:
            spans = [
                (start, end)
                for start in range(len(words))
                for end in range(start + 1, min(len(words), start + self.max_span_width) + 1)
            ]
            span_tensor = torch.tensor(spans, dtype=torch.long, device=self.device)
            result["spans"] = span_tensor
            result["head_span"] = span_tensor
            result["head_width"] = span_tensor[:, 1] - span_tensor[:, 0]
            if return_units:
                result["head"] = self.span_features(word_states, span_tensor, word_start)
        return result

    @staticmethod
    @contextmanager
    def _rgb_images(image_sources: Sequence[ImageSource]) -> Iterator[list[Image.Image]]:
        images = []
        owned = []
        try:
            for source in image_sources:
                if isinstance(source, Image.Image):
                    image = source if source.mode == "RGB" else source.convert("RGB")
                    if image is not source:
                        owned.append(image)
                else:
                    with Image.open(source) as original:
                        image = original.convert("RGB")
                    owned.append(image)
                images.append(image)
            yield images
        finally:
            for image in owned:
                image.close()

    def _vision_inputs(
        self, images: list[Image.Image], patch_count: int
    ) -> dict[str, torch.Tensor]:
        if self.is_naflex:
            inputs = self.image_processor(
                images=images, max_num_patches=patch_count, return_tensors="pt"
            )
            mask = inputs.get("pixel_attention_mask")
            if mask is None or tuple(mask.shape) != (len(images), patch_count):
                raise ValueError("NaFlex processor must return a fixed-length patch attention mask")
            if tuple(inputs["pixel_values"].shape[:2]) != (len(images), patch_count):
                raise ValueError("NaFlex processor did not pad to the requested patch count")
            if not bool(mask.bool().any(dim=1).all()):
                raise ValueError("NaFlex image has no valid visual patches")
        else:
            side = 128 if patch_count == self.global_patches else 256
            inputs = self.image_processor(
                images=images, size={"height": side, "width": side}, return_tensors="pt"
            )
        return {
            name: value.to(
                device=self.device, dtype=self.dtype if value.is_floating_point() else value.dtype
            )
            for name, value in inputs.items()
            if name in {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
        }

    def _vision_forward(self, inputs: dict[str, torch.Tensor]) -> Any:
        if self.is_naflex:
            return self.vision_model(**inputs)
        return self.vision_model(**inputs, interpolate_pos_encoding=True)

    @torch.no_grad()
    def encode_global(self, image_paths: Sequence[ImageSource]) -> torch.Tensor:
        """Screen every supplied attachment at 64 patches, with no internal cache."""
        if not image_paths:
            return torch.empty((0, self.vision_hidden), dtype=self.dtype, device=self.device)
        if isinstance(image_paths, (str, bytes, PathLike, Image.Image)):
            raise TypeError("encode_global expects a sequence of images, not a single image")
        with self._rgb_images(image_paths) as images:
            inputs = self._vision_inputs(images, self.global_patches)
            outputs = self._vision_forward(inputs)
        pooled = outputs.pooler_output
        if pooled is None:
            patches = outputs.last_hidden_state
            if self.is_naflex:
                mask = inputs["pixel_attention_mask"].bool().unsqueeze(-1)
                pooled = patches.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = patches.mean(dim=1)
        if tuple(pooled.shape) != (len(image_paths), self.vision_hidden):
            raise ValueError("Vision backbone returned an unexpected global feature shape")
        return pooled.detach()

    @torch.no_grad()
    def encode_detail(
        self, image_path: ImageSource, *, return_mask: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode 256 detail patches; padded rows are exactly zero.

        ``return_mask=True`` also returns a bool [256] validity mask; use it
        before patch Top-R selection and softmax. Without that option a caller
        can recover padding as ``patches.ne(0).any(dim=-1)`` before caching.
        A supplied PIL RGB image bypasses disk I/O and remains caller-owned.
        """
        with self._rgb_images([image_path]) as images:
            inputs = self._vision_inputs(images, self.detail_patches)
            patches = self._vision_forward(inputs).last_hidden_state[0]
        if tuple(patches.shape) != (self.detail_patches, self.vision_hidden):
            raise ValueError("Vision backbone returned an unexpected detail patch shape")
        mask = (
            inputs["pixel_attention_mask"][0].bool()
            if self.is_naflex
            else torch.ones(self.detail_patches, dtype=torch.bool, device=self.device)
        )
        patches = patches.masked_fill(~mask.unsqueeze(-1), 0).detach()
        return (patches, mask) if return_mask else patches
