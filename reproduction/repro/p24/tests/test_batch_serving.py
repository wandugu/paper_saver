"""Isolated P24 E7 CPU tests: fake and local-pretrained batched backbones."""

import json
import math
import sys
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from repro.p24 import batch_serving as serving
from repro.p24.encoders import FrozenEncoders, TextAlignmentError
from repro.p24.model import SaverCore


class Encoding(dict):
    def __init__(self, words):
        token_ids, self.alignment = [101], [None]
        for index, word in enumerate(words):
            pieces = 2 if word.startswith("multi") else 1
            token_ids.extend([sum(map(ord, word)) % 71 + 1] * pieces)
            self.alignment.extend([index] * pieces)
        token_ids.append(102)
        self.alignment.append(None)
        super().__init__(
            input_ids=torch.tensor([token_ids]),
            attention_mask=torch.ones(1, len(token_ids), dtype=torch.long),
        )

    def word_ids(self, batch_index=0):
        assert batch_index == 0
        return self.alignment


class Tokenizer:
    pad_token_id = 0

    def __call__(self, words, **kwargs):
        assert kwargs["is_split_into_words"]
        return Encoding(words)


class TextBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.config = SimpleNamespace(model_type="modernbert")

    def forward(self, input_ids, attention_mask):
        assert not torch.is_grad_enabled()
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        offsets = torch.arange(4, dtype=torch.float32)
        states = torch.sin(input_ids[..., None].float() + offsets)
        return SimpleNamespace(last_hidden_state=states)


class ImageProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, images, max_num_patches, return_tensors):
        assert return_tensors == "pt"
        self.calls.append((len(images), max_num_patches))
        values = torch.tensor([image.getpixel((0, 0))[0] for image in images]).float()
        mask = torch.ones(len(images), max_num_patches, dtype=torch.long)
        mask[:, -2:] = 0
        return {
            "pixel_values": values[:, None, None].expand(-1, max_num_patches, 2),
            "pixel_attention_mask": mask,
            "spatial_shapes": torch.tensor([[1, max_num_patches - 2]] * len(images)),
        }


class VisionBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.pooler = True

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes):
        assert not torch.is_grad_enabled()
        self.calls.append(tuple(pixel_values.shape[:2]))
        positions = torch.arange(pixel_values.shape[1], dtype=torch.float32)[None, :, None]
        channels = torch.arange(6, dtype=torch.float32)[None, None, :]
        patches = torch.cos(pixel_values[:, :, :1] + positions / 100 + channels)
        pooled = patches[:, 0].clone() if self.pooler else None
        patches = patches.masked_fill(~pixel_attention_mask.bool()[..., None], 10000)
        return SimpleNamespace(last_hidden_state=patches, pooler_output=pooled)


class FixedGate(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, interaction):
        if isinstance(self.logits, list):
            assert len(interaction) == len(self.logits)
            return interaction.new_tensor(self.logits)[:, None]
        return interaction.new_full((len(interaction), 1), self.logits)


@pytest.fixture
def encoders():
    encoder = FrozenEncoders.__new__(FrozenEncoders)
    encoder.device, encoder.dtype = torch.device("cpu"), torch.float32
    encoder.max_length = 128
    encoder.text_hidden, encoder.vision_hidden = 4, 6
    encoder.is_naflex = True
    encoder.tokenizer = Tokenizer()
    encoder.text_model = TextBackbone().eval()
    encoder.vision_model = VisionBackbone().eval()
    encoder.image_processor = ImageProcessor()
    return encoder


@pytest.fixture
def model():
    torch.manual_seed(42)
    return SaverCore(
        3, text_dim=4, vision_dim=6, hidden_dim=8, query_dim=4, width_dim=2, num_heads=2, dropout=0
    ).eval()


def record(index, *, post=None, pair=0, attachments=1, split="dev", extra_words=0):
    post = index if post is None else post
    return {
        "id": f"record-{index}",
        "split": split,
        "tokens": [f"post{post}", "multiword", "tail"] + ["context"] * extra_words,
        "head": [pair % 2, pair % 2 + 1],
        "tail": [2, 3],
        "images": [f"post{post}-image{image}.png" for image in range(attachments)],
    }


@pytest.fixture
def population():
    records = [
        record(0, post=0),
        record(1, post=0, pair=1),
        record(2, post=1, attachments=2, extra_words=2),
        record(3, post=2, attachments=0, extra_words=1),
    ]
    posts = serving.group_posts(records)
    bank = {
        post.post_id: [
            Image.new("RGB", (16, 16), (index + 3, 0, 0)) for index in range(len(post.images))
        ]
        for post in posts
    }
    yield posts, bank
    for images in bank.values():
        for image in images:
            image.close()


class Context:
    def __init__(self, root=None):
        self.rows = defaultdict(list)
        self.saved = {}
        self.tags = {}
        self.seed = 42
        self.runs_root = root

    def write_rows(self, name, rows):
        self.rows[name].extend(json.loads(json.dumps(row, allow_nan=False)) for row in rows)

    def log_event(self, **kwargs):
        pass

    def save_json(self, name, value):
        self.saved[name] = value


def test_group_posts_keeps_distinct_pair_records_and_uses_phase1_identity():
    rows = [record(1, post=2), record(0, post=2, pair=1), record(2, post=3)]
    posts = serving.group_posts(rows)
    assert len(posts) == 2
    combined = next(post for post in posts if len(post.records) == 2)
    assert [row["id"] for row in combined.records] == ["record-0", "record-1"]
    assert posts == serving.group_posts(list(reversed(rows)))
    assert len(combined.post_id) == 64


@pytest.mark.parametrize("split", ["test", "train"])
def test_group_rejects_nondev_and_duplicate_ids(split):
    with pytest.raises(ValueError, match="dev"):
        serving.group_posts([record(0, split=split)])
    with pytest.raises(ValueError, match="Duplicate"):
        serving.group_posts([record(0), record(0)])


def test_attachment_bucketing_does_not_drop_or_duplicate_posts(population):
    posts, _ = population
    batches = serving.make_batches(posts, 2, True)
    assert all(len({len(post.images) for post in batch}) == 1 for batch in batches)
    assert sorted(post.post_id for batch in batches for post in batch) == sorted(
        post.post_id for post in posts
    )
    single_image = serving.group_posts([record(index) for index in range(5)])
    assert serving.make_batches(single_image, 2, False) == serving.make_batches(
        single_image, 2, True
    )


def test_select_parent_dev_posts_then_expand_all_official_dev_pairs_only():
    records = [
        record(0, post=0),
        record(1, post=0, pair=1),
        record(2, post=1),
        record(3, post=0, split="test"),
        record(4, post=0, split="train"),
        record(5, post=2),
        record(6, post=0, pair=1, split="val"),
    ]
    posts, metadata = serving.select_dev_posts(records, ["record-2", "record-0"], sample_posts=1)
    assert len(posts) == 1
    assert [row["id"] for row in posts[0].records] == ["record-0", "record-1", "record-6"]
    assert metadata["full_official_dev_posts"] is True
    assert metadata["parent_dev_record_count"] == 2
    assert metadata["parent_dev_post_count"] == 2
    assert metadata["selected_parent_dev_record_ids"] == ["record-0"]
    assert metadata["expanded_dev_record_count"] == 3
    assert metadata["additional_official_dev_record_count"] == 2
    assert metadata["additional_official_dev_record_ids"] == ["record-1", "record-6"]
    reverse_posts, reverse_metadata = serving.select_dev_posts(
        list(reversed(records)), ["record-0", "record-2"], sample_posts=1
    )
    assert posts == reverse_posts
    assert metadata == reverse_metadata


@pytest.mark.parametrize("parent_ids", [[], ["missing"], ["record-0", "record-0"]])
def test_population_expansion_rejects_invalid_parent_dev_ids(parent_ids):
    with pytest.raises(ValueError, match="Parent dev"):
        serving.select_dev_posts([record(0)], parent_ids)


def test_population_expansion_never_allows_test_as_parent_anchor():
    with pytest.raises(ValueError, match="dev"):
        serving.select_dev_posts([record(0, split="test")], ["record-0"])


def test_modernbert_forward_is_batched_and_equal_windows_share_text(encoders):
    records = [record(0, post=0), record(1, post=0, pair=1), record(2, extra_words=3)]
    units, counts = serving.encode_text_batch(encoders, records, "mre")
    assert len(encoders.text_model.calls) == 1
    inputs, mask = encoders.text_model.calls[0]
    assert inputs.shape == (2, 9)
    assert mask.tolist() == [[1] * 6 + [0] * 3, [1] * 9]
    assert counts["text_sequences"] == 2
    assert counts["text_token_slots"] == 18
    assert counts["text_valid_tokens"] == 15
    for actual, source in zip(units, records):
        expected = encoders.encode_text(source, "mre")
        for name in ("head", "tail", "head_width", "tail_width"):
            torch.testing.assert_close(actual[name], expected[name])


def test_entity_preserving_windows_match_phase1(encoders):
    encoders.max_length = 7
    source = record(0, extra_words=8)
    actual, _ = serving.encode_text_batch(encoders, [source], "mre")
    expected = encoders.encode_text(source)
    assert actual[0]["truncated"]
    torch.testing.assert_close(actual[0]["head"], expected["head"])
    torch.testing.assert_close(actual[0]["tail"], expected["tail"])
    source["tail"] = [10, 11]
    with pytest.raises(TextAlignmentError):
        serving.encode_text_batch(encoders, [source], "mre")


def test_naflex_global_and_detail_are_true_single_batch_forwards(encoders):
    images = [Image.new("RGB", (16, 16), (index, 0, 0)) for index in range(4)]
    try:
        globals_, global_counts = serving.encode_vision_batch(encoders, images)
        (patches, mask), detail_counts = serving.encode_vision_batch(encoders, images, detail=True)
    finally:
        for image in images:
            image.close()
    assert encoders.vision_model.calls == [(4, 64), (4, 256)]
    assert globals_.shape == (4, 6)
    assert patches.shape == (4, 256, 6)
    assert not patches[:, -2:].any()
    assert not mask[:, -2:].any()
    assert global_counts == {"forward_calls": 1, "patch_slots": 256, "valid_patches": 248}
    assert detail_counts == {"forward_calls": 1, "patch_slots": 1024, "valid_patches": 1016}


def test_global_masked_pooling_ignores_padding(encoders):
    encoders.vision_model.pooler = False
    with Image.new("RGB", (16, 16)) as image:
        actual, _ = serving.encode_vision_batch(encoders, [image, image])
        expected = encoders.encode_global([image, image])
    torch.testing.assert_close(actual, expected)
    assert actual.abs().max() <= 1


def test_empty_vision_makes_no_forward_and_non_naflex_is_rejected(encoders):
    actual, counts = serving.encode_vision_batch(encoders, [])
    assert actual.shape == (0, 6)
    assert counts["forward_calls"] == 0
    assert encoders.vision_model.calls == []
    encoders.is_naflex = False
    with pytest.raises(ValueError, match="NaFlex"):
        serving.encode_vision_batch(encoders, [])


def test_native_tiny_modernbert_and_naflex_batch_forward_on_cpu(encoders, population):
    from transformers import (
        ModernBertConfig,
        ModernBertModel,
        Siglip2ImageProcessor,
        Siglip2VisionConfig,
        Siglip2VisionModel,
    )

    posts, bank = population
    encoders.text_model = ModernBertModel(
        ModernBertConfig(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
            cls_token_id=101,
            sep_token_id=102,
            reference_compile=False,
        )
    ).eval()
    encoders.vision_model = Siglip2VisionModel(
        Siglip2VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=16,
        )
    ).eval()
    encoders.image_processor = Siglip2ImageProcessor()
    encoders.text_hidden = encoders.vision_hidden = 16
    core = SaverCore(
        3,
        text_dim=16,
        vision_dim=16,
        hidden_dim=8,
        query_dim=4,
        width_dim=2,
        num_heads=2,
        dropout=0,
    ).eval()
    result = serving.forward_batch(core, encoders, posts, bank, "always_top1", math.inf)
    assert result["logits"].shape == (4, 3)
    assert torch.isfinite(result["logits"]).all()
    assert result["counts"]["text_sequences"] == 3
    assert result["counts"]["text_forward_calls"] == 1
    assert result["counts"]["global_forward_calls"] == 1
    assert result["counts"]["detail_forward_calls"] == 1
    assert result["counts"]["global_image_encodes"] == 3
    assert result["counts"]["detail_image_encodes"] == 2


def test_local_pretrained_batching_preserves_order_wordstates_and_detail_masks(tmp_path):
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import (
        ModernBertConfig,
        ModernBertModel,
        PreTrainedTokenizerFast,
        Siglip2ImageProcessor,
        Siglip2Config,
        Siglip2Model,
    )

    records = [
        record(2, extra_words=3),
        record(0, post=0),
        record(1, post=0, pair=1),
        record(3, extra_words=1),
    ]
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for word in sorted({word for source in records for word in source["tokens"]}):
        vocabulary[word] = len(vocabulary)
    backend = BackendTokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    text_path, vision_path = tmp_path / "text", tmp_path / "vision"
    tokenizer.save_pretrained(text_path)
    ModernBertModel(
        ModernBertConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            pad_token_id=0,
            cls_token_id=2,
            sep_token_id=3,
            reference_compile=False,
        )
    ).save_pretrained(text_path)
    Siglip2Model(
        Siglip2Config(
            text_config={"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                         "num_hidden_layers": 1, "num_attention_heads": 2},
            vision_config={"hidden_size": 16, "intermediate_size": 32,
                           "num_hidden_layers": 1, "num_attention_heads": 2, "patch_size": 16},
        )
    ).save_pretrained(vision_path)
    Siglip2ImageProcessor().save_pretrained(vision_path)
    encoders = FrozenEncoders(text_path, vision_path, "cpu")
    text_calls, vision_calls = [], []

    def track_text(module, args, kwargs):
        text_calls.append(tuple(kwargs["input_ids"].shape))

    def track_vision(module, args, kwargs):
        vision_calls.append(tuple(kwargs["pixel_values"].shape[:2]))

    text_hook = encoders.text_model.register_forward_pre_hook(track_text, with_kwargs=True)
    actual, counts = serving.encode_text_batch(encoders, records, "mre")
    text_hook.remove()
    assert len(text_calls) == 1
    assert text_calls[0][0] == counts["text_sequences"] == 3
    assert actual[0]["word_states"].shape[0] > actual[1]["word_states"].shape[0]
    for source, result in zip(records, actual):
        reference = encoders.encode_text(source, "mre")
        for name in ("word_states", "head", "tail", "head_width", "tail_width"):
            torch.testing.assert_close(result[name], reference[name], atol=1e-5, rtol=1e-5)
    reversed_units, _ = serving.encode_text_batch(encoders, list(reversed(records)), "mre")
    for original, reverse in zip(actual, reversed(reversed_units)):
        torch.testing.assert_close(
            original["word_states"], reverse["word_states"], atol=1e-5, rtol=1e-5
        )
    images = [
        Image.new("RGB", (32, 16), (10, 20, 30)),
        Image.new("RGB", (16, 48), (80, 40, 20)),
        Image.new("RGB", (32, 32), (30, 60, 90)),
    ]
    vision_hook = encoders.vision_model.register_forward_pre_hook(track_vision, with_kwargs=True)
    try:
        global_features, _ = serving.encode_vision_batch(encoders, images)
        (patches, masks), _ = serving.encode_vision_batch(encoders, images, detail=True)
        vision_hook.remove()
        assert vision_calls == [(3, 64), (3, 256)]
        for index, image in enumerate(images):
            reference_global = encoders.encode_global([image])[0]
            reference_patches, reference_mask = encoders.encode_detail(image, return_mask=True)
            torch.testing.assert_close(
                global_features[index], reference_global, atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(patches[index], reference_patches, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(masks[index], reference_mask)
        reverse_global, _ = serving.encode_vision_batch(encoders, list(reversed(images)))
        torch.testing.assert_close(global_features, reverse_global.flip(0), atol=1e-5, rtol=1e-5)
    finally:
        vision_hook.remove()
        for image in images:
            image.close()


@pytest.mark.parametrize(
    "method,gate,threshold",
    [
        ("saver_k1", 5.0, 0.5),
        ("saver_k1", -5.0, 0.5),
        ("saver_k1", 5.0, math.inf),
        ("always_top1", -5.0, math.inf),
    ],
)
def test_batched_logits_match_real_saver_and_text_fallback(
    encoders, model, population, method, gate, threshold
):
    posts, image_bank = population
    model.gate_mlp = FixedGate(gate)
    output = serving.forward_batch(model, encoders, posts, image_bank, method, threshold)
    assert len(encoders.text_model.calls) == 1
    assert encoders.text_model.calls[0][0].shape[0] == 3
    assert encoders.vision_model.calls[0] == (3, 64)
    for post, (start, end) in zip(posts, output["unit_ranges"]):
        parts = [encoders.encode_text(source) for source in post.records]
        units = {
            name: torch.cat([part[name] for part in parts])
            for name in ("head", "head_width", "tail", "tail_width")
        }
        global_features = encoders.encode_global(image_bank[post.post_id])
        details = {
            index: encoders.encode_detail(image, return_mask=True)
            for index, image in enumerate(image_bank[post.post_id])
        }
        details = {index: patches[mask] for index, (patches, mask) in details.items()}
        reference = model(
            units,
            global_features,
            details=details,
            method="saver",
            training=False,
            threshold=0 if method == "always_top1" else threshold,
        )
        torch.testing.assert_close(
            output["logits"][start:end], reference["logits"], atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(output["active"][start:end], reference["active"])
    if method == "saver_k1" and (gate < 0 or math.isinf(threshold)):
        torch.testing.assert_close(output["logits"], output["text_logits"], atol=0, rtol=0)
        assert output["counts"]["detail_forward_calls"] == 0
        assert output["counts"]["active_task_units"] == 0
    else:
        assert output["counts"]["detail_forward_calls"] == 1
        assert output["counts"]["detail_image_encodes"] == 2


def test_mixed_gate_batches_only_union_of_acquired_details(encoders, model):
    records = [record(0, post=0), record(1, post=0, pair=1), record(2, post=1), record(3, post=2)]
    posts = serving.group_posts(records)
    images = [Image.new("RGB", (16, 16), (index, 0, 0)) for index in range(3)]
    image_bank = {post.post_id: [image] for post, image in zip(posts, images)}
    model.gate_mlp = FixedGate([6, 6, -6, 6])
    try:
        result = serving.forward_batch(model, encoders, posts, image_bank, "saver_k1", 0.5)
    finally:
        for image in images:
            image.close()
    assert encoders.vision_model.calls == [(3, 64), (2, 256)]
    assert result["active"].tolist() == [True, True, False, True]
    assert result["counts"]["detail_image_encodes"] == 2
    assert result["counts"]["task_units"] == 4
    assert result["counts"]["active_task_units"] == 3
    assert result["counts"]["active_posts"] == 2
    assert result["selected"].flatten().tolist() == [0, 0, -1, 2]


def test_response_latency_uses_posts_not_units_or_amortized_time():
    base = {
        "status": "OK",
        "text_token_slots": 10,
        "text_valid_tokens": 8,
        "global_patch_slots": 64,
        "global_valid_patches": 62,
        "detail_patch_slots": 0,
        "detail_valid_patches": 0,
        "attachment_slots": 4,
        "valid_attachments": 3,
        "actual_attachment_padding": 0,
    }
    rows = [
        {**base, "posts": 2, "task_units": 8, "latency_ms": 20},
        {**base, "posts": 1, "task_units": 1, "latency_ms": 10},
    ]
    summary = serving.summarize_timings(rows)
    assert summary["posts_per_second"] == 100
    assert summary["units_per_second"] == 300
    assert summary["p50_ms"] == 20
    assert summary["p90_ms"] == 20
    assert summary["amortized_ms_per_post"] == 10
    assert summary["padding_waste"] == pytest.approx(4 / 74)
    assert summary["attachment_layout_padding_waste"] == 0.25
    assert summary["actual_attachment_padding"] == 0
    with pytest.raises(ValueError):
        serving.summarize_timings([{**rows[0], "status": "OOM"}])


def test_three_passes_raw_records_warmup_excluded_and_batch_denominators(encoders, model):
    posts = serving.group_posts(
        [record(0, post=0), record(1, post=0, pair=1), record(2, post=1), record(3, post=2)]
    )
    images = [Image.new("RGB", (16, 16), (index, 0, 0)) for index in range(3)]
    bank = {post.post_id: [image] for post, image in zip(posts, images)}
    times = iter(
        [value for setting in range(4) for repeat in range(3) for value in (0, 0.02, 1, 1.01)]
    )
    context = Context()
    try:
        summaries = serving.profile_batches(
            context,
            {"always_top1": model, "saver_k1": model},
            encoders,
            posts,
            bank,
            0.5,
            batch_sizes=(2,),
            clock=lambda: next(times),
        )
    finally:
        for image in images:
            image.close()
    assert len(summaries) == 4
    assert len(context.rows["batch_timings"]) == 24
    assert len(context.rows["batch_predictions"]) == 36
    assert len(encoders.text_model.calls) == 32
    assert any(inputs.shape[0] == 2 for inputs, _ in encoders.text_model.calls)
    for summary in summaries:
        assert summary["timed_posts"] == 9
        assert summary["timed_task_units"] == 12
        assert summary["warmup_batches"] == 2
        assert summary["passes"] == 3
        assert summary["posts_per_second"] == pytest.approx(100)
        assert summary["p50_ms"] == pytest.approx(20)
        assert summary["gpu_utilization"] is None
    first = context.rows["batch_predictions"][0]
    assert len(first["units"]) == 2
    assert [unit["record_id"] for unit in first["units"]] == ["record-0", "record-1"]


def test_only_typed_oom_becomes_explicit_row_without_smaller_batch_retry(
    monkeypatch, encoders, model, population
):
    posts, bank = population
    original = serving.forward_batch
    calls = []

    def one_oom(*args):
        calls.append(len(args[2]))
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("synthetic allocation boundary")
        return original(*args)

    monkeypatch.setattr(serving, "forward_batch", one_oom)
    context = Context()
    summaries = serving.profile_batches(
        context,
        {"always_top1": model, "saver_k1": model},
        encoders,
        posts,
        bank,
        0.5,
        batch_sizes=(8,),
        passes=1,
        warmups=2,
    )
    assert summaries[0]["status"] == "OOM"
    assert summaries[0]["stage"] == "warmup"
    assert summaries[0]["posts_per_second"] is None
    assert context.rows["batch_timings"][0]["status"] == "OOM"
    assert all(row["batch_size"] == 8 for row in summaries)
    assert all(row["status"] == "OK" for row in summaries[1:])


def test_measured_oom_retains_partial_timings_but_never_full_throughput(
    monkeypatch, encoders, model, population
):
    posts, bank = population
    original = serving.forward_batch
    calls = []

    def partial(*args):
        calls.append(len(args[2]))
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("measured allocation boundary")
        return original(*args)

    monkeypatch.setattr(serving, "forward_batch", partial)
    context = Context()
    summaries = serving.profile_batches(
        context,
        {"always_top1": model, "saver_k1": model},
        encoders,
        posts,
        bank,
        0.5,
        batch_sizes=(2,),
        passes=1,
        warmups=0,
    )
    assert summaries[0]["status"] == "OOM"
    assert summaries[0]["stage"] == "measured"
    assert summaries[0]["completed_batches"] == 1
    assert summaries[0]["posts_per_second"] is None
    assert context.rows["batch_timings"][0]["status"] == "OK"
    assert context.rows["batch_timings"][1]["status"] == "OOM"


@pytest.mark.parametrize(
    "error", [RuntimeError("CUDA out of memory: not a typed OOM"), ValueError("bad data")]
)
def test_non_oom_exceptions_are_not_masked(monkeypatch, encoders, model, population, error):
    def fail(*args):
        raise error

    monkeypatch.setattr(serving, "forward_batch", fail)
    posts, bank = population
    with pytest.raises(type(error), match=str(error)):
        serving.profile_batches(
            Context(),
            {"always_top1": model},
            encoders,
            posts,
            bank,
            0.5,
            batch_sizes=(1,),
            passes=1,
        )


def test_main_gpu_guard_precedes_encoder_and_model_loading(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2))
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "data_identity.json").write_text(json.dumps({"split_ids": {"dev": ["record-0"]}}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record(0, attachments=0)) + "\n")
    source = {"manifest": str(manifest)}
    provenance = {
        "method": "saver",
        "task": "mre",
        "parent_run_id": "parent",
        "manifest_sha256": "hash",
    }
    monkeypatch.setattr(
        serving,
        "_parent_artifacts",
        lambda *args: (source, {"None": 0}, [], provenance, parent / "best.pt"),
    )
    monkeypatch.setattr(serving, "_counterpart_run", lambda *args: None)
    calls = []

    def guard():
        calls.append("guard")
        raise RuntimeError("GPU busy; leave phase1 untouched")

    def forbidden(*args, **kwargs):
        pytest.fail("Model/encoder allocation before GPU guard")

    monkeypatch.setattr(serving, "_gpu_guard", guard)
    monkeypatch.setattr(serving, "FrozenEncoders", forbidden)
    monkeypatch.setattr(serving, "SaverCore", forbidden)
    with pytest.raises(RuntimeError, match="GPU busy"):
        serving.main(Context(tmp_path), {"parent_run": str(parent)})
    assert calls == ["guard"]


@pytest.mark.parametrize(
    "config", [{"profiling_passes": 1}, {"warmup_batches": 0}, {"batch_sizes": [1]}]
)
def test_main_rejects_protocol_overrides_before_parent_or_cuda(config):
    with pytest.raises(ValueError, match="preregistration"):
        serving.main(Context(), config)


def test_gpu_telemetry_queries_only_designated_device_and_enforces_limits(monkeypatch):
    monkeypatch.setenv("P24_PROFILE_GPU_UUID", "test-device")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="91, 9000, 70\n")

    monkeypatch.setattr(serving.subprocess, "run", run)
    with serving.GPUUtilization(interval=100) as monitor:
        monitor.check()
    assert len(monitor.samples) == 2
    assert all(command[2] == "test-device" for command in commands)
    monitor.samples[-1]["temperature_c"] = 81
    with pytest.raises(RuntimeError, match="temperature"):
        monitor.check()


def test_gpu_telemetry_errors_are_not_silently_dropped(monkeypatch):
    monkeypatch.setenv("P24_PROFILE_GPU_UUID", "test-device")
    monkeypatch.setattr(
        serving.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="bad")
    )
    with pytest.raises(RuntimeError, match="telemetry"), serving.GPUUtilization():
        pytest.fail("Malformed telemetry must not start profiling")
