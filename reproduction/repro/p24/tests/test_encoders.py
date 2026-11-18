from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from repro.p24.encoders import FrozenEncoders, TextAlignmentError


class FakeEncoding(dict):
    def __init__(self, words):
        ids = [101]
        self.alignment = [None]
        for index, word in enumerate(words):
            count = 2 if word.startswith("multi") else 1
            ids.extend([10 * (index + 1) + part for part in range(count)])
            self.alignment.extend([index] * count)
        ids.append(102)
        self.alignment.append(None)
        super().__init__(
            input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)
        )

    def word_ids(self, batch_index=0):
        assert batch_index == 0
        return self.alignment


class FakeText(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(model_type="modernbert", hidden_size=4)
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        assert not torch.is_grad_enabled()
        self.calls += 1
        states = input_ids.float().unsqueeze(-1).expand(-1, -1, 4) + self.anchor
        return SimpleNamespace(last_hidden_state=states)


class FakeVision(torch.nn.Module):
    def __init__(self, pooler=True):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(hidden_size=6, patch_size=16)
        self.pooler = pooler
        self.calls = []

    def forward(self, pixel_values, pixel_attention_mask=None, spatial_shapes=None, **kwargs):
        assert not torch.is_grad_enabled()
        self.calls.append((pixel_values.shape, pixel_attention_mask, kwargs))
        count = (
            pixel_values.shape[1]
            if pixel_attention_mask is not None
            else (pixel_values.shape[-1] // 16) ** 2
        )
        patches = torch.full((pixel_values.shape[0], count, 6), 2.0) + self.anchor
        if pixel_attention_mask is not None:
            patches = patches.masked_fill(~pixel_attention_mask.bool().unsqueeze(-1), 1000)
        return SimpleNamespace(
            last_hidden_state=patches,
            pooler_output=torch.full((pixel_values.shape[0], 6), 7.0) if self.pooler else None,
        )


class FakeProcessor:
    def __init__(self, naflex=True):
        self.naflex = naflex
        self.calls = []

    def __call__(self, images, return_tensors, **kwargs):
        assert return_tensors == "pt"
        assert all(image.mode == "RGB" for image in images)
        self.calls.append(kwargs)
        if not self.naflex:
            side = kwargs["size"]["height"]
            return {"pixel_values": torch.zeros(len(images), 3, side, side)}
        count = kwargs["max_num_patches"]
        mask = torch.zeros(len(images), count, dtype=torch.long)
        mask[:, : count - 2] = 1
        return {
            "pixel_values": torch.zeros(len(images), count, 16 * 16 * 3),
            "pixel_attention_mask": mask,
            "spatial_shapes": torch.tensor([[1, count - 2]] * len(images)),
        }


@pytest.fixture
def encoders():
    encoder = FrozenEncoders.__new__(FrozenEncoders)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.max_length = 128
    encoder.text_hidden = 4
    encoder.vision_hidden = 6
    encoder.vision_patch_size = 16
    encoder.is_naflex = True
    encoder.tokenizer = lambda words, **kwargs: FakeEncoding(words)
    encoder.text_model = FakeText().eval().requires_grad_(False)
    encoder.vision_model = FakeVision().eval().requires_grad_(False)
    encoder.image_processor = FakeProcessor()
    return encoder


def test_mre_uses_word_alignment_not_subtoken_indices(encoders):
    result = encoders.encode_text(
        {"token": ["first", "multiword", "third"], "h": {"pos": [1, 2]}, "t": {"pos": [1, 3]}}
    )
    assert result["head"].shape == (1, 12)
    assert result["tail"].shape == (1, 12)
    assert result["head"][0].tolist() == [20.0] * 12
    assert result["tail"][0].tolist() == [20.0] * 4 + [30.0] * 8
    assert result["head_width"].tolist() == [1]
    assert result["tail_width"].tolist() == [2]
    assert not result["head"].requires_grad
    assert encoders.text_model.calls == 1


def test_mner_enumeration_and_compact_cache(encoders):
    record = {"tokens": [f"word{index}" for index in range(128)]}
    encoders.max_length = 130
    result = encoders.encode_text(record, "mner")
    assert result["spans"].shape == (1235, 2)
    assert result["head"].shape == (1235, 12)
    assert result["spans"][0].tolist() == [0, 1]
    assert result["spans"][-1].tolist() == [127, 128]
    assert result["head_width"].max().item() == 10
    compact = encoders.encode_text(record, "mner", return_units=False)
    assert "head" not in compact
    reconstructed = encoders.span_features(compact["word_states"], compact["spans"])
    torch.testing.assert_close(reconstructed, result["head"])


def test_entity_preserving_window_records_original_coordinates(encoders):
    encoders.max_length = 8
    result = encoders.encode_text(
        {"tokens": [f"word{index}" for index in range(20)], "head": [14, 15], "tail": [16, 17]}
    )
    assert result["truncated"] is True
    assert result["head_span"].tolist() == [[14, 15]]
    assert result["tail_span"].tolist() == [[16, 17]]
    assert result["word_offsets"].tolist() == [12, 18]
    assert result["head"][0].tolist() == [30.0] * 12
    assert result["tail"][0].tolist() == [50.0] * 12


def test_refuses_truncation_that_loses_entities_or_mner_words(encoders):
    encoders.max_length = 6
    record = {
        "tokens": [f"word{index}" for index in range(10)],
        "head_span": [0, 1],
        "tail_span": [9, 10],
    }
    with pytest.raises(TextAlignmentError, match="Head and tail"):
        encoders.encode_text(record, "mre")
    with pytest.raises(TextAlignmentError, match="exclude record explicitly"):
        encoders.encode_text({"tokens": record["tokens"]}, "mner")
    assert encoders.text_model.calls == 0


@pytest.mark.parametrize("span", [[-1, 2], [2, 2], [0, 20], [0.5, 2], [False, 2]])
def test_rejects_invalid_entity_spans(encoders, span):
    with pytest.raises(TextAlignmentError, match="Invalid half-open"):
        encoders.encode_text({"tokens": ["one", "two", "three"], "head": span, "tail": [1, 2]})


def test_rejects_tokenizer_omission(encoders):
    def omit_word(words, **kwargs):
        encoded = FakeEncoding(words)
        encoded.alignment = [None if value == 1 else value for value in encoded.alignment]
        return encoded

    encoders.tokenizer = omit_word
    with pytest.raises(TextAlignmentError, match="omitted words"):
        encoders.encode_text({"tokens": ["one", "two", "three"]}, "mner")


def test_naflex_uses_pooler_for_screen_and_masked_detail_patches(encoders):
    image = Image.new("RGB", (31, 20))
    global_features = encoders.encode_global([image, image])
    detail, mask = encoders.encode_detail(image, return_mask=True)
    torch.testing.assert_close(global_features, torch.full((2, 6), 7.0))
    assert detail.shape == (256, 6)
    assert mask.shape == (256,)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 254
    assert detail[-2:].count_nonzero().item() == 0
    assert detail[:254].eq(2).all()
    assert encoders.image_processor.calls == [{"max_num_patches": 64}, {"max_num_patches": 256}]
    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert not global_features.requires_grad
    assert not detail.requires_grad


def test_global_masked_mean_ignores_padded_states(encoders):
    encoders.vision_model.pooler = False
    features = encoders.encode_global([Image.new("RGB", (16, 32))])
    torch.testing.assert_close(features, torch.full((1, 6), 2.0))


def test_siglip_substitute_interpolates_128_and_256(encoders):
    encoders.is_naflex = False
    encoders.image_processor = FakeProcessor(naflex=False)
    image = Image.new("L", (32, 32))
    assert encoders.encode_global([image]).shape == (1, 6)
    detail, mask = encoders.encode_detail(image, return_mask=True)
    assert detail.shape == (256, 6)
    assert mask.all()
    assert encoders.image_processor.calls == [
        {"size": {"height": 128, "width": 128}},
        {"size": {"height": 256, "width": 256}},
    ]
    assert all(
        call[2] == {"interpolate_pos_encoding": True} for call in encoders.vision_model.calls
    )
    assert image.mode == "L"


def test_empty_attachments_and_pil_input_never_open_files(encoders, monkeypatch):
    def unexpected_open(*args, **kwargs):
        raise AssertionError("PIL input must bypass disk I/O")

    monkeypatch.setattr(Image, "open", unexpected_open)
    assert encoders.encode_global([]).shape == (0, 6)
    assert encoders.vision_model.calls == []
    encoders.encode_global([Image.new("RGB", (32, 16))])
    encoders.encode_detail(Image.new("RGB", (32, 16)))


def test_frozen_features_can_feed_a_trainable_head(encoders):
    result = encoders.encode_text({"tokens": ["one", "two"]}, "mner")
    head = torch.nn.Linear(12, 3)
    head(result["head"]).sum().backward()
    assert head.weight.grad is not None
    assert encoders.text_model.anchor.grad is None


@pytest.mark.parametrize("backbone", ["siglip2", "siglip"])
def test_native_hf_local_loading_and_variable_resolution(tmp_path, backbone):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import (
        ModernBertConfig,
        ModernBertModel,
        PreTrainedTokenizerFast,
        Siglip2Config,
        Siglip2ImageProcessor,
        Siglip2Model,
        SiglipConfig,
        SiglipImageProcessor,
        SiglipModel,
    )

    text_path = tmp_path / "text"
    vision_path = tmp_path / "vision"
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "first": 4, "last": 5}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
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
    tokenizer.save_pretrained(text_path)
    text_config = ModernBertConfig(
        vocab_size=6,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
        reference_compile=False,
    )
    ModernBertModel(text_config).save_pretrained(text_path)
    vision_config = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "patch_size": 16,
    }
    vl_text_config = {
        "vocab_size": 6,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
    }
    if backbone == "siglip2":
        model = Siglip2Model(Siglip2Config(text_config=vl_text_config, vision_config=vision_config))
        processor = Siglip2ImageProcessor()
    else:
        vision_config["image_size"] = 224
        model = SiglipModel(SiglipConfig(text_config=vl_text_config, vision_config=vision_config))
        processor = SiglipImageProcessor()
    expected_weights = model.vision_model.embeddings.patch_embedding.weight.detach().clone()
    model.save_pretrained(vision_path)
    processor.save_pretrained(vision_path)
    encoder = FrozenEncoders(text_path, vision_path, "cpu")
    assert encoder.vision_hidden == 16
    assert encoder.text_hidden == 16
    assert encoder.text_model.config.reference_compile is False
    assert not encoder.text_model.training
    assert not encoder.vision_model.training
    assert all(not parameter.requires_grad for parameter in encoder.text_model.parameters())
    assert all(not parameter.requires_grad for parameter in encoder.vision_model.parameters())
    torch.testing.assert_close(
        encoder.vision_model.vision_model.embeddings.patch_embedding.weight, expected_weights
    )
    result = encoder.encode_text(
        {"tokens": ["first", "last"], "head": [0, 1], "tail": [1, 2]}, "mre"
    )
    assert result["head"].shape == (1, 48)
    image = Image.new("RGB", (97, 31), color=(40, 70, 90))
    features = encoder.encode_global([image])
    patches, mask = encoder.encode_detail(image, return_mask=True)
    assert features.shape == (1, 16)
    assert patches.shape == (256, 16)
    assert torch.isfinite(features).all()
    assert torch.isfinite(patches).all()
    assert patches[~mask].count_nonzero().item() == 0
    assert not patches.requires_grad
    assert not features.requires_grad
    assert 0 < mask.sum().item() <= 256
