import os
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extensions_built_in.diffusion_models.wan22.wan22_14b_model as wan22_module

from extensions_built_in.diffusion_models.wan22.wan22_14b_model import Wan2214bModel


PLAIN_LORA_KEY = "diffusion_model.layers.0.attention.to_k.lora_A.weight"
HIGH_STAGE_LORA_KEY = "diffusion_model.transformer_1.layers.0.attention.to_k.lora_A.weight"
LOW_STAGE_LORA_KEY = "diffusion_model.transformer_2.layers.0.attention.to_k.lora_A.weight"


def _make_model(
    train_high_noise=True,
    train_low_noise=True,
    lora_path=None,
    high_noise_lora_path=None,
    low_noise_lora_path=None,
):
    model = object.__new__(Wan2214bModel)
    model.train_high_noise = train_high_noise
    model.train_low_noise = train_low_noise
    model.model_config = SimpleNamespace(
        lora_path=lora_path,
        high_noise_lora_path=high_noise_lora_path,
        low_noise_lora_path=low_noise_lora_path,
    )
    return model


def _tensor_dict(key):
    return {key: torch.zeros(1)}


def test_single_stage_inference_ignores_directory_names():
    model = _make_model()

    assert (
        model._infer_single_stage_name_for_wan22_base_lora(
            "user/high_noise-loras/plain_model.safetensors"
        )
        is None
    )


def test_single_stage_inference_uses_filename_suffix():
    model = _make_model()

    assert (
        model._infer_single_stage_name_for_wan22_base_lora("/tmp/plain_model_high.safetensors")
        == "transformer_1"
    )
    assert (
        model._infer_single_stage_name_for_wan22_base_lora("/tmp/plain_model_low_noise.safetensors")
        == "transformer_2"
    )


def test_single_stage_inference_falls_back_to_train_config():
    high_only = _make_model(train_high_noise=True, train_low_noise=False)
    low_only = _make_model(train_high_noise=False, train_low_noise=True)

    assert (
        high_only._infer_single_stage_name_for_wan22_base_lora("/tmp/plain_model.safetensors")
        == "transformer_1"
    )
    assert (
        low_only._infer_single_stage_name_for_wan22_base_lora("/tmp/plain_model.safetensors")
        == "transformer_2"
    )


def test_explicit_base_merge_loads_both_stages(monkeypatch):
    model = _make_model(
        lora_path="legacy.safetensors",
        high_noise_lora_path="high.safetensors",
        low_noise_lora_path="low.safetensors",
    )

    def fake_resolve(path):
        if path == "legacy.safetensors":
            raise AssertionError("legacy lora_path should be ignored in explicit stage mode")
        return f"/resolved/{path}"

    def fake_load_file(path):
        if path.endswith("high.safetensors"):
            return _tensor_dict(PLAIN_LORA_KEY)
        if path.endswith("low.safetensors"):
            return _tensor_dict(PLAIN_LORA_KEY)
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", fake_resolve)
    monkeypatch.setattr(wan22_module, "load_file", fake_load_file)

    state_dict = model._load_wan22_base_lora_state_dict("legacy.safetensors")

    assert list(state_dict.keys()) == [HIGH_STAGE_LORA_KEY, LOW_STAGE_LORA_KEY]


def test_explicit_base_merge_loads_high_stage_only_without_sibling_inference(monkeypatch):
    model = _make_model(high_noise_lora_path="high.safetensors")
    resolved_paths = []

    def fake_resolve(path):
        resolved_paths.append(path)
        return f"/resolved/{path}"

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", fake_resolve)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(PLAIN_LORA_KEY),
    )

    state_dict = model._load_wan22_base_lora_state_dict("unused.safetensors")

    assert resolved_paths == ["high.safetensors"]
    assert list(state_dict.keys()) == [HIGH_STAGE_LORA_KEY]


def test_explicit_base_merge_loads_low_stage_only_without_sibling_inference(monkeypatch):
    model = _make_model(low_noise_lora_path="low.safetensors")
    resolved_paths = []

    def fake_resolve(path):
        resolved_paths.append(path)
        return f"/resolved/{path}"

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", fake_resolve)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(PLAIN_LORA_KEY),
    )

    state_dict = model._load_wan22_base_lora_state_dict("unused.safetensors")

    assert resolved_paths == ["low.safetensors"]
    assert list(state_dict.keys()) == [LOW_STAGE_LORA_KEY]


def test_explicit_high_stage_accepts_already_qualified_weights(monkeypatch):
    model = _make_model(high_noise_lora_path="high.safetensors")

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", lambda path: path)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(HIGH_STAGE_LORA_KEY),
    )

    state_dict = model._load_wan22_base_lora_state_dict("unused.safetensors")

    assert list(state_dict.keys()) == [HIGH_STAGE_LORA_KEY]


def test_explicit_low_stage_accepts_already_qualified_weights(monkeypatch):
    model = _make_model(low_noise_lora_path="low.safetensors")

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", lambda path: path)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(LOW_STAGE_LORA_KEY),
    )

    state_dict = model._load_wan22_base_lora_state_dict("unused.safetensors")

    assert list(state_dict.keys()) == [LOW_STAGE_LORA_KEY]


def test_explicit_high_stage_rejects_low_stage_weights(monkeypatch):
    model = _make_model(high_noise_lora_path="high.safetensors")

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", lambda path: path)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(LOW_STAGE_LORA_KEY),
    )

    with pytest.raises(ValueError, match="contains keys for transformer_2"):
        model._load_wan22_base_lora_state_dict("unused.safetensors")


def test_explicit_low_stage_rejects_high_stage_weights(monkeypatch):
    model = _make_model(low_noise_lora_path="low.safetensors")

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", lambda path: path)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(HIGH_STAGE_LORA_KEY),
    )

    with pytest.raises(ValueError, match="contains keys for transformer_1"):
        model._load_wan22_base_lora_state_dict("unused.safetensors")


def test_explicit_stage_rejects_combined_stage_qualified_weights(monkeypatch):
    model = _make_model(high_noise_lora_path="high.safetensors")

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", lambda path: path)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: {
            HIGH_STAGE_LORA_KEY: torch.zeros(1),
            LOW_STAGE_LORA_KEY: torch.zeros(1),
        },
    )

    with pytest.raises(ValueError, match="contains keys for transformer_2"):
        model._load_wan22_base_lora_state_dict("unused.safetensors")


def test_legacy_lora_path_still_uses_split_sibling_inference(monkeypatch):
    model = _make_model(lora_path="example_high_noise.safetensors")
    resolved_optional_paths = []

    def fake_resolve_optional(path):
        resolved_optional_paths.append(path)
        return f"/resolved/{path}"

    def fake_load_file(path):
        if path.endswith("_high_noise.safetensors"):
            return _tensor_dict(PLAIN_LORA_KEY)
        if path.endswith("_low_noise.safetensors"):
            return _tensor_dict(PLAIN_LORA_KEY)
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(model, "_resolve_optional_wan22_base_lora_path", fake_resolve_optional)
    monkeypatch.setattr(wan22_module, "load_file", fake_load_file)

    state_dict = model._load_wan22_base_lora_state_dict("example_high_noise.safetensors")

    assert resolved_optional_paths == [
        "example_high_noise.safetensors",
        "example_low_noise.safetensors",
    ]
    assert list(state_dict.keys()) == [HIGH_STAGE_LORA_KEY, LOW_STAGE_LORA_KEY]
    assert model.model_config.lora_path == "/resolved/example_high_noise.safetensors"
