import os
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extensions_built_in.diffusion_models.wan22.wan22_14b_model as wan22_module

from extensions_built_in.diffusion_models.wan22.wan22_14b_model import (
    DualWanTransformer3DModel,
    Wan2214bModel,
    boundary_ratio_t2v,
)


PLAIN_LORA_KEY = "diffusion_model.layers.0.attention.to_k.lora_A.weight"
HIGH_STAGE_LORA_KEY = "diffusion_model.transformer_1.layers.0.attention.to_k.lora_A.weight"
LOW_STAGE_LORA_KEY = "diffusion_model.transformer_2.layers.0.attention.to_k.lora_A.weight"


def _make_model(
    train_high_noise=True,
    train_low_noise=True,
    lora_path=None,
    lora_merge_strength=1.0,
    high_noise_lora_path=None,
    high_noise_lora_merge_strength=1.0,
    low_noise_lora_path=None,
    low_noise_lora_merge_strength=1.0,
):
    model = object.__new__(Wan2214bModel)
    model.train_high_noise = train_high_noise
    model.train_low_noise = train_low_noise
    model.model_config = SimpleNamespace(
        lora_path=lora_path,
        lora_merge_strength=lora_merge_strength,
        high_noise_lora_path=high_noise_lora_path,
        high_noise_lora_merge_strength=high_noise_lora_merge_strength,
        low_noise_lora_path=low_noise_lora_path,
        low_noise_lora_merge_strength=low_noise_lora_merge_strength,
        model_kwargs={},
    )
    return model


def _tensor_dict(key):
    return {key: torch.zeros(1)}


class _FakeWanTransformer(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.condition_embedder = SimpleNamespace(forward=lambda *args, **kwargs: None)
        self.config = SimpleNamespace(in_channels=16)
        self.scale_shift_table = torch.nn.Parameter(torch.zeros(1))
        self.blocks = []
        self._device = torch.device("cpu")

    @property
    def device(self):
        return self._device

    def to(self, device=None, *args, **kwargs):
        if device is not None and not isinstance(device, torch.dtype):
            self._device = torch.device(device)
        return self

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True


def _make_load_transformer_model(train_high_noise=True, train_low_noise=True):
    model = _make_model(train_high_noise=train_high_noise, train_low_noise=train_low_noise)
    model.model_config.model_kwargs["load_trainable_stages_only"] = True
    model.model_config.split_model_over_gpus = False
    model.model_config.assistant_lora_path = None
    model.model_config.inference_lora_path = None
    model.model_config.low_vram = True
    model.model_config.layer_offloading = False
    model.model_config.layer_offloading_transformer_percent = 0
    model.model_config.quantize = False
    model.model_config.accuracy_recovery_adapter = None
    model.torch_dtype = torch.float32
    model.device_torch = torch.device("cpu")
    model.print_and_status_update = lambda *args, **kwargs: None
    model._get_wan22_base_lora_merge_specs = lambda: []
    return model


def test_t2v_multistage_boundaries_use_t2v_boundary():
    model = object.__new__(Wan2214bModel)

    assert model._get_wan22_multistage_boundaries() == [boundary_ratio_t2v, 0.0]


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


def test_load_wan_transformer_low_only_skips_high_transformer(monkeypatch):
    model = _make_load_transformer_model(train_high_noise=False, train_low_noise=True)
    calls = []

    def fake_from_pretrained(path, subfolder=None, torch_dtype=None):
        calls.append((path, subfolder, torch_dtype))
        return _FakeWanTransformer(subfolder or os.path.basename(path))

    monkeypatch.setattr(wan22_module.WanTransformer3DModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(wan22_module, "flush", lambda: None)

    transformer = model.load_wan_transformer("/models/wan/transformer")

    assert calls == [("/models/wan/transformer_2", None, torch.float32)]
    assert transformer.transformer_1 is None
    assert transformer.transformer_2 is not None
    assert transformer.transformer is transformer.transformer_2


def test_load_wan_transformer_high_only_skips_low_transformer(monkeypatch):
    model = _make_load_transformer_model(train_high_noise=True, train_low_noise=False)
    calls = []

    def fake_from_pretrained(path, subfolder=None, torch_dtype=None):
        calls.append((path, subfolder, torch_dtype))
        return _FakeWanTransformer(subfolder or os.path.basename(path))

    monkeypatch.setattr(wan22_module.WanTransformer3DModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(wan22_module, "flush", lambda: None)

    transformer = model.load_wan_transformer("/models/wan/transformer")

    assert calls == [("/models/wan/transformer", None, torch.float32)]
    assert transformer.transformer_1 is not None
    assert transformer.transformer_2 is None
    assert transformer.transformer is transformer.transformer_1


def test_dual_transformer_raises_clear_error_for_unloaded_stage():
    transformer = DualWanTransformer3DModel(
        transformer_1=None,
        transformer_2=_FakeWanTransformer("low"),
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
        boundary_ratio=boundary_ratio_t2v,
        low_vram=True,
    )

    with pytest.raises(ValueError, match="transformer_1 is not loaded"):
        transformer(
            hidden_states=torch.zeros(1, 1, 1),
            timestep=torch.tensor([1000]),
            encoder_hidden_states=torch.zeros(1, 1, 1),
        )


def test_base_merge_filters_unloaded_stage_tensors():
    model = _make_model(train_high_noise=False, train_low_noise=True)
    transformer = DualWanTransformer3DModel(
        transformer_1=None,
        transformer_2=_FakeWanTransformer("low"),
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
        boundary_ratio=boundary_ratio_t2v,
        low_vram=True,
    )

    filtered = model._filter_wan22_lora_state_dict_for_loaded_stages(
        {
            HIGH_STAGE_LORA_KEY: torch.zeros(1),
            LOW_STAGE_LORA_KEY: torch.ones(1),
        },
        transformer,
    )

    assert list(filtered.keys()) == [LOW_STAGE_LORA_KEY]


def test_explicit_base_merge_loads_both_stages(monkeypatch):
    model = _make_model(
        lora_path="legacy.safetensors",
        high_noise_lora_path="high.safetensors",
        low_noise_lora_path="low.safetensors",
        high_noise_lora_merge_strength=0.75,
        low_noise_lora_merge_strength=1.25,
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

    merge_specs = model._get_wan22_base_lora_merge_specs()

    assert [list(spec["state_dict"].keys()) for spec in merge_specs] == [
        [HIGH_STAGE_LORA_KEY],
        [LOW_STAGE_LORA_KEY],
    ]
    assert [spec["strength"] for spec in merge_specs] == [0.75, 1.25]


def test_explicit_base_merge_loads_high_stage_only_without_sibling_inference(monkeypatch):
    model = _make_model(high_noise_lora_path="high.safetensors", high_noise_lora_merge_strength=0.6)
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

    merge_specs = model._get_wan22_base_lora_merge_specs()

    assert resolved_paths == ["high.safetensors"]
    assert len(merge_specs) == 1
    assert list(merge_specs[0]["state_dict"].keys()) == [HIGH_STAGE_LORA_KEY]
    assert merge_specs[0]["strength"] == 0.6


def test_explicit_base_merge_loads_low_stage_only_without_sibling_inference(monkeypatch):
    model = _make_model(low_noise_lora_path="low.safetensors", low_noise_lora_merge_strength=1.4)
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

    merge_specs = model._get_wan22_base_lora_merge_specs()

    assert resolved_paths == ["low.safetensors"]
    assert len(merge_specs) == 1
    assert list(merge_specs[0]["state_dict"].keys()) == [LOW_STAGE_LORA_KEY]
    assert merge_specs[0]["strength"] == 1.4


def test_explicit_high_stage_accepts_already_qualified_weights(monkeypatch):
    model = _make_model(high_noise_lora_path="high.safetensors")

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", lambda path: path)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(HIGH_STAGE_LORA_KEY),
    )

    merge_specs = model._get_wan22_base_lora_merge_specs()

    assert list(merge_specs[0]["state_dict"].keys()) == [HIGH_STAGE_LORA_KEY]


def test_explicit_low_stage_accepts_already_qualified_weights(monkeypatch):
    model = _make_model(low_noise_lora_path="low.safetensors")

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", lambda path: path)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(LOW_STAGE_LORA_KEY),
    )

    merge_specs = model._get_wan22_base_lora_merge_specs()

    assert list(merge_specs[0]["state_dict"].keys()) == [LOW_STAGE_LORA_KEY]


def test_explicit_high_stage_rejects_low_stage_weights(monkeypatch):
    model = _make_model(high_noise_lora_path="high.safetensors")

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", lambda path: path)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(LOW_STAGE_LORA_KEY),
    )

    with pytest.raises(ValueError, match="contains keys for transformer_2"):
        model._get_wan22_base_lora_merge_specs()


def test_explicit_low_stage_rejects_high_stage_weights(monkeypatch):
    model = _make_model(low_noise_lora_path="low.safetensors")

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", lambda path: path)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(HIGH_STAGE_LORA_KEY),
    )

    with pytest.raises(ValueError, match="contains keys for transformer_1"):
        model._get_wan22_base_lora_merge_specs()


def test_explicit_base_merge_entrypoint_runs_without_legacy_lora_path(monkeypatch):
    model = _make_model(high_noise_lora_path="high.safetensors", high_noise_lora_merge_strength=0.55)
    model.print_and_status_update = lambda *args, **kwargs: None

    calls = {}

    def fake_get_merge_specs():
        calls["merge_specs_called"] = True
        return [
            {
                "state_dict": _tensor_dict(HIGH_STAGE_LORA_KEY),
                "strength": 0.55,
                "source_path": "/resolved/high.safetensors",
                "stage_name": "transformer_1",
                "label": "high-noise",
            }
        ]

    def fake_infer_network_config(state_dict):
        calls["state_dict_keys"] = list(state_dict.keys())
        return SimpleNamespace(
            linear=4,
            linear_alpha=4,
            conv=None,
            conv_alpha=None,
            type="lora",
            network_kwargs={},
            transformer_only=False,
        )

    class FakeNetwork:
        def __init__(self, **kwargs):
            calls["network_kwargs"] = kwargs

        def apply_to(self, *args, **kwargs):
            calls["apply_to"] = (args, kwargs)

        def load_weights(self, state_dict):
            calls["loaded_weights"] = list(state_dict.keys())

        def merge_in(self, multiplier):
            calls["merge_in"] = multiplier

        def get_all_modules(self):
            return []

    monkeypatch.setattr(model, "_get_wan22_base_lora_merge_specs", fake_get_merge_specs)
    monkeypatch.setattr(model, "_infer_wan22_base_lora_network_config", fake_infer_network_config)
    monkeypatch.setattr(wan22_module, "LoRASpecialNetwork", FakeNetwork)
    monkeypatch.setattr(wan22_module, "flush", lambda: None)

    model._merge_base_lora_into_wan22_transformer(object())

    assert calls["merge_specs_called"] is True
    assert calls["state_dict_keys"] == [HIGH_STAGE_LORA_KEY]
    assert calls["loaded_weights"] == [HIGH_STAGE_LORA_KEY]
    assert calls["merge_in"] == 0.55


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
        model._get_wan22_base_lora_merge_specs()


def test_legacy_lora_path_still_uses_split_sibling_inference(monkeypatch):
    model = _make_model(lora_path="example_high_noise.safetensors", lora_merge_strength=0.8)
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

    merge_specs = model._get_wan22_base_lora_merge_specs()

    assert resolved_optional_paths == [
        "example_high_noise.safetensors",
        "example_low_noise.safetensors",
    ]
    assert [list(spec["state_dict"].keys()) for spec in merge_specs] == [
        [HIGH_STAGE_LORA_KEY],
        [LOW_STAGE_LORA_KEY],
    ]
    assert [spec["strength"] for spec in merge_specs] == [0.8, 0.8]
    assert model.model_config.lora_path == "/resolved/example_high_noise.safetensors"


def test_legacy_combined_lora_uses_legacy_strength(monkeypatch):
    model = _make_model(lora_path="combined.safetensors", lora_merge_strength=1.3)

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", lambda path: f"/resolved/{path}")
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(HIGH_STAGE_LORA_KEY),
    )

    merge_specs = model._get_wan22_base_lora_merge_specs()

    assert len(merge_specs) == 1
    assert merge_specs[0]["strength"] == 1.3
    assert list(merge_specs[0]["state_dict"].keys()) == [HIGH_STAGE_LORA_KEY]


def test_legacy_lora_path_list_uses_default_strength_in_order(monkeypatch):
    lora_paths = ["combined_a.safetensors", "combined_b.safetensors"]
    model = _make_model(lora_path=lora_paths, lora_merge_strength=0.65)

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", lambda path: f"/resolved/{path}")
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(HIGH_STAGE_LORA_KEY),
    )

    merge_specs = model._get_wan22_base_lora_merge_specs()

    assert [spec["source_path"] for spec in merge_specs] == [
        "/resolved/combined_a.safetensors",
        "/resolved/combined_b.safetensors",
    ]
    assert [spec["strength"] for spec in merge_specs] == [0.65, 0.65]
    assert [list(spec["state_dict"].keys()) for spec in merge_specs] == [
        [HIGH_STAGE_LORA_KEY],
        [HIGH_STAGE_LORA_KEY],
    ]
    assert model.model_config.lora_path == lora_paths


def test_legacy_lora_path_list_item_strength_overrides_default(monkeypatch):
    model = _make_model(
        lora_path=[
            {"path": "combined_a.safetensors", "strength": 0.35},
            "combined_b.safetensors",
        ],
        lora_merge_strength=1.2,
    )

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", lambda path: f"/resolved/{path}")
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(HIGH_STAGE_LORA_KEY),
    )

    merge_specs = model._get_wan22_base_lora_merge_specs()

    assert [spec["source_path"] for spec in merge_specs] == [
        "/resolved/combined_a.safetensors",
        "/resolved/combined_b.safetensors",
    ]
    assert [spec["strength"] for spec in merge_specs] == [0.35, 1.2]


def test_explicit_base_merge_lists_stage_entries_and_strengths(monkeypatch):
    model = _make_model(
        lora_path="legacy.safetensors",
        high_noise_lora_path=[
            {"path": "high_a.safetensors", "strength": 0.25},
            "high_b.safetensors",
        ],
        high_noise_lora_merge_strength=0.75,
        low_noise_lora_path=[
            "low_a.safetensors",
            {"path": "low_b.safetensors", "strength": 1.35},
        ],
        low_noise_lora_merge_strength=1.1,
    )

    def fake_resolve(path):
        if path == "legacy.safetensors":
            raise AssertionError("legacy lora_path should be ignored in explicit stage mode")
        return f"/resolved/{path}"

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", fake_resolve)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(PLAIN_LORA_KEY),
    )

    merge_specs = model._get_wan22_base_lora_merge_specs()

    assert [spec["source_path"] for spec in merge_specs] == [
        "/resolved/high_a.safetensors",
        "/resolved/high_b.safetensors",
        "/resolved/low_a.safetensors",
        "/resolved/low_b.safetensors",
    ]
    assert [spec["strength"] for spec in merge_specs] == [0.25, 0.75, 1.1, 1.35]
    assert [list(spec["state_dict"].keys()) for spec in merge_specs] == [
        [HIGH_STAGE_LORA_KEY],
        [HIGH_STAGE_LORA_KEY],
        [LOW_STAGE_LORA_KEY],
        [LOW_STAGE_LORA_KEY],
    ]


def test_lora_path_list_rejects_malformed_entries():
    missing_path = _make_model(lora_path=[{"strength": 0.5}])
    with pytest.raises(ValueError, match="must include a non-empty `path`"):
        missing_path._get_wan22_base_lora_merge_specs()

    bad_strength = _make_model(lora_path=[{"path": "combined.safetensors", "strength": "heavy"}])
    with pytest.raises(ValueError, match="strength must be numeric"):
        bad_strength._get_wan22_base_lora_merge_specs()

    bad_entry = _make_model(high_noise_lora_path=[123])
    with pytest.raises(ValueError, match="must be a path string or an object"):
        bad_entry._get_wan22_base_lora_merge_specs()


def test_explicit_mode_ignores_legacy_path_and_strength(monkeypatch):
    model = _make_model(
        lora_path="legacy.safetensors",
        lora_merge_strength=2.0,
        high_noise_lora_path="high.safetensors",
        high_noise_lora_merge_strength=0.7,
    )

    def fake_resolve(path):
        if path == "legacy.safetensors":
            raise AssertionError("legacy path should not be resolved in explicit mode")
        return f"/resolved/{path}"

    monkeypatch.setattr(model, "_resolve_wan22_base_lora_path", fake_resolve)
    monkeypatch.setattr(
        wan22_module,
        "load_file",
        lambda path: _tensor_dict(PLAIN_LORA_KEY),
    )

    merge_specs = model._get_wan22_base_lora_merge_specs()

    assert len(merge_specs) == 1
    assert merge_specs[0]["strength"] == 0.7


def test_merge_entrypoint_uses_separate_strengths_and_network_configs(monkeypatch):
    model = _make_model()
    model.print_and_status_update = lambda *args, **kwargs: None

    calls = {
        "network_kwargs": [],
        "merge_in": [],
        "loaded_weights": [],
        "infer_inputs": [],
    }
    merge_specs = [
        {
            "state_dict": _tensor_dict(HIGH_STAGE_LORA_KEY),
            "strength": 0.6,
            "source_path": "/resolved/high.safetensors",
            "stage_name": "transformer_1",
            "label": "high-noise",
        },
        {
            "state_dict": _tensor_dict(LOW_STAGE_LORA_KEY),
            "strength": 1.4,
            "source_path": "/resolved/low.safetensors",
            "stage_name": "transformer_2",
            "label": "low-noise",
        },
    ]

    def fake_get_merge_specs():
        return merge_specs

    def fake_infer_network_config(state_dict):
        keys = list(state_dict.keys())
        calls["infer_inputs"].append(keys)
        if keys == [HIGH_STAGE_LORA_KEY]:
            return SimpleNamespace(
                linear=4,
                linear_alpha=4,
                conv=None,
                conv_alpha=None,
                type="lora",
                network_kwargs={"only_if_contains": ["high"]},
                transformer_only=False,
            )
        return SimpleNamespace(
            linear=8,
            linear_alpha=8,
            conv=None,
            conv_alpha=None,
            type="lokr",
            network_kwargs={"only_if_contains": ["low"]},
            transformer_only=False,
        )

    class FakeNetwork:
        def __init__(self, **kwargs):
            calls["network_kwargs"].append(kwargs)

        def apply_to(self, *args, **kwargs):
            pass

        def load_weights(self, state_dict):
            calls["loaded_weights"].append(list(state_dict.keys()))

        def merge_in(self, multiplier):
            calls["merge_in"].append(multiplier)

        def get_all_modules(self):
            return []

    monkeypatch.setattr(model, "_get_wan22_base_lora_merge_specs", fake_get_merge_specs)
    monkeypatch.setattr(model, "_infer_wan22_base_lora_network_config", fake_infer_network_config)
    monkeypatch.setattr(wan22_module, "LoRASpecialNetwork", FakeNetwork)
    monkeypatch.setattr(wan22_module, "flush", lambda: None)

    model._merge_base_lora_into_wan22_transformer(object())

    assert calls["infer_inputs"] == [[HIGH_STAGE_LORA_KEY], [LOW_STAGE_LORA_KEY]]
    assert calls["loaded_weights"] == [[HIGH_STAGE_LORA_KEY], [LOW_STAGE_LORA_KEY]]
    assert calls["merge_in"] == [0.6, 1.4]
    assert calls["network_kwargs"][0]["lora_dim"] == 4
    assert calls["network_kwargs"][0]["network_type"] == "lora"
    assert calls["network_kwargs"][0]["only_if_contains"] == ["high"]
    assert calls["network_kwargs"][1]["lora_dim"] == 8
    assert calls["network_kwargs"][1]["network_type"] == "lokr"
    assert calls["network_kwargs"][1]["only_if_contains"] == ["low"]


def test_legacy_split_merge_entrypoint_uses_legacy_strength_for_each_spec(monkeypatch):
    model = _make_model()
    model.print_and_status_update = lambda *args, **kwargs: None

    calls = {
        "merge_in": [],
        "loaded_weights": [],
    }
    merge_specs = [
        {
            "state_dict": _tensor_dict(HIGH_STAGE_LORA_KEY),
            "strength": 0.9,
            "source_path": "/resolved/high.safetensors",
            "stage_name": "transformer_1",
            "label": "high-noise sibling",
        },
        {
            "state_dict": _tensor_dict(LOW_STAGE_LORA_KEY),
            "strength": 0.9,
            "source_path": "/resolved/low.safetensors",
            "stage_name": "transformer_2",
            "label": "low-noise sibling",
        },
    ]

    def fake_infer_network_config(state_dict):
        return SimpleNamespace(
            linear=4,
            linear_alpha=4,
            conv=None,
            conv_alpha=None,
            type="lora",
            network_kwargs={},
            transformer_only=False,
        )

    class FakeNetwork:
        def __init__(self, **kwargs):
            pass

        def apply_to(self, *args, **kwargs):
            pass

        def load_weights(self, state_dict):
            calls["loaded_weights"].append(list(state_dict.keys()))

        def merge_in(self, multiplier):
            calls["merge_in"].append(multiplier)

        def get_all_modules(self):
            return []

    monkeypatch.setattr(model, "_get_wan22_base_lora_merge_specs", lambda: merge_specs)
    monkeypatch.setattr(model, "_infer_wan22_base_lora_network_config", fake_infer_network_config)
    monkeypatch.setattr(wan22_module, "LoRASpecialNetwork", FakeNetwork)
    monkeypatch.setattr(wan22_module, "flush", lambda: None)

    model._merge_base_lora_into_wan22_transformer(object())

    assert calls["loaded_weights"] == [[HIGH_STAGE_LORA_KEY], [LOW_STAGE_LORA_KEY]]
    assert calls["merge_in"] == [0.9, 0.9]


def test_save_lora_high_only_writes_high_noise_file(monkeypatch):
    model = _make_model(train_high_noise=True, train_low_noise=False)
    model.network = SimpleNamespace(
        network_config=SimpleNamespace(split_multistage_loras=True)
    )
    saved = {}

    def fake_save_file(state_dict, path, metadata=None):
        saved[path] = state_dict

    monkeypatch.setattr(wan22_module, "save_file", fake_save_file)

    state_dict = {
        "diffusion_model.blocks.0.attn.to_q.lora_down.weight": torch.zeros(1),
    }
    model.save_lora(state_dict, "/tmp/test.safetensors")

    assert list(saved.keys()) == ["/tmp/test_high_noise.safetensors"]
    assert saved["/tmp/test_high_noise.safetensors"] == state_dict


def test_save_lora_low_only_writes_low_noise_file(monkeypatch):
    model = _make_model(train_high_noise=False, train_low_noise=True)
    model.network = SimpleNamespace(
        network_config=SimpleNamespace(split_multistage_loras=True)
    )
    saved = {}

    def fake_save_file(state_dict, path, metadata=None):
        saved[path] = state_dict

    monkeypatch.setattr(wan22_module, "save_file", fake_save_file)

    state_dict = {
        "diffusion_model.blocks.0.attn.to_q.lora_down.weight": torch.zeros(1),
    }
    model.save_lora(state_dict, "/tmp/test.safetensors")

    assert list(saved.keys()) == ["/tmp/test_low_noise.safetensors"]
    assert saved["/tmp/test_low_noise.safetensors"] == state_dict
