import importlib
import os
import pathlib
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file


# Import only the MiniMax-H3 extension under test. The diffusion-model package
# initializer eagerly imports every built-in architecture, which would turn
# this focused unit test into a full optional-dependency integration test.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_STUBBED_MODULE_NAMES = (
    "extensions_built_in",
    "extensions_built_in.diffusion_models",
    "toolkit.models.base_model",
)
_saved_modules = {name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES}
_modules_before_import = set(sys.modules)


class _BaseModel:
    def print_and_status_update(self, status):
        self._status_update(status)

    def _status_update(self, status):
        for hook in self._status_update_hooks:
            hook(status)


for _package_name, _package_path in (
    ("extensions_built_in", _PROJECT_ROOT / "extensions_built_in"),
    (
        "extensions_built_in.diffusion_models",
        _PROJECT_ROOT / "extensions_built_in" / "diffusion_models",
    ),
):
    _package = types.ModuleType(_package_name)
    _package.__path__ = [str(_package_path)]
    sys.modules[_package_name] = _package

_base_model_module = types.ModuleType("toolkit.models.base_model")
_base_model_module.BaseModel = _BaseModel
sys.modules["toolkit.models.base_model"] = _base_model_module

try:
    _minimax_module = importlib.import_module(
        "extensions_built_in.diffusion_models.minimax_h3.minimax_h3"
    )
finally:
    for _module_name in list(sys.modules):
        if (
            _module_name.startswith("extensions_built_in.diffusion_models.minimax_h3")
            and _module_name not in _modules_before_import
        ):
            del sys.modules[_module_name]
    for _module_name, _saved_module in _saved_modules.items():
        if _saved_module is None:
            sys.modules.pop(_module_name, None)
        else:
            sys.modules[_module_name] = _saved_module

MinimaxH3Model = _minimax_module.MinimaxH3Model
_resolve_flow_shift = _minimax_module._resolve_flow_shift
packing = _minimax_module.packing


class _Block(torch.nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim, bias=False)


class MiniMaxH3Transformer(torch.nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(dim)])


class MinimaxH3FrozenHelperTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _model(self, helper_path, strength=1.0):
        model = object.__new__(MinimaxH3Model)
        model.model_config = SimpleNamespace(
            lora_path=helper_path,
            lora_strength=strength,
            qtype=None,
        )
        model.target_lora_modules = ["MiniMaxH3Transformer"]
        model.device_torch = torch.device("cpu")
        model.torch_dtype = torch.float32
        model._status_update_hooks = []
        return model

    def _save(self, name, state_dict):
        path = os.path.join(self.temp_dir.name, name)
        save_file(state_dict, path)
        return path

    def test_custom_flow_shifts_keep_video_and_audio_on_same_base_position(self):
        model = object.__new__(MinimaxH3Model)
        model.flow_shift = 7.5
        model.audio_flow_shift = 2.25

        video_sigmas, audio_sigmas = model.build_sigma_schedules(20)
        base = torch.linspace(1.0, 0.0, 21, dtype=torch.float32)

        torch.testing.assert_close(video_sigmas, packing.shift_sigma(base, 7.5))
        torch.testing.assert_close(audio_sigmas, packing.shift_sigma(base, 2.25))
        scheduler = MinimaxH3Model.get_train_scheduler(7.5)
        self.assertEqual(float(scheduler.config.shift), 7.5)
        scheduler.set_train_timesteps(50, torch.device("cpu"), timestep_type="shift")
        train_video_sigmas = scheduler.timesteps / 1000.0
        base_positions = train_video_sigmas / (
            7.5 + train_video_sigmas * (1.0 - 7.5)
        )
        torch.testing.assert_close(
            model.remap_audio_sigma(train_video_sigmas),
            packing.shift_sigma(base_positions, 2.25),
        )

    def test_flow_shifts_require_positive_finite_numbers(self):
        self.assertEqual(_resolve_flow_shift(None, 12.0, "flow_shift"), 12.0)
        for value in (0, -1, float("nan"), float("inf"), "bad"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "must be a finite number greater than 0"
            ):
                _resolve_flow_shift(value, 12.0, "flow_shift")

    def test_fuses_ai_toolkit_lora_without_installing_a_live_wrapper(self):
        transformer = MiniMaxH3Transformer()
        original = transformer.blocks[0].proj.weight.detach().clone()
        down = torch.randn(2, 8)
        up = torch.randn(8, 2)
        path = self._save(
            "helper_lora.safetensors",
            {
                "diffusion_model.blocks.0.proj.lora_A.weight": down,
                "diffusion_model.blocks.0.proj.lora_B.weight": up,
            },
        )

        self._model(path).load_frozen_helper_adapter(transformer)

        torch.testing.assert_close(
            transformer.blocks[0].proj.weight,
            original + up @ down,
        )
        self.assertIs(transformer.blocks[0].proj.forward.__self__, transformer.blocks[0].proj)

    def test_fuses_full_rank_ai_toolkit_lokr(self):
        transformer = MiniMaxH3Transformer()
        original = transformer.blocks[0].proj.weight.detach().clone()
        w1 = torch.randn(2, 2)
        w2 = torch.randn(4, 4)
        path = self._save(
            "helper_full_lokr.safetensors",
            {
                "diffusion_model.blocks.0.proj.lokr_w1": w1,
                "diffusion_model.blocks.0.proj.lokr_w2": w2,
                # Full-rank LoKr's huge alpha becomes inf when saved as fp16.
                "diffusion_model.blocks.0.proj.alpha": torch.tensor(float("inf")),
            },
        )

        self._model(path).load_frozen_helper_adapter(transformer)

        torch.testing.assert_close(
            transformer.blocks[0].proj.weight,
            original + torch.kron(w1, w2),
        )

    def test_scales_lora_helper_merge_strength(self):
        transformer = MiniMaxH3Transformer()
        original = transformer.blocks[0].proj.weight.detach().clone()
        down = torch.randn(2, 8)
        up = torch.randn(8, 2)
        path = self._save(
            "scaled_helper_lora.safetensors",
            {
                "diffusion_model.blocks.0.proj.lora_A.weight": down,
                "diffusion_model.blocks.0.proj.lora_B.weight": up,
            },
        )
        statuses = []
        model = self._model(path, strength=0.35)
        model._status_update_hooks.append(statuses.append)

        model.load_frozen_helper_adapter(transformer)

        torch.testing.assert_close(
            transformer.blocks[0].proj.weight,
            original + 0.35 * (up @ down),
        )
        self.assertIn("at strength 0.35", statuses[-1])

    def test_scales_lokr_helper_merge_strength(self):
        transformer = MiniMaxH3Transformer()
        original = transformer.blocks[0].proj.weight.detach().clone()
        w1 = torch.randn(2, 2)
        w2 = torch.randn(4, 4)
        path = self._save(
            "scaled_helper_lokr.safetensors",
            {
                "diffusion_model.blocks.0.proj.lokr_w1": w1,
                "diffusion_model.blocks.0.proj.lokr_w2": w2,
                "diffusion_model.blocks.0.proj.alpha": torch.tensor(float("inf")),
            },
        )

        self._model(path, strength=-0.5).load_frozen_helper_adapter(transformer)

        torch.testing.assert_close(
            transformer.blocks[0].proj.weight,
            original - 0.5 * torch.kron(w1, w2),
        )

    def test_rejects_non_finite_helper_merge_strength(self):
        transformer = MiniMaxH3Transformer()
        path = self._save(
            "helper_lora.safetensors",
            {
                "diffusion_model.blocks.0.proj.lora_A.weight": torch.randn(2, 8),
                "diffusion_model.blocks.0.proj.lora_B.weight": torch.randn(8, 2),
            },
        )

        for strength in (float("nan"), float("inf"), "not-a-number"):
            with self.subTest(strength=strength), self.assertRaisesRegex(
                ValueError, "lora_strength must be a finite number"
            ):
                self._model(path, strength=strength).load_frozen_helper_adapter(
                    transformer
                )

    def test_zero_strength_does_not_touch_quantized_base(self):
        from toolkit.util.ostris_quant import (
            convert_linear_to_ostris,
            get_ostris_quantizer,
        )

        transformer = MiniMaxH3Transformer(dim=32)
        proj = transformer.blocks[0].proj
        self.assertTrue(
            convert_linear_to_ostris(proj, get_ostris_quantizer("convrot8"))
        )
        original = proj.weight.detach().clone()
        path = self._save(
            "zero_strength_helper.safetensors",
            {
                "diffusion_model.blocks.0.proj.lora_A.weight": torch.randn(2, 32),
                "diffusion_model.blocks.0.proj.lora_B.weight": torch.randn(32, 2),
            },
        )
        statuses = []
        model = self._model(path, strength=0)
        model._status_update_hooks.append(statuses.append)

        model.load_frozen_helper_adapter(transformer)

        torch.testing.assert_close(transformer.blocks[0].proj.weight, original)
        self.assertIn("merge skipped", statuses[-1])

    def test_fuses_factorized_ai_toolkit_lokr(self):
        transformer = MiniMaxH3Transformer()
        original = transformer.blocks[0].proj.weight.detach().clone()
        w1 = torch.randn(2, 2)
        w2_a = torch.randn(4, 1)
        w2_b = torch.randn(1, 4)
        path = self._save(
            "helper_factorized_lokr.safetensors",
            {
                "diffusion_model.blocks.0.proj.lokr_w1": w1,
                "diffusion_model.blocks.0.proj.lokr_w2_a": w2_a,
                "diffusion_model.blocks.0.proj.lokr_w2_b": w2_b,
                "diffusion_model.blocks.0.proj.alpha": torch.tensor(1),
            },
        )

        self._model(path).load_frozen_helper_adapter(transformer)

        torch.testing.assert_close(
            transformer.blocks[0].proj.weight,
            original + torch.kron(w1, w2_a @ w2_b),
        )

    def test_rejects_mixed_lora_and_lokr(self):
        state_dict = {
            "transformer.blocks.0.proj.lora_A.weight": torch.randn(2, 8),
            "transformer.blocks.0.proj.lora_B.weight": torch.randn(8, 2),
            "transformer.blocks.0.proj.lokr_w1": torch.randn(2, 2),
            "transformer.blocks.0.proj.lokr_w2": torch.randn(4, 4),
            "transformer.blocks.0.proj.alpha": torch.tensor(9_999_999_999),
        }

        with self.assertRaisesRegex(ValueError, "cannot mix LoRA and LoKr"):
            MinimaxH3Model._inspect_frozen_helper_state_dict(state_dict)

    def test_rejects_legacy_lycoris_keys(self):
        state_dict = {
            "lycoris_blocks_0_proj.lokr_w1": torch.randn(2, 2),
            "lycoris_blocks_0_proj.lokr_w2": torch.randn(4, 4),
            "lycoris_blocks_0_proj.alpha": torch.tensor(9_999_999_999),
        }

        with self.assertRaisesRegex(ValueError, "keys must start"):
            MinimaxH3Model._inspect_frozen_helper_state_dict(state_dict)

    def test_rejects_lokr_alpha_that_cannot_be_reconstructed_exactly(self):
        state_dict = {
            "transformer.blocks.0.proj.lokr_w1": torch.randn(2, 2),
            "transformer.blocks.0.proj.lokr_w2_a": torch.randn(4, 1),
            "transformer.blocks.0.proj.lokr_w2_b": torch.randn(1, 4),
            "transformer.blocks.0.proj.alpha": torch.tensor(0.5),
        }

        with self.assertRaisesRegex(ValueError, "unsupported alpha scaling"):
            MinimaxH3Model._inspect_frozen_helper_state_dict(state_dict)

    def test_rejects_unmatched_modules_before_merging(self):
        transformer = MiniMaxH3Transformer()
        original = transformer.blocks[0].proj.weight.detach().clone()
        path = self._save(
            "unknown_module.safetensors",
            {
                "diffusion_model.blocks.0.unknown.lora_A.weight": torch.randn(2, 8),
                "diffusion_model.blocks.0.unknown.lora_B.weight": torch.randn(8, 2),
            },
        )

        with self.assertRaisesRegex(ValueError, "module mismatch"):
            self._model(path).load_frozen_helper_adapter(transformer)
        torch.testing.assert_close(transformer.blocks[0].proj.weight, original)

    def test_rejects_incompatible_shapes_before_merging(self):
        transformer = MiniMaxH3Transformer()
        original = transformer.blocks[0].proj.weight.detach().clone()
        path = self._save(
            "wrong_shape.safetensors",
            {
                "diffusion_model.blocks.0.proj.lora_A.weight": torch.randn(2, 7),
                "diffusion_model.blocks.0.proj.lora_B.weight": torch.randn(8, 2),
            },
        )

        with self.assertRaisesRegex(ValueError, "shape"):
            self._model(path).load_frozen_helper_adapter(transformer)
        torch.testing.assert_close(transformer.blocks[0].proj.weight, original)

    def test_requires_local_safetensors_file(self):
        transformer = MiniMaxH3Transformer()
        missing = os.path.join(self.temp_dir.name, "missing.safetensors")
        with self.assertRaisesRegex(FileNotFoundError, "local file"):
            self._model(missing).load_frozen_helper_adapter(transformer)

        wrong_extension = os.path.join(self.temp_dir.name, "helper.pt")
        with open(wrong_extension, "wb") as handle:
            handle.write(b"not an adapter")
        with self.assertRaisesRegex(ValueError, "safetensors"):
            self._model(wrong_extension).load_frozen_helper_adapter(transformer)

    def test_requantizes_prequantized_ostris_linear_after_fusion(self):
        from toolkit.util.ostris_quant import (
            OstrisLinear,
            convert_linear_to_ostris,
            get_ostris_quantizer,
        )

        transformer = MiniMaxH3Transformer(dim=32)
        proj = transformer.blocks[0].proj
        self.assertTrue(
            convert_linear_to_ostris(proj, get_ostris_quantizer("convrot8"))
        )
        original = proj.weight.detach().clone()
        down = torch.randn(2, 32) * 0.01
        up = torch.randn(32, 2) * 0.01
        path = self._save(
            "quantized_helper_lora.safetensors",
            {
                "diffusion_model.blocks.0.proj.lora_A.weight": down,
                "diffusion_model.blocks.0.proj.lora_B.weight": up,
            },
        )

        self._model(path).load_frozen_helper_adapter(transformer)

        self.assertIsInstance(transformer.blocks[0].proj, OstrisLinear)
        torch.testing.assert_close(
            transformer.blocks[0].proj.weight,
            original + up @ down,
            rtol=2e-2,
            atol=2e-3,
        )

    def test_helper_loads_before_live_assistant(self):
        events = []
        model = object.__new__(MinimaxH3Model)
        model.model_config = SimpleNamespace(
            lora_path="helper.safetensors",
            assistant_lora_path="assistant.safetensors",
            quantize=False,
            quantize_te=False,
            layer_offloading=False,
            layer_offloading_transformer_percent=0,
            layer_offloading_text_encoder_percent=0,
            low_vram=False,
        )
        model.flow_shift = 12.0
        model.audio_flow_shift = 3.0
        model.torch_dtype = torch.float32
        model.device_torch = torch.device("cpu")
        model.vae_device_torch = torch.device("cpu")
        model._status_update_hooks = []
        transformer = MiniMaxH3Transformer()
        model._load_transformer = lambda: transformer
        model.load_frozen_helper_adapter = lambda value: events.append("helper")
        model.load_training_adapter = lambda value: events.append("assistant")
        model._load_text_encoder = lambda: (
            object(),
            object(),
            torch.nn.Linear(1, 1),
        )
        model._load_vaes = lambda: torch.nn.Linear(1, 1)

        with mock.patch.object(
            MinimaxH3Model,
            "get_train_scheduler",
            return_value=object(),
        ), mock.patch.object(
            _minimax_module,
            "MiniMaxH3Pipeline",
            return_value=object(),
        ):
            model.load_model()

        self.assertEqual(events, ["helper", "assistant"])


if __name__ == "__main__":
    unittest.main()
