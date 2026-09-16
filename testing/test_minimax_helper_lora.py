import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLKIT_ROOT)

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import (
    MinimaxH3FastModel,
    MinimaxH3Model,
)
from toolkit.models.v2.pool import ComponentPool


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()


class RecordingQuantizer:
    def __init__(self):
        self.random_sample = None
        self.stochastic = None
        self.merged_weight = None

    def requantize_codes_(self, module, fp_weight, stochastic=False):
        self.random_sample = torch.rand_like(fp_weight)
        self.stochastic = stochastic
        self.merged_weight = fp_weight.clone()


class FakeQuantLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        self.bias = None
        self.is_ostris_quantized = True
        self.ostris_quantizer = RecordingQuantizer()

    def dequantize_weight(self):
        return self.weight.detach().clone()


class QuantTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = FakeQuantLinear()


def make_holder(transformer, config=None):
    holder = object.__new__(MinimaxH3Model)
    holder.model = transformer
    holder.model_config = config or SimpleNamespace(
        helper_lora_path=None,
        helper_lora_strength=1.0,
    )
    holder.statuses = []
    holder.print_and_status_update = holder.statuses.append
    return holder


def write_lora(path, include_valid=True, include_invalid=False):
    state = {}
    if include_valid:
        state["transformer.proj.lora_A.weight"] = torch.tensor([[1.0, 2.0]])
        state["transformer.proj.lora_B.weight"] = torch.tensor([[3.0], [4.0]])
    if include_invalid:
        state["transformer.missing.lora_A.weight"] = torch.ones(1, 2)
        state["transformer.missing.lora_B.weight"] = torch.ones(2, 1)
    save_file(state, path)


class MiniMaxHelperLoRATest(unittest.TestCase):
    def test_zero_strength_disables_without_resolving_path(self):
        holder = make_holder(
            TinyTransformer(),
            SimpleNamespace(
                helper_lora_path="does/not/exist.safetensors",
                helper_lora_strength=0.0,
            ),
        )
        with mock.patch.object(
            holder, "_resolve_adapter_path", side_effect=AssertionError("must not resolve")
        ):
            self.assertEqual(holder._resolve_helper_lora(), (None, 0.0, None))

    def test_non_finite_strength_is_rejected(self):
        holder = make_holder(
            TinyTransformer(),
            SimpleNamespace(
                helper_lora_path="helper.safetensors",
                helper_lora_strength=float("nan"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "finite number"):
            holder._resolve_helper_lora()

    def test_local_path_resolution_and_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "helper.safetensors")
            write_lora(path)
            holder = make_holder(
                TinyTransformer(),
                SimpleNamespace(helper_lora_path=path, helper_lora_strength=0.75),
            )
            resolved, strength, identity = holder._resolve_helper_lora()
            self.assertEqual(resolved, os.path.realpath(path))
            self.assertEqual(strength, 0.75)
            self.assertEqual(identity, f"{resolved}@0.75")

    def test_invalid_local_path_is_rejected_before_download(self):
        holder = make_holder(TinyTransformer())
        with self.assertRaisesRegex(ValueError, "not a local file"):
            holder._resolve_adapter_path("not-a-file", "Helper")

    def test_plain_merge_is_strength_scaled_permanent_and_rng_safe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "helper.safetensors")
            write_lora(path)
            transformer = TinyTransformer()
            holder = make_holder(transformer)
            torch.manual_seed(9876)
            rng_before = torch.random.get_rng_state().clone()

            holder._merge_helper_lora(transformer, path, 0.5, "helper@0.5")

            expected = torch.tensor([[1.5, 3.0], [2.0, 4.0]])
            torch.testing.assert_close(transformer.proj.weight, expected)
            self.assertEqual(transformer._aitk_helper_lora, "helper@0.5")
            self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
            self.assertFalse(hasattr(holder, "assistant_lora"))

            # A repeated load with the same identity must not merge twice.
            holder._merge_helper_lora(transformer, path, 0.5, "helper@0.5")
            torch.testing.assert_close(transformer.proj.weight, expected)

    def test_quantized_merge_uses_deterministic_stochastic_rounding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "helper.safetensors")
            write_lora(path)
            samples = []
            for _ in range(2):
                transformer = QuantTransformer()
                holder = make_holder(transformer)
                torch.manual_seed(1234)
                rng_before = torch.random.get_rng_state().clone()
                holder._merge_helper_lora(transformer, path, 1.0, "helper@1")
                quantizer = transformer.proj.ostris_quantizer
                self.assertTrue(quantizer.stochastic)
                torch.testing.assert_close(
                    quantizer.merged_weight,
                    torch.tensor([[3.0, 6.0], [4.0, 8.0]]),
                )
                self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
                samples.append(quantizer.random_sample)
            torch.testing.assert_close(samples[0], samples[1])

    def test_no_matches_fail_and_partial_matches_warn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid = os.path.join(temp_dir, "invalid.safetensors")
            write_lora(invalid, include_valid=False, include_invalid=True)
            transformer = TinyTransformer()
            holder = make_holder(transformer)
            with self.assertRaisesRegex(ValueError, "matched no MiniMax-H3"):
                holder._merge_helper_lora(transformer, invalid, 1.0, "invalid@1")

            partial = os.path.join(temp_dir, "partial.safetensors")
            write_lora(partial, include_valid=True, include_invalid=True)
            transformer = TinyTransformer()
            holder = make_holder(transformer)
            holder._merge_helper_lora(transformer, partial, 1.0, "partial@1")
            self.assertTrue(any("1 unmatched" in status for status in holder.statuses))

    def test_pool_evicts_only_incompatible_helper_identity(self):
        pool = ComponentPool()
        helper_a = TinyTransformer()
        helper_a._aitk_helper_lora = "helper-a@1"
        clean = TinyTransformer()
        pool.put("helper-a", helper_a)
        pool.put("clean", clean)
        ComponentPool.current = pool
        try:
            module_name = (
                "extensions_built_in.diffusion_models.minimax_h3.minimax_h3."
                "MiniMaxH3Transformer"
            )
            with mock.patch(module_name, TinyTransformer):
                freed = MinimaxH3Model._evict_incompatible_helper_transformers(
                    "helper-a@1"
                )
                self.assertGreaterEqual(freed, 0)
                self.assertIn("helper-a", pool.entries)
                self.assertNotIn("clean", pool.entries)
                self.assertEqual(
                    MinimaxH3Model._evict_incompatible_helper_transformers(
                        "helper-a@1"
                    ),
                    0,
                )
                MinimaxH3Model._evict_incompatible_helper_transformers("helper-b@1")
                self.assertNotIn("helper-a", pool.entries)
        finally:
            ComponentPool.current = None

    def test_helper_merges_before_toggleable_assistant(self):
        order = []

        class FakeComponent(torch.nn.Module):
            def aitk_post_load(self, **kwargs):
                order.append("post-load-transformer")
                return self

        class FakeTextEncoder(torch.nn.Module):
            def aitk_post_load(self, **kwargs):
                return self

        holder = object.__new__(MinimaxH3Model)
        holder.torch_dtype = torch.float32
        holder.vae_device_torch = torch.device("cpu")
        holder.model_config = SimpleNamespace(assistant_lora_path="assistant.safetensors")
        holder.print_and_status_update = lambda message: None
        holder._resolve_helper_lora = lambda: ("helper.safetensors", 1.0, "helper@1")
        holder._evict_incompatible_helper_transformers = lambda identity: 0
        transformer = FakeComponent()
        holder._load_transformer = lambda: order.append("load-transformer") or transformer
        holder._merge_helper_lora = (
            lambda *args: order.append("merge-helper")
        )
        holder.load_training_adapter = (
            lambda loaded: order.append("load-assistant")
        )
        holder.component_load_kwargs = lambda role: {}
        holder._load_text_encoder = lambda: (object(), object(), FakeTextEncoder())
        holder._load_vaes = lambda: torch.nn.Module()

        with mock.patch.object(
            MinimaxH3Model, "get_train_scheduler", return_value=object()
        ):
            MinimaxH3Model.load_model(holder)

        self.assertEqual(
            order[:4],
            [
                "load-transformer",
                "merge-helper",
                "load-assistant",
                "post-load-transformer",
            ],
        )

    def test_fasth3_rejects_enabled_helper(self):
        enabled = SimpleNamespace(
            helper_lora_path="helper.safetensors",
            helper_lora_strength=1.0,
            model_kwargs={},
        )

        def fake_parent_init(instance, *args, **kwargs):
            instance.model_config = enabled

        with mock.patch.object(MinimaxH3Model, "__init__", fake_parent_init):
            with self.assertRaisesRegex(ValueError, "not minimax_h3_vsa"):
                MinimaxH3FastModel()


if __name__ == "__main__":
    unittest.main()
