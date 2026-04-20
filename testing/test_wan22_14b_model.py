import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions_built_in.diffusion_models.wan22.wan22_14b_model import Wan2214bModel


def _make_model(train_high_noise=True, train_low_noise=True):
    model = object.__new__(Wan2214bModel)
    model.train_high_noise = train_high_noise
    model.train_low_noise = train_low_noise
    return model


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
