import os
import sys
from types import SimpleNamespace

import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extensions_built_in.diffusion_models.wan22.wan22_14b_i2v_model as wan22_i2v_module

from extensions_built_in.concept_slider.ConceptSliderTrainer import (
    ConceptSliderTrainer,
    ConceptSliderTrainerConfig,
)
from extensions_built_in.diffusion_models.wan22.wan22_14b_i2v_model import (
    Wan2214bI2VModel,
)
from toolkit.prompt_utils import PromptEmbeds


class _FakeTransformer:
    def __init__(self, in_channels=26):
        self.hidden_states = []
        self.patch_embedding = SimpleNamespace(in_channels=in_channels)

    def __call__(self, hidden_states, **kwargs):
        self.hidden_states.append(hidden_states)
        return (hidden_states.clone().requires_grad_(True),)


class _FakeUnet:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _FakeNetwork:
    def __init__(self):
        self.is_active = True
        self.multipliers = []

    def set_multiplier(self, value):
        self.multipliers.append(value)


def _make_prompt_embeds():
    return PromptEmbeds(torch.zeros(1, 4))


def _make_i2v_model(in_channels=26):
    model = object.__new__(Wan2214bI2VModel)
    model.model = _FakeTransformer(in_channels=in_channels)
    model.vae = object()
    return model


def test_force_t2i_single_frame_skips_first_frame_conditioning(monkeypatch):
    model = _make_i2v_model()
    latent_model_input = torch.randn(1, 16, 1, 2, 2)
    batch = SimpleNamespace(
        tensor=torch.randn(1, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=1),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("single-frame conditioning should be skipped")

    monkeypatch.setattr(wan22_i2v_module, "add_first_frame_conditioning", fail_if_called)

    model.get_noise_prediction(
        latent_model_input=latent_model_input,
        timestep=torch.tensor([1]),
        text_embeddings=_make_prompt_embeds(),
        batch=batch,
        force_t2i_single_frame=True,
    )

    hidden_states = model.model.hidden_states[-1]
    assert hidden_states.shape == (1, 26, 1, 2, 2)
    assert torch.equal(hidden_states[:, :16], latent_model_input)
    assert torch.count_nonzero(hidden_states[:, 16:]) == 0


def test_image_batches_keep_first_frame_conditioning_by_default(monkeypatch):
    model = _make_i2v_model()
    latent_model_input = torch.randn(1, 16, 1, 2, 2)
    sentinel = torch.randn(1, 26, 1, 2, 2)
    batch = SimpleNamespace(
        tensor=torch.randn(1, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=1),
    )

    monkeypatch.setattr(
        wan22_i2v_module,
        "add_first_frame_conditioning",
        lambda **kwargs: sentinel,
    )

    model.get_noise_prediction(
        latent_model_input=latent_model_input,
        timestep=torch.tensor([1]),
        text_embeddings=_make_prompt_embeds(),
        batch=batch,
    )

    assert torch.equal(model.model.hidden_states[-1], sentinel)


def test_video_batches_still_use_first_frame_conditioning(monkeypatch):
    model = _make_i2v_model()
    latent_model_input = torch.randn(1, 16, 3, 2, 2)
    sentinel = torch.randn(1, 26, 3, 2, 2)
    calls = {}
    batch = SimpleNamespace(
        tensor=torch.randn(1, 9, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=9),
    )

    def fake_add_first_frame_conditioning(**kwargs):
        calls["first_frame_shape"] = kwargs["first_frame"].shape
        return sentinel

    monkeypatch.setattr(
        wan22_i2v_module,
        "add_first_frame_conditioning",
        fake_add_first_frame_conditioning,
    )

    model.get_noise_prediction(
        latent_model_input=latent_model_input,
        timestep=torch.tensor([1]),
        text_embeddings=_make_prompt_embeds(),
        batch=batch,
        force_t2i_single_frame=True,
    )

    assert calls["first_frame_shape"] == (1, 3, 8, 8)
    assert torch.equal(model.model.hidden_states[-1], sentinel)


def _make_concept_slider_trainer(arch):
    trainer = object.__new__(ConceptSliderTrainer)
    trainer.sd = SimpleNamespace(
        arch=arch,
        unet=_FakeUnet(),
    )
    trainer.network = _FakeNetwork()
    trainer.train_config = SimpleNamespace(dtype="fp32")
    trainer.device_torch = torch.device("cpu")
    trainer.slider = ConceptSliderTrainerConfig(guidance_strength=1.0, anchor_strength=1.0)
    trainer.anchor_class_embeds = None
    trainer.positive_prompt_embeds = _make_prompt_embeds()
    trainer.target_class_embeds = _make_prompt_embeds()
    trainer.negative_prompt_embeds = _make_prompt_embeds()
    return trainer


def test_concept_slider_sets_force_flag_for_wan22_single_frame():
    trainer = _make_concept_slider_trainer("wan22_14b_i2v")
    calls = []

    def fake_predict_noise(**kwargs):
        calls.append(kwargs.get("force_t2i_single_frame"))
        return kwargs["latents"].clone().requires_grad_(True)

    trainer.sd.predict_noise = fake_predict_noise

    batch = SimpleNamespace(
        tensor=torch.randn(1, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=1),
    )

    trainer.get_guided_loss(
        noisy_latents=torch.randn(1, 16, 1, 2, 2),
        conditional_embeds=_make_prompt_embeds(),
        match_adapter_assist=False,
        network_weight_list=[],
        timesteps=torch.tensor([1]),
        pred_kwargs={},
        batch=batch,
        noise=torch.randn(1, 16, 1, 2, 2),
    )

    assert calls == [True, True, True]


def test_concept_slider_does_not_force_flag_for_wan22_video_batches():
    trainer = _make_concept_slider_trainer("wan22_14b_i2v")
    calls = []

    def fake_predict_noise(**kwargs):
        calls.append(kwargs.get("force_t2i_single_frame"))
        return kwargs["latents"].clone().requires_grad_(True)

    trainer.sd.predict_noise = fake_predict_noise

    batch = SimpleNamespace(
        tensor=torch.randn(1, 9, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=9),
    )

    trainer.get_guided_loss(
        noisy_latents=torch.randn(1, 16, 3, 2, 2),
        conditional_embeds=_make_prompt_embeds(),
        match_adapter_assist=False,
        network_weight_list=[],
        timesteps=torch.tensor([1]),
        pred_kwargs={},
        batch=batch,
        noise=torch.randn(1, 16, 3, 2, 2),
    )

    assert calls == [False, False, False]


def test_concept_slider_does_not_force_flag_for_non_wan_arch():
    trainer = _make_concept_slider_trainer("wan22_14b")
    calls = []

    def fake_predict_noise(**kwargs):
        calls.append(kwargs.get("force_t2i_single_frame"))
        return kwargs["latents"].clone().requires_grad_(True)

    trainer.sd.predict_noise = fake_predict_noise

    batch = SimpleNamespace(
        tensor=torch.randn(1, 3, 8, 8),
        dataset_config=SimpleNamespace(num_frames=1),
    )

    trainer.get_guided_loss(
        noisy_latents=torch.randn(1, 16, 1, 2, 2),
        conditional_embeds=_make_prompt_embeds(),
        match_adapter_assist=False,
        network_weight_list=[],
        timesteps=torch.tensor([1]),
        pred_kwargs={},
        batch=batch,
        noise=torch.randn(1, 16, 1, 2, 2),
    )

    assert calls == [False, False, False]
