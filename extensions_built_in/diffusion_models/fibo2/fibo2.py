"""Fibo 2.0 (briaai/Fibo-2.0-dev) for ai-toolkit, text-to-image.

  - denoiser: ``BriaFibo2Transformer2DModel`` (``src/transformer.py``), an 8B
    Z-Image style single-stream DiT; a Perceiver squeezes the text features
    into gist tokens that join the image tokens, and five blocks also read the
    text encoder through gated cross-attention
  - text encoder: Qwen3-VL-4B-Instruct (the repo's text_encoder/), 15 hidden
    layers regrouped into six 3-layer bundles (``src/pipeline.py``)
  - VAE: FLUX.2 (the repo's vae/), BatchNorm normalized latents, 32 channels at
    1/8 resolution

Captions can be Fibo's structured JSON captions or plain text (the model was
trained on short descriptions alone too). ``model.name_or_path`` is the gated
diffusers repo (accept the license on the hub and set HF_TOKEN), a local copy
of it, or a transformer ``.safetensors`` with ``extras_name_or_path`` naming the
repo for the rest.

Flow convention matches ai-toolkit (sigma 1 = noise, target = noise - clean).
"""

import os
from typing import List, Optional

import torch

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextEncoder
from toolkit.models.v2.vae.autoencoder_kl_flux2 import Flux2KLVAE
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.pipeline import (
    Fibo2Pipeline,
    encode_prompt,
    patchify_latents,
    predict_velocity,
    unpatchify_latents,
)
from .src.transformer import BriaFibo2Transformer2DModel

FIBO2_REPO = "briaai/Fibo-2.0-dev"

# used until the repo's scheduler/ config is read in load_model: FLUX style
# resolution-dependent shifting, the defaults of the reference pipeline's
# calculate_shift
scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "time_shift_type": "exponential",
}


class Fibo2Model(BaseModel):
    arch = "fibo2"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["BriaFibo2Transformer2DModel"]
        self.use_old_lokr_format = False
        self.vae_scale_factor = 8
        self.scheduler_config = dict(scheduler_config)

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_sample_scheduler(self):
        return CustomFlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config)

    def get_bucket_divisibility(self):
        return self.vae_scale_factor * 2  # 8 for the VAE, 2 for the patch size

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_scheduler_config(self, base_model_path):
        try:
            config = CustomFlowMatchEulerDiscreteScheduler.load_config(
                base_model_path, subfolder="scheduler"
            )
        except Exception as e:
            self.print_and_status_update(
                f"Could not read the scheduler config from {base_model_path} ({e}); "
                "using the reference pipeline's shift defaults"
            )
            return
        self.scheduler_config = dict(config)

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Fibo 2.0 model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path
        if os.path.isdir(model_path) and os.path.isdir(
            os.path.join(model_path, "text_encoder")
        ):
            # a full local checkpoint is its own base
            base_model_path = model_path
        elif model_path.endswith(".safetensors") and base_model_path == model_path:
            base_model_path = FIBO2_REPO

        self.print_and_status_update("Loading transformer")
        transformer = BriaFibo2Transformer2DModel.load(
            model_path,
            config_path=base_model_path if model_path.endswith(".safetensors") else None,
            **self.component_load_kwargs("transformer"),
        )
        flush()

        self.print_and_status_update("Loading Qwen3-VL text encoder")
        tokenizer = Qwen3VLTextEncoder.load_tokenizer(base_model_path, subfolder="processor")
        text_encoder = Qwen3VLTextEncoder.load_model(
            base_model_path, dtype=dtype, subfolder="text_encoder"
        )
        # text-to-image only ever encodes text
        text_encoder.drop_vision_tower()
        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()

        self.print_and_status_update("Loading VAE")
        vae = Flux2KLVAE.load_model(base_model_path, dtype=self.vae_torch_dtype)
        vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)
        vae.eval()
        vae.requires_grad_(False)

        self._load_scheduler_config(base_model_path)
        self.noise_scheduler = CustomFlowMatchEulerDiscreteScheduler.from_config(
            self.scheduler_config
        )

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = transformer
        self.pipeline = Fibo2Pipeline(self)
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # VAE
    # ------------------------------------------------------------------
    def _bn_stats(self, device, dtype):
        bn = self.vae.bn
        mean = bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
        std = torch.sqrt(
            bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(device, dtype)
        return mean, std

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        images = image_list
        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        images = images.to(device, dtype=dtype)

        # as the reference pipeline encodes: the posterior mode, normalized in
        # the 2x2 packed space, stored unpacked
        latents = patchify_latents(self.vae.encode(images).latent_dist.mode().float())
        mean, std = self._bn_stats(latents.device, latents.dtype)
        latents = unpatchify_latents((latents - mean) / std)
        return latents.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = patchify_latents(latents.to(device, torch.float32))
        mean, std = self._bn_stats(latents.device, latents.dtype)
        latents = unpatchify_latents(latents * std + mean)
        return self.vae.decode(latents.to(dtype), return_dict=False)[0]

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------
    def get_prompt_embeds(self, prompt) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        # one (L, 15 * 2560) tensor per caption at its natural length; the six
        # bundles are rebuilt and padded at the model call
        features = [
            encode_prompt(
                self.text_encoder, self.tokenizer, p, self.device_torch, self.torch_dtype
            )
            for p in prompt
        ]
        return AdvancedPromptEmbeds(text_embeds=features)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 32, h, w)
        timestep: torch.Tensor,  # 0..1000, 1000 = pure noise
        text_embeddings: AdvancedPromptEmbeds,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sigma = timestep.to(self.device_torch, dtype=torch.float32) / 1000.0
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.shape[0] != latent_model_input.shape[0]:
            sigma = sigma.expand(latent_model_input.shape[0])

        return predict_velocity(
            self.transformer,
            latent_model_input.to(self.device_torch),
            sigma,
            text_embeddings.text_embeds,
            self.torch_dtype,
        )

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    # ------------------------------------------------------------------
    # Sampling (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return Fibo2Pipeline(self)

    def generate_single_image(
        self,
        pipeline: Fibo2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        return pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
        )[0]

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save_model(self, output_path, meta, save_dtype):
        transformer: BriaFibo2Transformer2DModel = unwrap_model(self.model)
        if not output_path.endswith(".safetensors"):
            output_path += ".safetensors"
        transformer.save_model(
            output_path,
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name=self.arch),
        )

    def get_quantization_exclude_modules(self):
        return BriaFibo2Transformer2DModel.get_quantization_exclude_modules()

    def get_base_model_version(self):
        return self.arch

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["noise_refiner", "layers"]

    lora_keys_use_comfy_prefix = True
