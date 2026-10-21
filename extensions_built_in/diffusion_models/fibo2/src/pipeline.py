"""Text conditioning, latent packing and a preview sampler for Fibo 2.0.

Mirrors diffusers' ``BriaFibo2Pipeline`` (huggingface/diffusers#14991) for
text-to-image:

  - prompts: a JSON caption is cleaned the way training cleaned captions (empty
    values and score fields dropped, serialized compactly); anything else, such
    as a short description, is encoded as given. The empty prompt, which caption
    dropout and the default negative prompt produce, becomes the null caption
    Fibo 2.0 was trained with for classifier-free guidance.
  - text features: 15 Qwen3-VL hidden layers, each scaled to unit RMS per token,
    regrouped at the model call into the six 3-layer bundles the transformer
    reads (the Perceiver's, then one per injection block).
  - latents: FLUX.2 VAE latents normalized with the VAE's BatchNorm statistics in
    the 2x2 packed space, stored unpacked as (B, 32, H/8, W/8); the transformer
    patchifies them itself.

Time convention matches ai-toolkit: sigma 1 is pure noise, the transformer takes
sigma in [0, 1] and predicts the velocity ``noise - clean``.
"""

import json
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

# The null caption Fibo 2.0 was trained with for classifier-free guidance: every
# key of the structured caption, every value empty. Encoded as is, never cleaned
DEFAULT_NEGATIVE_PROMPT = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)

# The chat turn each prompt is wrapped in for Qwen3-VL
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

TOKENIZER_MAX_LENGTH = 4096

# Qwen3-VL layers behind the transformer's six text inputs: the Perceiver's, then
# one per injection block
TEXT_ENCODER_OUT_LAYERS = (
    (9, 20, 31),
    (5, 16, 27),
    (7, 18, 29),
    (9, 20, 31),
    (11, 22, 33),
    (13, 24, 35),
)
# the distinct layers, in the order they are stored per caption
ENCODED_LAYERS = tuple(sorted({l for layers in TEXT_ENCODER_OUT_LAYERS for l in layers}))
_BUNDLE_INDEX = [[ENCODED_LAYERS.index(l) for l in layers] for layers in TEXT_ENCODER_OUT_LAYERS]


def clean_caption(prompt: str) -> str:
    """A generation prompt in the form Fibo 2.0 was trained with
    (``BriaFibo2Pipeline._parse_prompt`` + the serialization in ``__call__``)."""
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt
    if not isinstance(caption, dict):
        return prompt

    def empty(value):
        if isinstance(value, float):
            return math.isnan(value)
        return value is None or (isinstance(value, (str, dict, list)) and not value)

    def clean(value):
        if isinstance(value, dict):
            value = {k: clean(v) for k, v in value.items()}
            return {k: v for k, v in value.items() if not empty(v)}
        if isinstance(value, list):
            return [v for v in map(clean, value) if not empty(v)]
        return value

    keys = set(caption)
    scores = ("aesthetic_score", "preference_score")
    caption = {
        key: {k: v for k, v in value.items() if k not in scores}
        if isinstance(value, dict)
        else value
        for key, value in clean(caption).items()
        if key not in scores
    }
    caption = {key: value for key, value in caption.items() if value != {}}
    # exactly a "short_description" and a "text_render" list keep those keys in
    # that order, and the list even when it's empty: the image has no text
    if keys == {"short_description", "text_render"}:
        caption = {
            "short_description": caption.get("short_description") or "",
            "text_render": caption.get("text_render") or [],
        }
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def prepare_prompt(prompt: str) -> str:
    if not prompt.strip() or prompt == DEFAULT_NEGATIVE_PROMPT:
        return DEFAULT_NEGATIVE_PROMPT
    return clean_caption(prompt)


@torch.no_grad()
def encode_prompt(text_encoder, tokenizer, prompt: str, device, dtype) -> torch.Tensor:
    """One prompt at its natural length -> ``(L, len(ENCODED_LAYERS) * hidden)``,
    each layer scaled to unit RMS per token in float32."""
    inputs = tokenizer(
        [PROMPT_TEMPLATE.format(prepare_prompt(prompt))],
        max_length=TOKENIZER_MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    hidden_states = text_encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        output_hidden_states=True,
        logits_to_keep=1,
    ).hidden_states
    layers = []
    for layer in ENCODED_LAYERS:
        states = hidden_states[layer][0].float()
        states = states * torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        layers.append(states.to(dtype))
    return torch.cat(layers, dim=-1)


def bundle_text_features(
    features_list: List[torch.Tensor], device, dtype
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Per-caption ``(L_i, n * hidden)`` features -> the transformer's
    ``(B, 6, L, 3 * hidden)`` bundles, right-padded to the longest caption, and
    the bool mask of real tokens (``None`` when nothing is padded, so attention
    runs unmasked as in the reference pipeline)."""
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    num_layers = len(ENCODED_LAYERS)
    hidden = features_list[0].shape[-1] // num_layers
    layers = torch.zeros(
        len(features_list), max_len, num_layers, hidden, device=device, dtype=dtype
    )
    mask = torch.zeros(len(features_list), max_len, dtype=torch.bool, device=device)
    for i, f in enumerate(features_list):
        layers[i, : f.shape[0]] = f.to(device, dtype).view(f.shape[0], num_layers, hidden)
        mask[i, : f.shape[0]] = True
    bundles = torch.stack(
        [layers[:, :, idx].flatten(2) for idx in _BUNDLE_INDEX], dim=1
    )  # (B, 6, L, 3 * hidden)
    if mask.all():
        mask = None
    return bundles, mask


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def predict_velocity(transformer, latents, sigma, features_list, dtype):
    text, text_mask = bundle_text_features(features_list, latents.device, dtype)
    return transformer(
        hidden_states=latents.to(dtype),
        timestep=sigma,
        encoder_hidden_states=text,
        encoder_attention_mask=text_mask,
        return_dict=False,
    )[0]


class Fibo2Pipeline:
    """Flow-matching Euler sampler for training previews, the denoising loop of
    ``BriaFibo2Pipeline`` on precomputed embeds."""

    def __init__(self, model):
        # the Fibo2Model, for its transformer, scheduler and VAE decode
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds,
        unconditional_embeds,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[Image.Image]:
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer
        scheduler = model.get_sample_scheduler()
        do_cfg = guidance_scale > 1.0 and unconditional_embeds is not None

        if latents is None:
            shape = (
                1,
                transformer.config.in_channels,
                height // model.vae_scale_factor,
                width // model.vae_scale_factor,
            )
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        latents = latents.to(device, dtype=torch.float32)

        # sigmas from 1 down to 1 / steps, shifted by the image size
        patch = transformer.config.patch_size
        image_seq_len = (latents.shape[2] // patch) * (latents.shape[3] // patch)
        cfg = scheduler.config
        mu = calculate_shift(
            image_seq_len,
            cfg.get("base_image_seq_len", 256),
            cfg.get("max_image_seq_len", 4096),
            cfg.get("base_shift", 0.5),
            cfg.get("max_shift", 1.15),
        )
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        timesteps = scheduler.timesteps

        cond = conditional_embeds.text_embeds
        uncond = unconditional_embeds.text_embeds if do_cfg else None
        for i, t in enumerate(timesteps):
            sigma = (t / 1000).expand(latents.shape[0]).to(device)
            pred = predict_velocity(transformer, latents, sigma, cond, dtype).float()
            if do_cfg:
                pred_uncond = predict_velocity(transformer, latents, sigma, uncond, dtype).float()
                pred = pred_uncond + guidance_scale * (pred - pred_uncond)
            # inference engine preview: x0 estimate for this step
            emit = getattr(model, "_emit_sample_step", None)
            if emit is not None and getattr(model, "sample_step_hook", None) is not None:
                emit(latents - float(t / 1000) * pred, i, len(timesteps))
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]

        images = model.decode_latents(latents, device=device, dtype=model.vae_torch_dtype)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
