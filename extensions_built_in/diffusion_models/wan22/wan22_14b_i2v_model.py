import random
import torch
from toolkit.models.wan21.wan_utils import add_first_frame_conditioning
from toolkit.prompt_utils import PromptEmbeds
from PIL import Image
from toolkit.config_modules import GenerateImageConfig
from .wan22_pipeline import Wan22Pipeline

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from torchvision.transforms import functional as TF

from .wan22_14b_model import Wan2214bModel

class Wan2214bI2VModel(Wan2214bModel):
    arch = "wan22_14b_i2v"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_config = kwargs.get("model_config", None)
        if model_config is None and len(args) > 1:
            model_config = args[1]
        model_kwargs = getattr(model_config, "model_kwargs", {}) if model_config is not None else {}
        self.image_i2v_conditioning = model_kwargs.get("image_i2v_conditioning", False)
        self.image_i2v_conditioning_prob = float(
            model_kwargs.get("image_i2v_conditioning_prob", 0.2)
        )
        self.image_i2v_conditioning_prob = max(
            0.0, min(1.0, self.image_i2v_conditioning_prob)
        )

    @staticmethod
    def degrade_image_i2v_conditioning(first_frames: torch.Tensor) -> torch.Tensor:
        degraded = first_frames
        if degraded.shape[-1] >= 3 and degraded.shape[-2] >= 3:
            degraded = TF.gaussian_blur(degraded, kernel_size=[3, 3], sigma=[0.1, 1.0])

        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        degraded = degraded * brightness
        channel_mean = degraded.mean(dim=(-2, -1), keepdim=True)
        degraded = (degraded - channel_mean) * contrast + channel_mean

        noise = torch.randn_like(degraded) * 0.025
        degraded = degraded + noise

        return degraded.clamp(-1.0, 1.0)

    @staticmethod
    def _get_first_frames_from_batch(batch: DataLoaderBatchDTO) -> torch.Tensor:
        frames = batch.tensor
        if len(frames.shape) == 4:
            return frames
        if len(frames.shape) == 5:
            return frames[:, 0]
        raise ValueError(f"Unknown frame shape {frames.shape}")

    @staticmethod
    def _is_single_frame_batch(batch: DataLoaderBatchDTO) -> bool:
        if batch is None:
            return False

        frames = getattr(batch, "tensor", None)
        if frames is not None:
            if len(frames.shape) == 4:
                return True
            if len(frames.shape) == 5:
                return frames.shape[1] == 1

        batch_num_frames = getattr(batch, "num_frames", None)
        if batch_num_frames is not None:
            return batch_num_frames == 1

        return getattr(getattr(batch, "dataset_config", None), "num_frames", None) == 1

    def _should_use_image_i2v_conditioning(self) -> bool:
        return (
            getattr(self, "image_i2v_conditioning", False)
            and random.random() < getattr(self, "image_i2v_conditioning_prob", 0.2)
        )
    
    
    def generate_single_image(
        self,
        pipeline: Wan22Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        
        # todo 
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)

        num_frames = (
            (gen_config.num_frames - 1) // 4
        ) * 4 + 1  # make sure it is divisible by 4 + 1
        gen_config.num_frames = num_frames

        height = gen_config.height
        width = gen_config.width
        first_frame_n1p1 = None
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img).convert("RGB")

            d = self.get_bucket_divisibility()

            # make sure they are divisible by d
            height = height // d * d
            width = width // d * d

            # resize the control image
            control_img = control_img.resize((width, height), Image.LANCZOS)

            # 5. Prepare latent variables
            # num_channels_latents = self.transformer.config.in_channels
            num_channels_latents = 16
            latents = pipeline.prepare_latents(
                1,
                num_channels_latents,
                height,
                width,
                gen_config.num_frames,
                torch.float32,
                self.device_torch,
                generator,
                None,
            ).to(self.torch_dtype)

            first_frame_n1p1 = (
                TF.to_tensor(control_img)
                .unsqueeze(0)
                .to(self.device_torch, dtype=self.torch_dtype)
                * 2.0
                - 1.0
            )  # normalize to [-1, 1]
            
            # Add conditioning using the standalone function
            gen_config.latents = add_first_frame_conditioning(
                latent_model_input=latents,
                first_frame=first_frame_n1p1,
                vae=self.vae
            )

        output = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            height=height,
            width=width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="pil",
            **extra,
        )[0]

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item  # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img
    
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: DataLoaderBatchDTO,
        force_t2i_single_frame: bool = False,
        **kwargs
    ):
        # videos come in (bs, num_frames, channels, height, width)
        # images come in (bs, channels, height, width)
        is_single_frame_batch = self._is_single_frame_batch(batch)
        should_use_degraded_image_conditioning = (
            is_single_frame_batch
            and not force_t2i_single_frame
            and self._should_use_image_i2v_conditioning()
        )

        if is_single_frame_batch and not should_use_degraded_image_conditioning:
            target_in_channels = getattr(
                getattr(self.model, "patch_embedding", None), "in_channels", None
            )
            if target_in_channels is None and hasattr(self.model, "config"):
                target_in_channels = getattr(self.model.config, "in_channels", None)
            if target_in_channels is None and hasattr(self.model, "patch_embedding"):
                target_in_channels = self.model.patch_embedding.weight.shape[1]

            if target_in_channels is None or target_in_channels <= latent_model_input.shape[1]:
                conditioned_latent = latent_model_input
            else:
                extra_channels = target_in_channels - latent_model_input.shape[1]
                empty_conditioning = torch.zeros(
                    latent_model_input.shape[0],
                    extra_channels,
                    latent_model_input.shape[2],
                    latent_model_input.shape[3],
                    latent_model_input.shape[4],
                    device=latent_model_input.device,
                    dtype=latent_model_input.dtype,
                )
                conditioned_latent = torch.cat(
                    [latent_model_input, empty_conditioning], dim=1
                )
        else:
            with torch.no_grad():
                first_frames = self._get_first_frames_from_batch(batch)
                if should_use_degraded_image_conditioning:
                    first_frames = self.degrade_image_i2v_conditioning(first_frames)

                # Add conditioning using the standalone function
                conditioned_latent = add_first_frame_conditioning(
                    latent_model_input=latent_model_input,
                    first_frame=first_frames,
                    vae=self.vae
                )

        noise_pred = self.model(
            hidden_states=conditioned_latent,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred
