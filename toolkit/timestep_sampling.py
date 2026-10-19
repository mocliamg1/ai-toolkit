"""Batch-specific timestep policy shared by sampling and loss calculation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolkit.config_modules import TrainConfig
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


TIMESTEP_TYPES = (
    "sigmoid", "linear", "shift", "flux_shift", "lumina2_shift",
    "weighted", "lognorm_blend", "next_sample", "one_step", "two_step",
    "four_step", "eight_step",
)


def validate_timestep_overrides(train_config: "TrainConfig") -> None:
    enabled = False
    for field in ("image_timestep_type", "video_timestep_type"):
        value = getattr(train_config, field, None)
        if value is None or value == "":
            continue
        if value not in TIMESTEP_TYPES:
            raise ValueError(
                f"train.{field} must be one of {', '.join(TIMESTEP_TYPES)}, "
                f"or null to inherit timestep_type; got {value!r}"
            )
        enabled = True
    if not enabled:
        return
    if train_config.noise_scheduler != "flowmatch":
        raise ValueError("Image/video timestep overrides require noise_scheduler: flowmatch")
    if train_config.linear_timesteps or train_config.linear_timesteps2:
        raise ValueError(
            "Image/video timestep overrides require linear_timesteps and "
            "linear_timesteps2 to be disabled; use timestep_type instead"
        )


def resolve_timestep_type(
    train_config: "TrainConfig", batch: "DataLoaderBatchDTO"
) -> str:
    # Use target frames, not dataset configuration or latent shape: a mixed
    # video dataset can yield images, and image latents can still be 5-D.
    num_frames = getattr(batch, "num_frames", None)
    if num_frames == 1:
        override = getattr(train_config, "image_timestep_type", None)
    elif num_frames is not None and num_frames > 1:
        override = getattr(train_config, "video_timestep_type", None)
    else:
        override = None
    return override or train_config.timestep_type
