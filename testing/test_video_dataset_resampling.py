import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig
from toolkit.dataloader_mixins import (
    build_time_resampled_frame_indices,
    latent_frame_count_to_source_count,
    normalize_video_frame_count,
    source_frame_count_to_latent_count,
)


def test_dataset_config_accepts_max_frames_aliases_and_forces_video_disk_cache():
    snake = DatasetConfig(max_frames=81, fps=16)
    dashed = DatasetConfig(**{"max-frames": 81, "fps": 16})
    legacy = DatasetConfig(num_frames=81, fps=16)

    assert snake.max_frames == 81
    assert dashed.max_frames == 81
    assert legacy.max_frames == 81
    assert snake.num_frames == 81
    assert dashed.num_frames == 81
    assert legacy.num_frames == 81
    assert snake.cache_latents_to_disk is True
    assert dashed.cache_latents_to_disk is True
    assert legacy.cache_latents_to_disk is True


def test_dataset_config_rejects_video_without_positive_fps():
    try:
        DatasetConfig(max_frames=81, fps=0)
    except ValueError as exc:
        assert "fps" in str(exc)
    else:
        raise AssertionError("Expected DatasetConfig to reject fps=0 for video")


def test_time_resampling_preserves_duration_when_downsampling_30fps_to_16fps():
    indices = build_time_resampled_frame_indices(
        total_frames=300,
        source_fps=30.0,
        target_fps=16,
        temporal_compression=1,
    )

    assert len(indices) == 160
    assert indices[:6] == [0, 2, 4, 6, 8, 9]
    assert indices[-1] == 298


def test_time_resampling_duplicates_frames_when_target_fps_is_higher():
    indices = build_time_resampled_frame_indices(
        total_frames=10,
        source_fps=10.0,
        target_fps=20,
        temporal_compression=1,
    )

    assert len(indices) == 20
    assert indices[:4] == [0, 0, 1, 2]
    assert indices[-1] == 9


def test_video_frame_count_normalizes_to_temporal_compression():
    assert normalize_video_frame_count(40, 4) == 37
    assert source_frame_count_to_latent_count(40, 4) == 10
    assert latent_frame_count_to_source_count(10, 4) == 37


if __name__ == "__main__":
    test_dataset_config_accepts_max_frames_aliases_and_forces_video_disk_cache()
    test_dataset_config_rejects_video_without_positive_fps()
    test_time_resampling_preserves_duration_when_downsampling_30fps_to_16fps()
    test_time_resampling_duplicates_frames_when_target_fps_is_higher()
    test_video_frame_count_normalizes_to_temporal_compression()
