"""Run with: python -m unittest testing.test_timestep_sampling"""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

# Load the dependency-free policy directly: toolkit.__init__ otherwise imports
# huggingface_hub even for unit tests that do not use the model runtime.
POLICY_PATH = Path(__file__).resolve().parents[1] / "toolkit" / "timestep_sampling.py"
spec = importlib.util.spec_from_file_location("timestep_sampling", POLICY_PATH)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
resolve_timestep_type = policy.resolve_timestep_type
validate_timestep_overrides = policy.validate_timestep_overrides


def config(**kwargs):
    values = dict(
        timestep_type="shift",
        image_timestep_type=None,
        video_timestep_type=None,
        noise_scheduler="flowmatch",
        linear_timesteps=False,
        linear_timesteps2=False,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


class TimestepPolicyTest(unittest.TestCase):
    def test_old_configs_keep_global_setting(self):
        old_config = SimpleNamespace(timestep_type="weighted")
        for frames in (1, 5, 39, 107):
            with self.subTest(frames=frames):
                self.assertEqual(
                    resolve_timestep_type(old_config, SimpleNamespace(num_frames=frames)),
                    "weighted",
                )

    def test_alternating_batches_do_not_change_global_config(self):
        cfg = config(image_timestep_type="sigmoid", video_timestep_type="linear")
        before = vars(cfg).copy()
        for frames, expected in ((1, "sigmoid"), (39, "linear"), (1, "sigmoid"), (5, "linear")):
            self.assertEqual(resolve_timestep_type(cfg, SimpleNamespace(num_frames=frames)), expected)
        self.assertEqual(vars(cfg), before)

    def test_each_override_falls_back_independently(self):
        for unset in (None, ""):
            with self.subTest(unset=unset):
                cfg = config(image_timestep_type="sigmoid", video_timestep_type=unset)
                self.assertEqual(resolve_timestep_type(cfg, SimpleNamespace(num_frames=39)), "shift")
                cfg = config(image_timestep_type=unset, video_timestep_type="weighted")
                self.assertEqual(resolve_timestep_type(cfg, SimpleNamespace(num_frames=1)), "shift")

    def test_target_frames_take_precedence_over_video_dataset_and_references(self):
        batch = SimpleNamespace(
            num_frames=1,
            dataset_config=SimpleNamespace(num_frames=107, auto_frame_count=True),
            control_video_paths_list=[["reference.mp4"]],
            latents=SimpleNamespace(shape=(1, 16, 1, 32, 32)),
        )
        self.assertEqual(resolve_timestep_type(config(image_timestep_type="sigmoid"), batch), "sigmoid")

    def test_unknown_modality_keeps_global_setting(self):
        cfg = config(image_timestep_type="sigmoid", video_timestep_type="linear")
        for batch in (None, SimpleNamespace(), SimpleNamespace(num_frames=None), SimpleNamespace(num_frames=0)):
            self.assertEqual(resolve_timestep_type(cfg, batch), "shift")

    def test_advanced_types_are_preserved_for_downstream_sampling(self):
        for kind in ("weighted", "next_sample", "one_step", "two_step", "four_step", "eight_step",
                     "lognorm_blend", "flux_shift", "lumina2_shift"):
            with self.subTest(kind=kind):
                cfg = config(image_timestep_type=kind)
                validate_timestep_overrides(cfg)
                self.assertEqual(resolve_timestep_type(cfg, SimpleNamespace(num_frames=1)), kind)
                self.assertEqual(resolve_timestep_type(cfg, SimpleNamespace(num_frames=39)), "shift")

    def test_invalid_overrides_fail_early_with_field_name(self):
        for field in ("image_timestep_type", "video_timestep_type"):
            for value in ("sigmod", False, 12, ["shift"]):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        validate_timestep_overrides(config(**{field: value}))

    def test_non_flowmatch_overrides_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "flowmatch"):
            validate_timestep_overrides(config(noise_scheduler="ddpm", image_timestep_type="sigmoid"))

    def test_legacy_linear_flags_cannot_silently_override_selection(self):
        for flag in ("linear_timesteps", "linear_timesteps2"):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, "linear_timesteps"):
                    validate_timestep_overrides(config(video_timestep_type="shift", **{flag: True}))

    def test_validation_does_not_change_legacy_jobs(self):
        validate_timestep_overrides(config(noise_scheduler="ddpm", linear_timesteps=True))
        validate_timestep_overrides(config(image_timestep_type="", linear_timesteps2=True))


HAS_SCHEDULER_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "diffusers", "huggingface_hub")
)


@unittest.skipUnless(HAS_SCHEDULER_DEPS, "requires torch and diffusers")
class TimestepSchedulerTest(unittest.TestCase):
    def test_switching_image_video_image_changes_actual_noise_distribution(self):
        import torch
        from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

        scheduler = CustomFlowMatchEulerDiscreteScheduler(shift=12, use_dynamic_shifting=False)
        cfg = config(image_timestep_type="sigmoid")
        for frames, expected_median in ((1, 500), (39, 923), (1, 500)):
            with self.subTest(frames=frames):
                torch.manual_seed(42)
                scheduler.set_train_timesteps(
                    1000, device="cpu",
                    timestep_type=resolve_timestep_type(cfg, SimpleNamespace(num_frames=frames)),
                )
                self.assertAlmostEqual(scheduler.timesteps.median().item(), expected_median, delta=30)
                noisy = scheduler.add_noise(torch.zeros(1), torch.ones(1), scheduler.timesteps[500:501])
                torch.testing.assert_close(noisy, scheduler.timesteps[500:501] / 1000)

    def test_weighted_override_uses_matching_grid_and_weights(self):
        import torch
        from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
        from toolkit.timestep_weighing.default_weighing_scheme import default_weighing_scheme

        scheduler = CustomFlowMatchEulerDiscreteScheduler(shift=12, use_dynamic_shifting=False)
        kind = resolve_timestep_type(config(image_timestep_type="weighted"), SimpleNamespace(num_frames=1))
        scheduler.set_train_timesteps(1000, device="cpu", timestep_type=kind)
        indices = [0, 499, 999]
        weights = scheduler.get_weights_for_timesteps(scheduler.timesteps[indices], timestep_type=kind)
        torch.testing.assert_close(weights, torch.tensor([default_weighing_scheme[i] for i in indices]))


if __name__ == "__main__":
    unittest.main()
