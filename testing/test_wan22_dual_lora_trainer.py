from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from extensions_built_in.sd_trainer.Wan22DualLoraTrainer import Wan22DualLoraTrainer
from toolkit.config_modules import ModelConfig


class _FakeLoraModule(torch.nn.Module):
    def __init__(self, name: str, in_dim: int = 4, rank: int = 2, out_dim: int = 4):
        super().__init__()
        self.lora_name = name
        self.lora_down = torch.nn.Linear(in_dim, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, out_dim, bias=False)


class _FakeNetwork:
    def __init__(self, modules):
        self._modules = modules
        self.force_to_calls = []

    def get_all_modules(self):
        return self._modules

    def parameters(self):
        for module in self._modules:
            yield from module.parameters()

    def force_to(self, *args, **kwargs):
        self.force_to_calls.append((args, kwargs))
        return self


class _FakeOptimizer:
    def __init__(self):
        self.step_calls = 0
        self.zero_grad_calls = []

    def step(self):
        self.step_calls += 1

    def zero_grad(self, *args, **kwargs):
        self.zero_grad_calls.append((args, kwargs))


class _FakeScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


class _FakeAccelerator:
    def __init__(self):
        self.clip_grad_norm_calls = []

    def clip_grad_norm_(self, params, max_norm):
        self.clip_grad_norm_calls.append((params, max_norm))


class _FakeTimer:
    def __call__(self, _name):
        return self

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _make_validation_subject(
    primary_arch="wan22_14b_i2v",
    secondary_arch="wan22_14b",
    primary_high=True,
    primary_low=True,
    secondary_high=True,
    secondary_low=True,
    network_type="lora",
    train_text_encoder=False,
):
    trainer = object.__new__(Wan22DualLoraTrainer)
    trainer.model_config = SimpleNamespace(
        arch=primary_arch,
        model_kwargs={
            "train_high_noise": primary_high,
            "train_low_noise": primary_low,
        },
        refiner_name_or_path=None,
    )
    trainer.dual_t2v_model_config = SimpleNamespace(
        arch=secondary_arch,
        model_kwargs={
            "train_high_noise": secondary_high,
            "train_low_noise": secondary_low,
        },
    )
    trainer.network_config = SimpleNamespace(type=network_type)
    trainer.train_config = SimpleNamespace(
        train_text_encoder=train_text_encoder,
        train_refiner=False,
    )
    trainer.adapter_config = None
    trainer.embed_config = None
    trainer.decorator_config = None
    trainer.get_conf = lambda *args, **kwargs: None
    return trainer


def _make_paired_hook_subject():
    trainer = object.__new__(Wan22DualLoraTrainer)
    trainer.primary_sd = SimpleNamespace(
        name="i2v_sd",
        is_multistage=True,
        trainable_multistage_boundaries=[0, 1],
        multistage_boundaries=[0.9, 0.0],
    )
    trainer.secondary_sd = SimpleNamespace(
        name="t2v_sd",
        is_multistage=True,
        trainable_multistage_boundaries=[0, 1],
        multistage_boundaries=[0.875, 0.0],
    )
    trainer.primary_network = _FakeNetwork([])
    trainer.secondary_network = _FakeNetwork([])
    trainer.primary_model_config = SimpleNamespace(arch="wan22_14b_i2v", low_vram=False)
    trainer.dual_t2v_model_config = SimpleNamespace(arch="wan22_14b", low_vram=False)
    trainer.device_torch = torch.device("cpu")
    trainer.dual_offload_inactive_to_cpu = False
    trainer.active_dual_mode = None
    trainer.dual_training_mode = "paired"
    trainer.dual_i2v_steps = 8
    trainer.dual_t2v_steps = 2
    trainer._active_paired_loss_weight = 1.0
    trainer.current_boundary_index = 0
    trainer.steps_this_boundary = 10
    trainer.train_config = SimpleNamespace(
        optimizer="adamw",
        max_grad_norm=1.0,
        switch_boundary_every=10,
    )
    trainer.params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]
    trainer.optimizer = _FakeOptimizer()
    trainer.lr_scheduler = _FakeScheduler()
    trainer.accelerator = _FakeAccelerator()
    trainer.timer = _FakeTimer()
    trainer.adapter = None
    trainer.embedding = None
    trainer.ema = None
    trainer.is_grad_accumulation_step = False
    trainer.end_of_training_loop = lambda: None
    return trainer


def test_dual_schedule_repeats_i2v_then_t2v():
    trainer = object.__new__(Wan22DualLoraTrainer)
    trainer.dual_i2v_steps = 8
    trainer.dual_t2v_steps = 2
    trainer.step_num = 0

    modes = [trainer._get_dual_mode_for_step(step) for step in range(12)]

    assert modes == [
        "i2v",
        "i2v",
        "i2v",
        "i2v",
        "i2v",
        "i2v",
        "i2v",
        "i2v",
        "t2v",
        "t2v",
        "i2v",
        "i2v",
    ]


def test_paired_loss_weights_follow_step_ratio():
    trainer = object.__new__(Wan22DualLoraTrainer)
    trainer.dual_i2v_steps = 8
    trainer.dual_t2v_steps = 2

    i2v_weight, t2v_weight = trainer._get_paired_loss_weights()

    assert i2v_weight == pytest.approx(0.8)
    assert t2v_weight == pytest.approx(0.2)


def test_normalize_process_config_defaults_train_refiner_false():
    config = OrderedDict(
        {
            "model": {
                "name_or_path": "ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16",
                "arch": "wan22_14b_i2v_t2v",
            },
            "train": {},
        }
    )

    normalized = Wan22DualLoraTrainer.normalize_process_config(config)

    assert normalized["model"]["arch"] == "wan22_14b_i2v"
    assert normalized["model"]["model_kwargs"]["load_trainable_stages_only"] is True
    assert normalized["train"]["train_refiner"] is False
    assert "train_refiner" not in config["train"]


def test_normalize_process_config_preserves_explicit_train_refiner_value():
    config = OrderedDict(
        {
            "model": {
                "name_or_path": "ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16",
                "arch": "wan22_14b_i2v_t2v",
            },
            "train": {
                "train_refiner": True,
            },
        }
    )

    normalized = Wan22DualLoraTrainer.normalize_process_config(config)

    assert normalized["train"]["train_refiner"] is True
    assert normalized["model"]["model_kwargs"]["load_trainable_stages_only"] is True


def test_dual_t2v_model_config_preserves_base_lora_merge_fields(monkeypatch):
    def fake_sd_trainer_init(self, process_id, job, config, **kwargs):
        self.model_config = ModelConfig(**config["model"])
        self.train_config = SimpleNamespace(
            dtype="bf16",
            train_text_encoder=False,
            train_refiner=False,
        )
        self.network_config = SimpleNamespace(type="lora")
        self.adapter_config = None
        self.embed_config = None
        self.decorator_config = None

        def fake_get_conf(key, default=None, required=False):
            if required and key not in config:
                raise ValueError(f"Missing required config key {key}")
            return config.get(key, default)

        self.get_conf = fake_get_conf

    monkeypatch.setattr(SDTrainer, "__init__", fake_sd_trainer_init)

    config = OrderedDict(
        {
            "model": {
                "name_or_path": "ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16",
                "arch": "wan22_14b_i2v_t2v",
                "model_kwargs": {
                    "train_high_noise": True,
                    "train_low_noise": True,
                },
            },
            "dual_model": {
                "t2v_model": {
                    "name_or_path": "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
                    "arch": "wan22_14b",
                    "lora_path": [{"path": "t2v_base.safetensors", "strength": 0.75}],
                    "lora_merge_strength": 0.9,
                    "high_noise_lora_path": "t2v_high.safetensors",
                    "high_noise_lora_merge_strength": 0.8,
                    "low_noise_lora_path": [
                        {"path": "t2v_low_a.safetensors", "strength": 0.7},
                        {"path": "t2v_low_b.safetensors", "strength": 0.6},
                    ],
                    "low_noise_lora_merge_strength": 0.5,
                    "model_kwargs": {
                        "train_high_noise": True,
                        "train_low_noise": True,
                    },
                },
            },
        }
    )

    trainer = Wan22DualLoraTrainer(0, object(), config)

    assert trainer.model_config.arch == "wan22_14b_i2v"
    assert trainer.dual_training_mode == "alternating"
    assert trainer.model_config.model_kwargs["load_trainable_stages_only"] is True
    assert trainer.dual_t2v_model_config.model_kwargs["load_trainable_stages_only"] is True
    assert trainer.dual_t2v_model_config.lora_path == [
        {"path": "t2v_base.safetensors", "strength": 0.75}
    ]
    assert trainer.dual_t2v_model_config.lora_merge_strength == 0.9
    assert trainer.dual_t2v_model_config.high_noise_lora_path == "t2v_high.safetensors"
    assert trainer.dual_t2v_model_config.high_noise_lora_merge_strength == 0.8
    assert trainer.dual_t2v_model_config.low_noise_lora_path == [
        {"path": "t2v_low_a.safetensors", "strength": 0.7},
        {"path": "t2v_low_b.safetensors", "strength": 0.6},
    ]
    assert trainer.dual_t2v_model_config.low_noise_lora_merge_strength == 0.5


def test_constructor_accepts_paired_training_mode(monkeypatch):
    def fake_sd_trainer_init(self, process_id, job, config, **kwargs):
        self.model_config = ModelConfig(**config["model"])
        self.train_config = SimpleNamespace(
            dtype="bf16",
            train_text_encoder=False,
            train_refiner=False,
        )
        self.network_config = SimpleNamespace(type="lora")
        self.adapter_config = None
        self.embed_config = None
        self.decorator_config = None
        self.get_conf = lambda key, default=None, required=False: config.get(key, default)

    monkeypatch.setattr(SDTrainer, "__init__", fake_sd_trainer_init)

    trainer = Wan22DualLoraTrainer(
        0,
        object(),
        OrderedDict(
            {
                "model": {
                    "name_or_path": "i2v",
                    "arch": "wan22_14b_i2v_t2v",
                    "model_kwargs": {
                        "train_high_noise": True,
                        "train_low_noise": True,
                    },
                },
                "dual_model": {
                    "training_mode": "paired",
                    "t2v_model": {
                        "name_or_path": "t2v",
                        "arch": "wan22_14b",
                        "model_kwargs": {
                            "train_high_noise": True,
                            "train_low_noise": True,
                        },
                    },
                },
            }
        ),
    )

    assert trainer.dual_training_mode == "paired"


def test_constructor_rejects_unknown_training_mode(monkeypatch):
    def fake_sd_trainer_init(self, process_id, job, config, **kwargs):
        self.model_config = ModelConfig(**config["model"])
        self.train_config = SimpleNamespace(
            dtype="bf16",
            train_text_encoder=False,
            train_refiner=False,
        )
        self.network_config = SimpleNamespace(type="lora")
        self.adapter_config = None
        self.embed_config = None
        self.decorator_config = None
        self.get_conf = lambda key, default=None, required=False: config.get(key, default)

    monkeypatch.setattr(SDTrainer, "__init__", fake_sd_trainer_init)

    with pytest.raises(ValueError, match="training_mode"):
        Wan22DualLoraTrainer(
            0,
            object(),
            OrderedDict(
                {
                    "model": {
                        "name_or_path": "i2v",
                        "arch": "wan22_14b_i2v_t2v",
                    },
                    "dual_model": {
                        "training_mode": "invalid",
                        "t2v_model": {
                            "name_or_path": "t2v",
                            "arch": "wan22_14b",
                        },
                    },
                }
            ),
        )


def test_tie_secondary_lora_parameters_shares_matching_parameter_objects():
    primary_shared = _FakeLoraModule("shared")
    primary_network = _FakeNetwork([primary_shared])

    secondary_shared = _FakeLoraModule("shared")
    secondary_extra = _FakeLoraModule("secondary_only")
    secondary_network = _FakeNetwork([secondary_shared, secondary_extra])

    module_count, tensor_count = Wan22DualLoraTrainer._tie_secondary_lora_parameters(
        primary_network,
        secondary_network,
    )

    assert module_count == 1
    assert tensor_count == 2
    assert secondary_shared.lora_down.weight is primary_shared.lora_down.weight
    assert secondary_shared.lora_up.weight is primary_shared.lora_up.weight
    assert all(not param.requires_grad for param in secondary_extra.parameters())


def test_tie_secondary_lora_parameters_skips_shape_mismatches():
    primary_network = _FakeNetwork([_FakeLoraModule("mismatch", out_dim=4)])
    secondary_mismatch = _FakeLoraModule("mismatch", out_dim=8)
    original_secondary_up = secondary_mismatch.lora_up.weight
    secondary_network = _FakeNetwork([secondary_mismatch])

    module_count, tensor_count = Wan22DualLoraTrainer._tie_secondary_lora_parameters(
        primary_network,
        secondary_network,
    )

    assert module_count == 0
    assert tensor_count == 0
    assert secondary_mismatch.lora_up.weight is original_secondary_up
    assert all(not param.requires_grad for param in secondary_mismatch.parameters())


def test_dedupe_optimizer_params_removes_duplicate_parameter_references():
    p1 = torch.nn.Parameter(torch.zeros(1))
    p2 = torch.nn.Parameter(torch.ones(1))

    params = Wan22DualLoraTrainer._dedupe_optimizer_params(
        [
            {"params": [p1, p2, p1], "lr": 1.0},
            {"params": [p2], "lr": 2.0},
        ]
    )

    assert len(params) == 1
    assert params[0]["params"] == [p1, p2]


def test_optimizer_params_keep_only_shared_primary_parameter_objects():
    primary_shared = _FakeLoraModule("shared")
    primary_extra = _FakeLoraModule("primary_only")
    primary_network = _FakeNetwork([primary_shared, primary_extra])
    secondary_network = _FakeNetwork([_FakeLoraModule("shared")])

    Wan22DualLoraTrainer._tie_secondary_lora_parameters(primary_network, secondary_network)
    shared_param_ids = Wan22DualLoraTrainer._get_shared_parameter_ids(
        primary_network,
        secondary_network,
    )
    Wan22DualLoraTrainer._freeze_unshared_primary_parameters(
        primary_network,
        shared_param_ids,
    )

    params = Wan22DualLoraTrainer._dedupe_optimizer_params(
        [
            {
                "params": list(primary_network.parameters()),
                "lr": 1.0,
            }
        ],
        keep_param_ids=shared_param_ids,
    )

    assert params[0]["params"] == [
        primary_shared.lora_down.weight,
        primary_shared.lora_up.weight,
    ]
    assert all(param.requires_grad for param in params[0]["params"])
    assert all(not param.requires_grad for param in primary_extra.parameters())


def test_activate_dual_mode_switches_active_sd_network_and_model_config():
    trainer = object.__new__(Wan22DualLoraTrainer)
    trainer.primary_sd = SimpleNamespace(name="i2v_sd")
    trainer.secondary_sd = SimpleNamespace(name="t2v_sd")
    trainer.primary_network = _FakeNetwork([])
    trainer.secondary_network = _FakeNetwork([])
    trainer.primary_model_config = SimpleNamespace(arch="wan22_14b_i2v")
    trainer.dual_t2v_model_config = SimpleNamespace(arch="wan22_14b")
    trainer.device_torch = torch.device("cpu")
    trainer.dual_offload_inactive_to_cpu = False
    trainer.active_dual_mode = None

    trainer._activate_dual_mode("i2v")

    assert trainer.sd is trainer.primary_sd
    assert trainer.network is trainer.primary_network
    assert trainer.model_config is trainer.primary_model_config
    assert trainer.active_dual_mode == "i2v"

    trainer._activate_dual_mode("t2v")

    assert trainer.sd is trainer.secondary_sd
    assert trainer.network is trainer.secondary_network
    assert trainer.model_config is trainer.dual_t2v_model_config
    assert trainer.active_dual_mode == "t2v"
    assert trainer.secondary_network.force_to_calls[-1][0][0] == torch.device("cpu")


def test_hook_train_loop_switches_model_before_parent_training(monkeypatch):
    trainer = object.__new__(Wan22DualLoraTrainer)
    trainer.primary_sd = SimpleNamespace(name="i2v_sd")
    trainer.secondary_sd = SimpleNamespace(name="t2v_sd")
    trainer.primary_network = _FakeNetwork([])
    trainer.secondary_network = _FakeNetwork([])
    trainer.primary_model_config = SimpleNamespace(arch="wan22_14b_i2v")
    trainer.dual_t2v_model_config = SimpleNamespace(arch="wan22_14b")
    trainer.device_torch = torch.device("cpu")
    trainer.dual_offload_inactive_to_cpu = False
    trainer.active_dual_mode = None
    trainer.dual_i2v_steps = 1
    trainer.dual_t2v_steps = 1
    trainer.step_num = 1

    parent_observed_state = {}

    def fake_parent_hook_train_loop(self, batch):
        parent_observed_state["sd"] = self.sd
        parent_observed_state["network"] = self.network
        parent_observed_state["model_config"] = self.model_config
        parent_observed_state["active_dual_mode"] = self.active_dual_mode
        parent_observed_state["batch"] = batch
        return "parent-result"

    monkeypatch.setattr(SDTrainer, "hook_train_loop", fake_parent_hook_train_loop)

    result = trainer.hook_train_loop("batch")

    assert result == "parent-result"
    assert parent_observed_state == {
        "sd": trainer.secondary_sd,
        "network": trainer.secondary_network,
        "model_config": trainer.dual_t2v_model_config,
        "active_dual_mode": "t2v",
        "batch": "batch",
    }


def test_paired_hook_trains_both_modes_with_one_optimizer_step(monkeypatch):
    trainer = _make_paired_hook_subject()
    batch = SimpleNamespace(
        tensor="original_tensor",
        latents="original_latents",
        num_frames=1,
        sigmas="original_sigmas",
        audio_pred="original_audio_pred",
        audio_target="original_audio_target",
    )
    train_calls = []

    def fake_train_single_accumulation(self, batch_item):
        train_calls.append(
            {
                "mode": self.active_dual_mode,
                "weight": self._active_paired_loss_weight,
                "tensor": batch_item.tensor,
                "latents": batch_item.latents,
                "num_frames": batch_item.num_frames,
                "boundary_index": self.current_boundary_index,
            }
        )
        batch_item.tensor = f"{self.active_dual_mode}_mutated_tensor"
        batch_item.latents = f"{self.active_dual_mode}_mutated_latents"
        batch_item.num_frames = 99
        return torch.tensor(self._active_paired_loss_weight)

    monkeypatch.setattr(
        Wan22DualLoraTrainer,
        "train_single_accumulation",
        fake_train_single_accumulation,
    )

    loss_dict = trainer.hook_train_loop(batch)

    assert train_calls == [
        {
            "mode": "i2v",
            "weight": pytest.approx(0.8),
            "tensor": "original_tensor",
            "latents": "original_latents",
            "num_frames": 1,
            "boundary_index": 1,
        },
        {
            "mode": "t2v",
            "weight": pytest.approx(0.2),
            "tensor": "original_tensor",
            "latents": "original_latents",
            "num_frames": 1,
            "boundary_index": 1,
        },
    ]
    assert batch.tensor == "original_tensor"
    assert batch.latents == "original_latents"
    assert batch.num_frames == 1
    assert trainer.steps_this_boundary == 1
    assert trainer.optimizer.step_calls == 1
    assert trainer.lr_scheduler.step_calls == 1
    assert len(trainer.accelerator.clip_grad_norm_calls) == 1
    assert loss_dict["loss"] == pytest.approx(1.0)


def test_paired_calculate_loss_applies_active_weight(monkeypatch):
    trainer = object.__new__(Wan22DualLoraTrainer)
    trainer._active_paired_loss_weight = 0.2

    monkeypatch.setattr(
        SDTrainer,
        "calculate_loss",
        lambda self, *args, **kwargs: torch.tensor(5.0),
    )

    loss = trainer.calculate_loss()

    assert loss.item() == pytest.approx(1.0)


def test_dual_config_validation_rejects_wrong_primary_arch():
    trainer = _make_validation_subject(primary_arch="wan22_14b")

    with pytest.raises(ValueError, match="primary model.arch"):
        trainer._validate_dual_training_config()


def test_dual_config_validation_rejects_wrong_secondary_arch():
    trainer = _make_validation_subject(secondary_arch="wan22_14b_i2v")

    with pytest.raises(ValueError, match="t2v_model.arch"):
        trainer._validate_dual_training_config()


def test_dual_config_validation_rejects_stage_mismatch():
    trainer = _make_validation_subject(primary_high=True, primary_low=False, secondary_high=True, secondary_low=True)

    with pytest.raises(ValueError, match="settings to match"):
        trainer._validate_dual_training_config()


def test_dual_config_validation_rejects_non_lora_network():
    trainer = _make_validation_subject(network_type="lokr")

    with pytest.raises(ValueError, match="network.type: lora"):
        trainer._validate_dual_training_config()


def test_dual_config_validation_rejects_text_encoder_training():
    trainer = _make_validation_subject(train_text_encoder=True)

    with pytest.raises(ValueError, match="train_text_encoder"):
        trainer._validate_dual_training_config()
