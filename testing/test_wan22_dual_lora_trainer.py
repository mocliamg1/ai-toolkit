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

    def get_all_modules(self):
        return self._modules

    def parameters(self):
        for module in self._modules:
            yield from module.parameters()


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
