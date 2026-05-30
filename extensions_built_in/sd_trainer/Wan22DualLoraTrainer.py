import copy
from collections import OrderedDict
from typing import Literal, Optional, Tuple

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.basic import flush
from toolkit.config_modules import ModelConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.print import print_acc
from toolkit.train_tools import get_torch_dtype
from toolkit.util.get_model import get_model_class


DualMode = Literal["i2v", "t2v"]


class Wan22DualLoraTrainer(SDTrainer):
    @staticmethod
    def normalize_process_config(config: OrderedDict) -> OrderedDict:
        normalized = copy.deepcopy(config)
        model_config = normalized.get("model", {})
        if model_config.get("arch") == "wan22_14b_i2v_t2v":
            model_config["arch"] = "wan22_14b_i2v"
        train_config = normalized.setdefault("train", {})
        train_config.setdefault("train_refiner", False)
        return normalized

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        config = self.normalize_process_config(config)
        super().__init__(process_id, job, config, **kwargs)

        self.primary_sd = None
        self.primary_network = None
        self.primary_model_config = self.model_config

        dual_config = self.get_conf("dual_model", required=True)
        t2v_model_config = copy.deepcopy(dual_config.get("t2v_model", None))
        if t2v_model_config is None:
            raise ValueError("Wan2.2 dual LoRA training requires dual_model.t2v_model")
        t2v_model_config["dtype"] = self.train_config.dtype

        self.dual_t2v_model_config = ModelConfig(**t2v_model_config)
        self.dual_i2v_steps = max(1, int(dual_config.get("i2v_steps", 8)))
        self.dual_t2v_steps = max(1, int(dual_config.get("t2v_steps", 2)))
        self.dual_offload_inactive_to_cpu = bool(
            dual_config.get("offload_inactive_to_cpu", True)
        )

        self.secondary_sd = None
        self.secondary_network = None
        self.active_dual_mode: Optional[DualMode] = None
        self._shared_lora_module_count = 0
        self._shared_lora_tensor_count = 0
        self._shared_primary_parameter_ids = set()

        self._validate_dual_training_config()

    def _validate_dual_training_config(self):
        if self.model_config.arch != "wan22_14b_i2v":
            raise ValueError(
                "Wan2.2 dual LoRA training requires primary model.arch to be wan22_14b_i2v"
            )
        if self.dual_t2v_model_config.arch != "wan22_14b":
            raise ValueError(
                "Wan2.2 dual LoRA training requires dual_model.t2v_model.arch to be wan22_14b"
            )
        if self.network_config is None or self.network_config.type.lower() != "lora":
            raise ValueError("Wan2.2 dual LoRA training supports network.type: lora only")
        if self.train_config.train_text_encoder:
            raise ValueError("Wan2.2 dual LoRA training requires train_text_encoder: false")
        if self.adapter_config is not None:
            raise ValueError("Wan2.2 dual LoRA training does not support adapter training in v1")
        if self.embed_config is not None:
            raise ValueError("Wan2.2 dual LoRA training does not support embedding training in v1")
        if self.decorator_config is not None:
            raise ValueError("Wan2.2 dual LoRA training does not support decorator training in v1")
        if self.train_config.train_refiner or self.model_config.refiner_name_or_path is not None:
            raise ValueError("Wan2.2 dual LoRA training does not support refiner training in v1")
        if self.get_conf("slider", None) is not None:
            raise ValueError("Wan2.2 dual LoRA training does not support slider training in v1")

        primary_stage_flags = self._get_stage_flags(self.model_config)
        secondary_stage_flags = self._get_stage_flags(self.dual_t2v_model_config)
        if primary_stage_flags != secondary_stage_flags:
            raise ValueError(
                "Wan2.2 dual LoRA training requires primary and T2V train_high_noise/train_low_noise "
                "settings to match in v1"
            )

    @staticmethod
    def _get_stage_flags(model_config: ModelConfig) -> Tuple[bool, bool]:
        model_kwargs = getattr(model_config, "model_kwargs", {}) or {}
        return (
            bool(model_kwargs.get("train_high_noise", True)),
            bool(model_kwargs.get("train_low_noise", True)),
        )

    def _get_secondary_train_scheduler(self):
        model_class = get_model_class(self.dual_t2v_model_config)
        if not hasattr(model_class, "get_train_scheduler"):
            raise ValueError("Wan2.2 dual LoRA secondary model must provide get_train_scheduler")
        return model_class.get_train_scheduler()

    def hook_after_model_load(self):
        self.primary_sd = self.sd

        secondary_model_class = get_model_class(self.dual_t2v_model_config)
        self.secondary_sd = secondary_model_class(
            device=self.accelerator.device,
            model_config=copy.deepcopy(self.dual_t2v_model_config),
            dtype=self.train_config.dtype,
            custom_pipeline=self.custom_pipeline,
            noise_scheduler=self._get_secondary_train_scheduler(),
        )
        self.secondary_sd.load_model()
        self._configure_secondary_model_after_load()
        if self.dual_offload_inactive_to_cpu:
            self._offload_sd_transformers(self.secondary_sd)
        flush()

    def _configure_secondary_model_after_load(self):
        dtype = get_torch_dtype(self.train_config.dtype)
        secondary_unet = self.secondary_sd.unet
        secondary_text_encoder = self.secondary_sd.text_encoder
        secondary_vae = self.secondary_sd.vae

        if self.train_config.xformers:
            if hasattr(secondary_vae, "enable_xformers_memory_efficient_attention"):
                secondary_vae.enable_xformers_memory_efficient_attention()
            if hasattr(secondary_unet, "enable_xformers_memory_efficient_attention"):
                secondary_unet.enable_xformers_memory_efficient_attention()
            if isinstance(secondary_text_encoder, list):
                for text_encoder in secondary_text_encoder:
                    if hasattr(text_encoder, "enable_xformers_memory_efficient_attention"):
                        text_encoder.enable_xformers_memory_efficient_attention()
            elif hasattr(secondary_text_encoder, "enable_xformers_memory_efficient_attention"):
                secondary_text_encoder.enable_xformers_memory_efficient_attention()

        if self.train_config.attention_backend != "native":
            if hasattr(secondary_vae, "set_attention_backend"):
                secondary_vae.set_attention_backend(self.train_config.attention_backend)
            if hasattr(secondary_unet, "set_attention_backend"):
                secondary_unet.set_attention_backend(self.train_config.attention_backend)
            if isinstance(secondary_text_encoder, list):
                for text_encoder in secondary_text_encoder:
                    if hasattr(text_encoder, "set_attention_backend"):
                        text_encoder.set_attention_backend(self.train_config.attention_backend)
            elif hasattr(secondary_text_encoder, "set_attention_backend"):
                secondary_text_encoder.set_attention_backend(self.train_config.attention_backend)

        if self.train_config.gradient_checkpointing:
            if hasattr(secondary_unet, "enable_gradient_checkpointing"):
                secondary_unet.enable_gradient_checkpointing()
            elif hasattr(secondary_unet, "gradient_checkpointing"):
                secondary_unet.gradient_checkpointing = True

        if isinstance(secondary_text_encoder, list):
            for text_encoder in secondary_text_encoder:
                text_encoder.requires_grad_(False)
                text_encoder.eval()
        elif secondary_text_encoder is not None:
            secondary_text_encoder.requires_grad_(False)
            secondary_text_encoder.eval()

        secondary_unet.requires_grad_(False)
        secondary_unet.eval()
        secondary_unet.to(self.device_torch, dtype=dtype)

        secondary_vae = secondary_vae.to(torch.device("cpu"), dtype=dtype)
        secondary_vae.requires_grad_(False)
        secondary_vae.eval()
        self.secondary_sd.vae = secondary_vae

    def hook_add_extra_train_params(self, params):
        if self.network is None:
            raise ValueError("Wan2.2 dual LoRA training requires a primary LoRA network")
        self.primary_network = self.network

        self.secondary_network = self._create_secondary_lora_network()
        self.secondary_sd.network = self.secondary_network

        (
            self._shared_lora_module_count,
            self._shared_lora_tensor_count,
        ) = self._tie_secondary_lora_parameters(self.primary_network, self.secondary_network)
        self._shared_primary_parameter_ids = self._get_shared_parameter_ids(
            self.primary_network,
            self.secondary_network,
        )

        if self._shared_lora_module_count == 0:
            raise ValueError(
                "Wan2.2 dual LoRA training could not find any compatible shared LoRA modules "
                "between I2V and T2V"
            )

        self._freeze_unshared_primary_parameters(
            self.primary_network,
            self._shared_primary_parameter_ids,
        )
        params = self._dedupe_optimizer_params(
            params,
            keep_param_ids=self._shared_primary_parameter_ids,
        )
        if len(params) == 0:
            raise ValueError(
                "Wan2.2 dual LoRA training could not find shared LoRA parameters "
                "in the optimizer parameter groups"
            )
        print_acc(
            f"Wan2.2 dual LoRA shared {self._shared_lora_tensor_count} tensors across "
            f"{self._shared_lora_module_count} modules"
        )
        self._activate_dual_mode("i2v")
        return params

    def _create_secondary_lora_network(self) -> LoRASpecialNetwork:
        network_kwargs = copy.deepcopy(self.network_config.network_kwargs)
        if hasattr(self.secondary_sd, "target_lora_modules"):
            network_kwargs["target_lin_modules"] = self.secondary_sd.target_lora_modules

        secondary_network = LoRASpecialNetwork(
            text_encoder=self.secondary_sd.text_encoder,
            unet=self.secondary_sd.get_model_to_train(),
            lora_dim=self.network_config.linear,
            multiplier=1.0,
            alpha=self.network_config.linear_alpha,
            train_unet=self.train_config.train_unet,
            train_text_encoder=False,
            conv_lora_dim=self.network_config.conv,
            conv_alpha=self.network_config.conv_alpha,
            is_sdxl=False,
            is_v2=False,
            is_v3=False,
            is_pixart=False,
            is_auraflow=False,
            is_flux=False,
            is_lumina2=False,
            dropout=self.network_config.dropout,
            use_text_encoder_1=self.dual_t2v_model_config.use_text_encoder_1,
            use_text_encoder_2=self.dual_t2v_model_config.use_text_encoder_2,
            network_config=self.network_config,
            network_type=self.network_config.type,
            transformer_only=self.network_config.transformer_only,
            is_transformer=self.secondary_sd.is_transformer,
            base_model=self.secondary_sd,
            **network_kwargs,
        )
        secondary_network.force_to(self.device_torch, dtype=torch.float32)
        secondary_network._update_torch_multiplier()
        secondary_network.apply_to(
            self.secondary_sd.text_encoder,
            self.secondary_sd.unet,
            False,
            self.train_config.train_unet,
        )
        secondary_network.can_merge_in = False
        secondary_network.prepare_grad_etc(
            self.secondary_sd.text_encoder,
            self.secondary_sd.unet,
        )
        if self.train_config.gradient_checkpointing:
            secondary_network.enable_gradient_checkpointing()
        return secondary_network

    @classmethod
    def _tie_secondary_lora_parameters(
        cls,
        primary_network: LoRASpecialNetwork,
        secondary_network: LoRASpecialNetwork,
    ) -> Tuple[int, int]:
        primary_modules = {
            module.lora_name: module for module in primary_network.get_all_modules()
        }
        shared_module_count = 0
        shared_tensor_count = 0

        for secondary_module in secondary_network.get_all_modules():
            primary_module = primary_modules.get(secondary_module.lora_name)
            if primary_module is None:
                cls._freeze_module_params(secondary_module)
                continue

            primary_params = dict(primary_module.named_parameters())
            secondary_params = dict(secondary_module.named_parameters())
            if primary_params.keys() != secondary_params.keys():
                cls._freeze_module_params(secondary_module)
                continue

            if any(
                primary_params[name].shape != secondary_params[name].shape
                for name in primary_params.keys()
            ):
                cls._freeze_module_params(secondary_module)
                continue

            for name, primary_param in primary_params.items():
                cls._set_nested_parameter(secondary_module, name, primary_param)
                shared_tensor_count += 1
            shared_module_count += 1

        return shared_module_count, shared_tensor_count

    @staticmethod
    def _freeze_module_params(module: torch.nn.Module):
        for param in module.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _get_shared_parameter_ids(
        primary_network: LoRASpecialNetwork,
        secondary_network: LoRASpecialNetwork,
    ) -> set:
        primary_parameter_ids = {id(param) for param in primary_network.parameters()}
        return {
            id(param)
            for param in secondary_network.parameters()
            if id(param) in primary_parameter_ids
        }

    @staticmethod
    def _freeze_unshared_primary_parameters(
        primary_network: LoRASpecialNetwork,
        shared_parameter_ids: set,
    ):
        for param in primary_network.parameters():
            if id(param) not in shared_parameter_ids:
                param.requires_grad_(False)

    @staticmethod
    def _set_nested_parameter(
        module: torch.nn.Module,
        parameter_name: str,
        parameter: torch.nn.Parameter,
    ):
        parent = module
        parts = parameter_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], parameter)

    @staticmethod
    def _dedupe_optimizer_params(params, keep_param_ids=None):
        seen = set()
        deduped = []
        for group in params:
            if isinstance(group, dict):
                new_group = dict(group)
                unique_params = []
                for param in list(group.get("params", [])):
                    param_id = id(param)
                    if keep_param_ids is not None and param_id not in keep_param_ids:
                        continue
                    if param_id in seen:
                        continue
                    seen.add(param_id)
                    unique_params.append(param)
                if len(unique_params) == 0:
                    continue
                new_group["params"] = unique_params
                deduped.append(new_group)
            else:
                param_id = id(group)
                if keep_param_ids is not None and param_id not in keep_param_ids:
                    continue
                if param_id in seen:
                    continue
                seen.add(param_id)
                deduped.append(group)
        return deduped

    def _get_dual_mode_for_step(self, step: Optional[int] = None) -> DualMode:
        step = self.step_num if step is None else step
        cycle = self.dual_i2v_steps + self.dual_t2v_steps
        cycle_index = step % cycle
        return "i2v" if cycle_index < self.dual_i2v_steps else "t2v"

    def _activate_dual_mode(self, mode: DualMode):
        if mode == self.active_dual_mode:
            return

        if mode == "i2v":
            active_sd = self.primary_sd
            active_network = self.primary_network
            active_model_config = self.primary_model_config
            inactive_sd = self.secondary_sd
        else:
            active_sd = self.secondary_sd
            active_network = self.secondary_network
            active_model_config = self.dual_t2v_model_config
            inactive_sd = self.primary_sd

        if active_sd is None or active_network is None:
            raise ValueError("Wan2.2 dual LoRA trainer is not fully initialized")

        if self.dual_offload_inactive_to_cpu and inactive_sd is not None:
            self._offload_sd_transformers(inactive_sd)

        self._move_lora_network_to_train_device(active_network)
        self.sd = active_sd
        self.network = active_network
        self.model_config = active_model_config
        self.active_dual_mode = mode

    def _move_lora_network_to_train_device(self, network: LoRASpecialNetwork):
        if hasattr(network, "force_to"):
            network.force_to(self.device_torch, dtype=torch.float32)
        else:
            network.to(self.device_torch, dtype=torch.float32)

    @staticmethod
    def _offload_sd_transformers(sd):
        model = getattr(sd, "model", None)
        if model is None:
            return
        for attr in ("transformer_1", "transformer_2"):
            transformer = getattr(model, attr, None)
            if transformer is not None:
                transformer.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def hook_train_loop(self, batch):
        self._activate_dual_mode(self._get_dual_mode_for_step())
        return super().hook_train_loop(batch)

    def sample(self, *args, **kwargs):
        if self.primary_sd is not None and self.primary_network is not None:
            self._activate_dual_mode("i2v")
        return super().sample(*args, **kwargs)

    def save(self, *args, **kwargs):
        if self.primary_sd is not None and self.primary_network is not None:
            self._activate_dual_mode("i2v")
        return super().save(*args, **kwargs)
