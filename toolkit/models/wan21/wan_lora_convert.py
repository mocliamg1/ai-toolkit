def _split_kohya_suffix(key):
    for suffix in (
        ".alpha",
        ".lora_down.weight",
        ".lora_up.weight",
        ".lora_A.weight",
        ".lora_B.weight",
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix
    return key, ""


def _convert_kohya_wan_key_to_original(key):
    if not key.startswith("lora_unet_"):
        return key

    module_key, suffix = _split_kohya_suffix(key)
    module_key = module_key[len("lora_unet_") :]

    stage_name = None
    for candidate in ("transformer_1_", "transformer_2_"):
        if module_key.startswith(candidate):
            stage_name = candidate[:-1]
            module_key = module_key[len(candidate) :]
            break

    if not module_key.startswith("blocks_"):
        return key

    parts = module_key.split("_")
    if len(parts) < 4:
        return key

    _, block_idx, *remainder = parts
    remainder_key = "_".join(remainder)

    if remainder_key.startswith("ffn_"):
        if remainder_key == "ffn_0":
            target = f"blocks.{block_idx}.ffn.0"
        elif remainder_key == "ffn_2":
            target = f"blocks.{block_idx}.ffn.2"
        else:
            return key
    else:
        attn_prefix_map = {
            "self_attn_": "self_attn",
            "cross_attn_": "cross_attn",
            "attn1_": "self_attn",
            "attn2_": "cross_attn",
        }
        attention_name = None
        projection_key = None
        for prefix, mapped_name in attn_prefix_map.items():
            if remainder_key.startswith(prefix):
                attention_name = mapped_name
                projection_key = remainder_key[len(prefix) :]
                break

        if attention_name is None or projection_key is None:
            return key

        projection_map = {
            "q": "q",
            "k": "k",
            "v": "v",
            "o": "o",
            "to_q": "q",
            "to_k": "k",
            "to_v": "v",
            "to_out": "o",
            "to_out_0": "o",
            "k_img": "k_img",
            "v_img": "v_img",
            "add_k_proj": "k_img",
            "add_v_proj": "v_img",
        }
        projection_name = projection_map.get(projection_key)
        if projection_name is None:
            return key

        target = f"blocks.{block_idx}.{attention_name}.{projection_name}"

    prefix = "diffusion_model."
    if stage_name is not None:
        prefix += f"{stage_name}."
    return f"{prefix}{target}{suffix}"


def convert_to_diffusers(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_key = _convert_kohya_wan_key_to_original(key)
        # Base model name change
        if new_key.startswith("diffusion_model."):
            new_key = new_key.replace("diffusion_model.", "transformer.", 1)

        # Attention blocks conversion
        if "self_attn" in new_key:
            new_key = new_key.replace("self_attn", "attn1")
        elif "cross_attn" in new_key:
            new_key = new_key.replace("cross_attn", "attn2")

        # Attention components conversion
        parts = new_key.split(".")
        for i, part in enumerate(parts):
            if part in ["q", "k", "v"]:
                parts[i] = f"to_{part}"
            elif part == "k_img":
                parts[i] = "add_k_proj"
            elif part == "v_img":
                parts[i] = "add_v_proj"
            elif part == "o":
                parts[i] = "to_out.0"
        new_key = ".".join(parts)

        # FFN conversion
        if "ffn.0" in new_key:
            new_key = new_key.replace("ffn.0", "ffn.net.0.proj")
        elif "ffn.2" in new_key:
            new_key = new_key.replace("ffn.2", "ffn.net.2")

        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def convert_to_original(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        # Base model name change
        if key.startswith("transformer."):
            new_key = key.replace("transformer.", "diffusion_model.")

        # Attention blocks conversion
        if "attn1" in new_key:
            new_key = new_key.replace("attn1", "self_attn")
        elif "attn2" in new_key:
            new_key = new_key.replace("attn2", "cross_attn")

        # Attention components conversion
        if "to_out.0" in new_key:
            new_key = new_key.replace("to_out.0", "o")
        elif "to_q" in new_key:
            new_key = new_key.replace("to_q", "q")
        elif "to_k" in new_key:
            new_key = new_key.replace("to_k", "k")
        elif "to_v" in new_key:
            new_key = new_key.replace("to_v", "v")
        
        # img attn projection
        elif "add_k_proj" in new_key:
            new_key = new_key.replace("add_k_proj", "k_img")
        elif "add_v_proj" in new_key:
            new_key = new_key.replace("add_v_proj", "v_img")

        # FFN conversion
        if "ffn.net.0.proj" in new_key:
            new_key = new_key.replace("ffn.net.0.proj", "ffn.0")
        elif "ffn.net.2" in new_key:
            new_key = new_key.replace("ffn.net.2", "ffn.2")

        new_state_dict[new_key] = state_dict[key]
    return new_state_dict
