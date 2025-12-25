#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import rank0_print


def load_pretrained_model(
    model_path,
    model_base=None,
    model_name=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    torch_dtype="float16",
    attn_implementation="flash_attention_2",
    customized_config=None,
    overwrite_config=None,
    force_model_type=None,
    **kwargs
):
    """
    Load a pretrained model for inference.

    Supported models:
    - DiffusionVL-QwenVL (diffusionvl_qwenvl): Qwen2.5-VL + BD3-LM
    - DiffusionVL-Qwen (diffusionvl_qwen): Qwen2.5 + BD3-LM
    - LLaVA-Qwen (llava_qwen): Standard autoregressive Qwen
    - LLaVA-LLaDA (llava_llada): LLaDA diffusion model
    - LLaVA-LLaDA-BD3LM (llava_llada_bd3lm): LLaDA + BD3-LM
    """
    kwargs["device_map"] = device_map

    if model_name is None:
        model_name = os.path.basename(model_path)

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16

    if customized_config is not None:
        kwargs["config"] = customized_config

    # Determine model type
    model_type = _detect_model_type(model_path, model_name, force_model_type)
    rank0_print(f"Detected model type: {model_type}")

    # Load model based on type
    if model_type == "diffusionvl_qwenvl":
        tokenizer, model = _load_diffusionvl_qwenvl(model_path, attn_implementation, customized_config, overwrite_config, kwargs)
    elif model_type == "diffusionvl_qwen":
        tokenizer, model = _load_diffusionvl_qwen(model_path, attn_implementation, customized_config, overwrite_config, kwargs)
    elif model_type == "llava_qwen":
        tokenizer, model = _load_llava_qwen(model_path, attn_implementation, customized_config, overwrite_config, kwargs)
    elif model_type == "llava_llada_bd3lm":
        tokenizer, model = _load_llava_llada_bd3lm(model_path, attn_implementation, customized_config, overwrite_config, kwargs)
    elif model_type == "llava_llada":
        tokenizer, model = _load_llava_llada(model_path, attn_implementation, customized_config, overwrite_config, kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: diffusionvl_qwenvl, diffusionvl_qwen, llava_qwen, llava_llada_bd3lm, llava_llada")

    rank0_print(f"Model Class: {model.__class__.__name__}")

    # Setup image processor
    image_processor = None
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    if mm_use_im_patch_token or mm_use_im_start_end:
        model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(model_path=model_path, device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device="cuda", dtype=torch.float16)
    image_processor = vision_tower.image_processor

    # Determine context length
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def _detect_model_type(model_path, model_name, force_model_type):
    """Detect model type from config or model name."""
    if force_model_type is not None:
        return force_model_type

    # Try to read from config.json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        import json
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            model_type = config_dict.get("model_type", "")

            if model_type in ["diffusionvl_qwenvl", "llada_qwen"]:
                return "diffusionvl_qwenvl"
            elif model_type == "diffusionvl_qwen":
                return "diffusionvl_qwen"
            elif model_type == "llava_llada_bd3lm":
                return "llava_llada_bd3lm"
            elif model_type == "llava_llada":
                return "llava_llada"
            elif model_type == "llava_qwen":
                return "llava_qwen"
        except:
            pass

    # Fallback to model name heuristics
    model_name_lower = model_name.lower()
    if "diffusionvl" in model_name_lower and "qwenvl" in model_name_lower:
        return "diffusionvl_qwenvl"
    elif "diffusionvl" in model_name_lower:
        return "diffusionvl_qwen"
    elif "llada" in model_name_lower and "bd3lm" in model_name_lower:
        return "llava_llada_bd3lm"
    elif "llada" in model_name_lower:
        return "llava_llada"
    elif "qwen" in model_name_lower:
        return "llava_qwen"

    raise ValueError(f"Cannot detect model type for {model_name}. Please specify force_model_type.")


def _load_diffusionvl_qwenvl(model_path, attn_implementation, customized_config, overwrite_config, kwargs):
    """Load DiffusionVL-QwenVL model (Qwen2.5-VL + BD3-LM)."""
    from llava.model.language_model.llava_diffusionvl_qwenvl import DiffusionVLQwenVLConfig, DiffusionVLQwenVLForCausalLM

    rank0_print(f"Loading DiffusionVL-QwenVL (Qwen2.5-VL + BD3-LM) from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if customized_config is None:
        llava_cfg = DiffusionVLQwenVLConfig.from_pretrained(model_path)
    else:
        llava_cfg = customized_config

    if overwrite_config is not None:
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(llava_cfg, k, v)

    model = DiffusionVLQwenVLForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        config=llava_cfg,
        **kwargs
    )
    return tokenizer, model


def _load_diffusionvl_qwen(model_path, attn_implementation, customized_config, overwrite_config, kwargs):
    """Load DiffusionVL-Qwen model (Qwen2.5 + BD3-LM)."""
    from llava.model.language_model.llava_diffusionvl_qwen import DiffusionVLQwenConfig, DiffusionVLQwenForCausalLM

    rank0_print(f"Loading DiffusionVL-Qwen (Qwen2.5 + BD3-LM) from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if customized_config is None:
        llava_cfg = DiffusionVLQwenConfig.from_pretrained(model_path)
    else:
        llava_cfg = customized_config

    if overwrite_config is not None:
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(llava_cfg, k, v)

    model = DiffusionVLQwenForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        config=llava_cfg,
        **kwargs
    )
    return tokenizer, model


def _load_llava_qwen(model_path, attn_implementation, customized_config, overwrite_config, kwargs):
    """Load standard LLaVA-Qwen model (Autoregressive)."""
    from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM

    rank0_print(f"Loading LLaVA-Qwen (Autoregressive) from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if customized_config is None:
        llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
        # Remove text_config if it's a dict to avoid GenerationConfig error
        if hasattr(llava_cfg, 'text_config') and isinstance(llava_cfg.text_config, dict):
            delattr(llava_cfg, 'text_config')
    else:
        llava_cfg = customized_config

    if overwrite_config is not None:
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(llava_cfg, k, v)

    model = LlavaQwenForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        config=llava_cfg,
        **kwargs
    )
    return tokenizer, model


def _load_llava_llada_bd3lm(model_path, attn_implementation, customized_config, overwrite_config, kwargs):
    """Load LLaVA-LLaDA-BD3LM model (LLaDA + BD3-LM)."""
    from llava.model.language_model.llava_llada_bd3lm import LlavaLladaBD3LMConfig, LlavaLladaBD3LMForCausalLM

    rank0_print(f"Loading LLaVA-LLaDA-BD3LM (LLaDA + BD3-LM) from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if customized_config is None:
        llada_cfg = LlavaLladaBD3LMConfig.from_pretrained(model_path)
    else:
        llada_cfg = customized_config

    if overwrite_config is not None:
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(llada_cfg, k, v)

    model = LlavaLladaBD3LMForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        config=llada_cfg,
        **kwargs
    )
    return tokenizer, model


def _load_llava_llada(model_path, attn_implementation, customized_config, overwrite_config, kwargs):
    """Load LLaVA-LLaDA model (full diffusion)."""
    from llava.model.language_model.llava_llada import LlavaLLaDAConfig, LlavaLLaDAModelLM

    rank0_print(f"Loading LLaVA-LLaDA (full diffusion) from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if customized_config is None:
        llada_cfg = LlavaLLaDAConfig.from_pretrained(model_path)
    else:
        llada_cfg = customized_config

    if overwrite_config is not None:
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(llada_cfg, k, v)

    model = LlavaLLaDAModelLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        config=llada_cfg,
        **kwargs
    )
    return tokenizer, model
