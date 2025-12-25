# coding=utf-8
# Copyright 2025 The HustVL Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen2.5-VL (https://github.com/QwenLM/Qwen2.5-VL). It has been modified to create DiffusionVL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VisionRotaryEmbedding
from transformers import AutoConfig, AutoImageProcessor
import os
from llava.utils import rank0_print
from safetensors.torch import load_file

DEFAULT_MIN_PIXELS = 384 * 384  # 147456
DEFAULT_MAX_PIXELS = 512 * 512  # 262144

class LlavaQwenVisionTower(nn.Module):
    def __init__(self, vision_tower_path, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_path
        self.select_layer = getattr(args, "mm_vision_select_layer", -1) # Default to final layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        
        # set the image processor for the vision tower
        processor_kwargs = {"use_fast": False}
        min_pixels = getattr(args, "min_pixels", None)
        if min_pixels is None:
            min_pixels = DEFAULT_MIN_PIXELS
            rank0_print(f"Using default min_pixels: {min_pixels}")
        else:
            rank0_print(f"Using min_pixels: {min_pixels}")
        processor_kwargs["min_pixels"] = min_pixels
        
        max_pixels = getattr(args, "max_pixels", None)
        if max_pixels is None:
            max_pixels = DEFAULT_MAX_PIXELS
            rank0_print(f"Using default max_pixels: {max_pixels}")
        else:
            rank0_print(f"Using max_pixels: {max_pixels}")
        processor_kwargs["max_pixels"] = max_pixels
        
        self.image_processor = AutoImageProcessor.from_pretrained(vision_tower_path, **processor_kwargs)
        self.vision_tower = None
        
        if hasattr(args, 'vision_config'):
            self._config = args.vision_config
        else:
            self._config = AutoConfig.from_pretrained(vision_tower_path).vision_config
        
        head_dim = self._config.hidden_size // self._config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

    def load_model(self, model_path=None, device_map=None):
        if self.is_loaded:
            return

        path_to_load = model_path if model_path is not None else self.vision_tower_name
        rank0_print(f"Loading vision tower weights from path: {path_to_load}")
        self.vision_tower = Qwen2_5_VisionTransformerPretrainedModel(self._config)

        try:
            # load the vision tower weights
            full_state_dict = {}
            vision_tower_path = os.path.join(path_to_load, 'vision_tower.safetensors')
            safetensors_index_path = os.path.join(path_to_load, 'model.safetensors.index.json')
            safetensors_path = os.path.join(path_to_load, 'model.safetensors')

            if os.path.exists(vision_tower_path):
                rank0_print(f"Found dedicated vision tower file: {vision_tower_path}")
                full_state_dict = load_file(vision_tower_path, device='cpu')
            elif os.path.exists(safetensors_path):
                rank0_print(f"Found single safetensors file: {safetensors_path}")
                full_state_dict = load_file(safetensors_path, device='cpu')
            elif os.path.exists(safetensors_index_path):
                rank0_print(f"Found sharded safetensors index: {safetensors_index_path}")
                import json
                with open(safetensors_index_path, 'r') as f:
                    index = json.load(f)
                
                shard_files = set(index['weight_map'].values())
                for shard_file in shard_files:
                    shard_path = os.path.join(path_to_load, shard_file)
                    rank0_print(f"Loading shard: {shard_path}")
                    full_state_dict.update(load_file(shard_path, device='cpu'))
            else:
                raise FileNotFoundError(f"Could not find model weights file (safetensors or bin) in {path_to_load}")

            vision_tower_weights = {}
            prefix_to_strip = None

            # filter the state dict for vision tower keys
            if any(k.startswith("model.vision_tower.vision_tower.") for k in full_state_dict.keys()):
                prefix_to_strip = "model.vision_tower.vision_tower."
            elif any(k.startswith("visual.") for k in full_state_dict.keys()):
                prefix_to_strip = "visual."
            elif any(k.startswith("vision_tower.") for k in full_state_dict.keys()):
                prefix_to_strip = "vision_tower."

            if prefix_to_strip:
                vision_tower_weights = {
                    k[len(prefix_to_strip):]: v
                    for k, v in full_state_dict.items()
                    if k.startswith(prefix_to_strip)
                }
                rank0_print(f"Found and stripped prefix '{prefix_to_strip}'. Extracted {len(vision_tower_weights)} tensors.")
            else:
                rank0_print("No standard prefix found, attempting to load the full state dict.")
                vision_tower_weights = full_state_dict

            incompatible_keys = self.vision_tower.load_state_dict(vision_tower_weights, strict=False)
            real_missing = [k for k in incompatible_keys.missing_keys if not k.startswith("merger.")]
            real_unexpected = [k for k in incompatible_keys.unexpected_keys
                              if not k.startswith("model.") and not k.startswith("lm_head.")]
            if real_missing or real_unexpected:
                 rank0_print(f"Vision tower weights loaded with incompatibilities: missing={real_missing}, unexpected={real_unexpected}")
            else:
                 rank0_print("Vision tower weights loaded successfully.")

        except Exception as e:
            rank0_print(f"ERROR: Failed to load vision tower weights from {path_to_load}: {e}. The vision tower will have random weights.")

        self.vision_tower.rotary_pos_emb = self.rotary_pos_emb
        
        if hasattr(self.vision_tower, 'merger'):
            delattr(self.vision_tower, 'merger')
            rank0_print("Removed merger module from vision tower.")
        
        self.is_loaded = True

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        """
        A forward pass that replicates Qwen2.5-VL's visual model but stops *before* the final merger/projector.
        It returns the features ready for the projector.
        """
        # This is a partial copy of Qwen2_5_VisionTransformerPretrainedModel.forward
        hidden_states = self.vision_tower.patch_embed(hidden_states)
        rotary_pos_emb = self.vision_tower.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.vision_tower.get_window_index(grid_thw)
        
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens, device=hidden_states.device, dtype=torch.int32
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.vision_tower.spatial_merge_unit, self.vision_tower.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.vision_tower.spatial_merge_unit, self.vision_tower.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.vision_tower.blocks):
            cu_seqlens_now = cu_seqlens if layer_num in self.vision_tower.fullatt_block_indexes else cu_window_seqlens
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        
        # Return features AND the window_index needed for reversing the shuffle
        return hidden_states, window_index

    @property
    def dtype(self):
        return self.vision_tower.dtype
    
    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        return self._config

    @property
    def hidden_size(self):
        return self._config.hidden_size