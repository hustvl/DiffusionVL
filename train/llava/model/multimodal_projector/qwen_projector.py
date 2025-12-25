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
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPatchMerger

class LlavaQwenProjector(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.merger = Qwen2_5_VLPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=vision_config.hidden_size,
            spatial_merge_size=vision_config.spatial_merge_size,
        )

    def forward(self, features_tuple):
        """
        The forward pass for the projector, which takes the output of the vision tower.
        """
        hidden_states, window_index = features_tuple
        
        # 1. Project the features using the merger
        projected_features = self.merger(hidden_states)
        
        # 2. Reverse the windowing shuffle to restore original order
        reverse_indices = torch.argsort(window_index)
        final_features = projected_features[reverse_indices, :]
        
        return final_features


