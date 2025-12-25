# coding=utf-8
# Copyright 2025 The HustVL Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on LLaDA-V (https://github.com/ML-GSAI/LLaDA-V) and
# Block Diffusion (https://github.com/kuleshov-group/bd3lm). It has been modified to create DiffusionVL.
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

"""LLaVA-LLaDA-BD3LM model implementation."""

from typing import List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.cache_utils import Cache, DynamicCache

from .configuration_llada import LLaDAConfig
from .modeling_llada import (
    LLaDAPreTrainedModel,
    LLaDAModel,
    LLaDARMSNorm,
    LLaDARotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    LLaDAAttention,
    LLaDAFlashAttention2,
    LLaDASdpaAttention,
    LLaDADecoderLayer,
)
from transformers.modeling_layers import GradientCheckpointingLayer
import math
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.constants import IGNORE_INDEX


class LlavaLladaBD3LMConfig(LLaDAConfig):
    model_type = "llava_llada_bd3lm"

class LlavaLladaBD3LMDecoderLayer(GradientCheckpointingLayer, LLaDADecoderLayer):
    def __init__(self, config: LLaDAConfig, layer_idx: int):
        LLaDADecoderLayer.__init__(self, config, layer_idx)


class LlavaLladaBD3LMAttention(LLaDAAttention):
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        store_kv = kwargs.pop("store_kv", True)
        
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            if store_kv:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                if isinstance(past_key_value, Cache):
                    if self.layer_idx < len(past_key_value):
                        past_key_states, past_value_states = past_key_value[self.layer_idx]
                        key_states = torch.cat([past_key_states, key_states], dim=2)
                        value_states = torch.cat([past_value_states, value_states], dim=2)
                else:
                    # Legacy cache format
                    if self.layer_idx < len(past_key_value):
                        past_key_states, past_value_states = past_key_value[self.layer_idx]
                        key_states = torch.cat([past_key_states, key_states], dim=2)
                        value_states = torch.cat([past_value_states, value_states], dim=2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlavaLladaBD3LMFlashAttention2(LLaDAFlashAttention2):
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        store_kv = kwargs.pop("store_kv", True)
        
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            if store_kv:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                if isinstance(past_key_value, Cache):
                    if self.layer_idx < len(past_key_value):
                        past_key_states, past_value_states = past_key_value[self.layer_idx]
                        key_states = torch.cat([past_key_states, key_states], dim=2)
                        value_states = torch.cat([past_value_states, value_states], dim=2)
                else:
                    if self.layer_idx < len(past_key_value):
                        past_key_states, past_value_states = past_key_value[self.layer_idx]
                        key_states = torch.cat([past_key_states, key_states], dim=2)
                        value_states = torch.cat([past_value_states, value_states], dim=2)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlavaLladaBD3LMSdpaAttention(LLaDASdpaAttention):
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        store_kv = kwargs.pop("store_kv", True)
        
        if output_attentions:
            # Fallback to manual attention
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            if store_kv:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                if isinstance(past_key_value, Cache):
                    if self.layer_idx < len(past_key_value):
                        past_key_states, past_value_states = past_key_value[self.layer_idx]
                        key_states = torch.cat([past_key_states, key_states], dim=2)
                        value_states = torch.cat([past_value_states, value_states], dim=2)
                else:
                    if self.layer_idx < len(past_key_value):
                        past_key_states, past_value_states = past_key_value[self.layer_idx]
                        key_states = torch.cat([past_key_states, key_states], dim=2)
                        value_states = torch.cat([past_value_states, value_states], dim=2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            is_causal=False,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class LlavaLladaBD3LMModel(LlavaMetaModel, LLaDAModel):
    config_class = LlavaLladaBD3LMConfig
    
    def __init__(self, config: LLaDAConfig):
        super(LlavaLladaBD3LMModel, self).__init__(config)
        
        if getattr(config, 'enable_bd3lm', False):
            attn_implementation = config._attn_implementation
            BD3LM_ATTENTION_CLASSES = {
                "eager": LlavaLladaBD3LMAttention,
                "flash_attention_2": LlavaLladaBD3LMFlashAttention2,
                "sdpa": LlavaLladaBD3LMSdpaAttention,
            }
            bd3lm_attn_class = BD3LM_ATTENTION_CLASSES.get(attn_implementation, LlavaLladaBD3LMAttention)
            
            new_layers = nn.ModuleList()
            for layer_idx, layer in enumerate(self.layers):
                new_layer = LlavaLladaBD3LMDecoderLayer(config, layer_idx)
                new_layer.load_state_dict(layer.state_dict(), strict=False)
                new_layer.self_attn = bd3lm_attn_class(
                    config=config, layer_idx=layer_idx
                )
                new_layers.append(new_layer)
            self.layers = new_layers
            
            self._init_bd3lm_components(config)
    
    def _init_bd3lm_components(self, config):
        from .bd3lm_utils import LogLinearNoise
        
        self.noise_scheduler = LogLinearNoise()
        self.mask_token_id = 126336
        self.bd3lm_block_size = config.bd3lm_block_size
        self.antithetic_sampling = getattr(config, 'bd3lm_antithetic_sampling', True)
        self.sampling_eps_min = getattr(config, 'bd3lm_sampling_eps_min', 1e-3)
        self.sampling_eps_max = getattr(config, 'bd3lm_sampling_eps_max', 1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        store_kv: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                import logging
                logger = logging.get_logger(__name__)
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            from transformers.cache_utils import StaticCache
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            from transformers.cache_utils import StaticCache
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


from .modeling_llada import LLaDAModelLM

class LlavaLladaBD3LMForCausalLM(LLaDAModelLM, LlavaMetaForCausalLM):
    config_class = LlavaLladaBD3LMConfig
    
    def __init__(self, config):
        LLaDAModelLM.__init__(self, config)
        config.model_type = "llava_llada_bd3lm"
        
        self.model = LlavaLladaBD3LMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        image_grid_thws: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                conversation_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities=modalities,
                image_sizes=image_sizes,
                image_grid_thws=image_grid_thws,
                is_llada=True
            )
        return self._forward_bd3lm(
            inputs_embeds, labels, attention_mask,
            position_ids, past_key_values, use_cache,
            output_attentions, output_hidden_states, return_dict
        )
    
    def _forward_bd3lm(
        self, inputs_embeds, labels, attention_mask=None,
        position_ids=None, past_key_values=None, use_cache=None,
        output_attentions=None, output_hidden_states=None, return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        bd3lm_inputs, move_indices, loss_scale, x0_embeds, avg_noise_level = \
            self._apply_bd3lm_noise_embedding(inputs_embeds, labels)

        from .bd3lm_utils import block_diff_mask

        seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device

        q_idx = torch.arange(seq_len * 2, device=device)[:, None]
        kv_idx = torch.arange(seq_len * 2, device=device)[None, :]

        mask = block_diff_mask(
            b=None, h=None,
            q_idx=q_idx,
            kv_idx=kv_idx,
            block_size=self.model.bd3lm_block_size,
            n=seq_len
        )
        if attention_mask is not None and attention_mask.dim() == 2:
            batch_size = inputs_embeds.shape[0]
            extended_attention_mask = torch.cat([attention_mask, attention_mask], dim=1)  # (B, 2L)
            query_validity_mask = extended_attention_mask.unsqueeze(-1)  # (B, 2L, 1)
            key_validity_mask = extended_attention_mask.unsqueeze(-2)  # (B, 1, 2L)
            combined_padding_mask_2d = query_validity_mask & key_validity_mask
            mask = mask & combined_padding_mask_2d

        attention_mask_4d = torch.zeros(mask.shape, dtype=inputs_embeds.dtype, device=device)
        attention_mask_4d.masked_fill_(~mask, torch.finfo(inputs_embeds.dtype).min)
        attention_mask_4d = attention_mask_4d.unsqueeze(1)

        if position_ids is None:
            batch_size, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
            device = inputs_embeds.device
            pos_ids_part = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            position_ids = torch.cat([pos_ids_part, pos_ids_part], dim=1)  # [batch_size, 2*seq_len]
        
        outputs = self.model(
            input_ids=None,
            inputs_embeds=bd3lm_inputs,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        hidden_states = hidden_states[:, :inputs_embeds.shape[1]]
        
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            loss = self._compute_bd3lm_loss_embedding(logits, labels, move_indices, loss_scale)
        
        if self.training:
            if not hasattr(self, '_current_custom_metrics'):
                self._current_custom_metrics = {}
            self._current_custom_metrics["anneal/noise_level"] = avg_noise_level
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _apply_bd3lm_noise_embedding(self, inputs_embeds, labels):
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        
        block_size = self.model.bd3lm_block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        _eps_b = torch.rand((batch_size, num_blocks), device=device)

        if self.model.antithetic_sampling:
            num_samples = _eps_b.numel()
            offset = torch.arange(num_samples, device=device) / num_samples
            offset = offset.view(_eps_b.shape)
            _eps_b = (_eps_b / num_samples + offset) % 1

        t = _eps_b.repeat_interleave(block_size, dim=-1)
        t = t[:, :seq_len]
        t = t * (self.model.sampling_eps_max - self.model.sampling_eps_min) + self.model.sampling_eps_min
        
        # Compute loss_scale and noise probability `p` from `t`
        loss_scale, p = self.model.noise_scheduler(t)
        
        move_probabilities = torch.rand(batch_size, seq_len, device=device)
        move_chance = p
        
        text_token_mask = (labels != IGNORE_INDEX)
        move_indices = (move_probabilities <= move_chance) & text_token_mask
        
        mask_embed = self.get_input_embeddings()(torch.tensor([self.model.mask_token_id], device=device))
        xt_embeds = torch.where(move_indices.unsqueeze(-1), mask_embed, inputs_embeds)
        
        avg_noise_level = torch.mean(move_chance).item()

        bd3lm_inputs = torch.cat([xt_embeds, inputs_embeds], dim=1)  # [B, 2L, H]

        return bd3lm_inputs, move_indices, loss_scale, inputs_embeds, avg_noise_level
    
    def _compute_bd3lm_loss_embedding(self, logits, labels, move_indices, loss_scale):
        masked_positions = move_indices & (labels != IGNORE_INDEX)
        
        if not masked_positions.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        logits_flat = logits[masked_positions]
        labels_flat = labels[masked_positions]
        token_loss_unweighted = F.cross_entropy(logits_flat, labels_flat, reduction='none')
        
        loss_scale_flat = loss_scale[masked_positions]
        weighted_loss = token_loss_unweighted * loss_scale_flat.abs()
        
        prompt_index = (labels == IGNORE_INDEX).to(torch.int64)
        noisy_data_length = torch.sum((1 - prompt_index), dim=-1, keepdim=True)
        noisy_data_length = torch.max(noisy_data_length, torch.ones_like(noisy_data_length))
        noisy_data_length_flat = noisy_data_length.expand_as(labels)[masked_positions]
        
        loss = torch.sum(weighted_loss / noisy_data_length_flat) / labels.shape[0]
        return loss
    
    # ========================================================================
    # ========================================================================
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        image_grid_thws: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = None,
        steps: int = 4,
        gen_length: int = 128,
        temperature: float = 0.0,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if modalities is None:
            modalities = ["image"]
        
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        if images is not None:
            (_, _, _, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None,
                images, modalities=modalities, image_sizes=image_sizes, image_grid_thws=image_grid_thws,
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.get_model().embed_tokens(inputs)
        
        kwargs.pop("input_ids", None)
        
        return self.generate_with_bd3lm(
            inputs_embeds=inputs_embeds,
            steps=steps,
            gen_length=gen_length,
            temperature=temperature,
            **kwargs,
        )
    
    @torch.no_grad()
    def generate_with_bd3lm(
        self,
        inputs_embeds: torch.FloatTensor,
        steps: int = 4,
        gen_length: int = 128,
        temperature: float = 0.0,
        **kwargs,
    ):
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        prompt_len = inputs_embeds.shape[1]
        block_size = self.model.bd3lm_block_size
        mask_id = self.model.mask_token_id

        is_full_diffusion_ablation = block_size >= (prompt_len + gen_length)
        
        if is_full_diffusion_ablation:
            total_length = prompt_len + gen_length
            num_blocks = 1
        else:
            num_blocks = (prompt_len + gen_length + block_size - 1) // block_size
            total_length = num_blocks * block_size
        
        x_ids = torch.full((batch_size, total_length), mask_id, dtype=torch.long, device=device)
        
        mask_embed = self.get_input_embeddings()(torch.tensor([mask_id], device=device))
        x_embeds = mask_embed.repeat(batch_size, total_length, 1)
        x_embeds[:, :prompt_len] = inputs_embeds.clone()
        
        prompt_logits = self.lm_head(inputs_embeds)
        prompt_ids_reconstructed = torch.argmax(prompt_logits, dim=-1)
        x_ids[:, :prompt_len] = prompt_ids_reconstructed

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device)).to(inputs_embeds.dtype)
        block_diffusion_mask_bool = block_mask.repeat_interleave(block_size, dim=0)\
                                              .repeat_interleave(block_size, dim=1).unsqueeze(0)
        block_diffusion_mask = block_diffusion_mask_bool.unsqueeze(1)
        block_diffusion_mask = torch.where(
            block_diffusion_mask == 0., 
            torch.full_like(block_diffusion_mask, float('-inf')), 
            0.
        )
        
        if is_full_diffusion_ablation:
            block_diffusion_mask = block_diffusion_mask[:,:, :total_length, :total_length]

        position_ids = torch.arange(total_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        prefill_blocks = prompt_len // block_size
        prefill_length = prefill_blocks * block_size
        
        past_key_values = DynamicCache()

        if prefill_length > 0:
            prefill_embeds = x_embeds[:, :prefill_length]
            prefill_mask = block_diffusion_mask[:, :, :prefill_length, :prefill_length]
            prefill_pos_ids = position_ids[:, :prefill_length]
            
            prefill_outputs = self.model(
                inputs_embeds=prefill_embeds,
                attention_mask=prefill_mask,
                position_ids=prefill_pos_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True
            )
            past_key_values = prefill_outputs.past_key_values
        
        if is_full_diffusion_ablation:
            num_transfer_tokens = self.get_bd3lm_num_transfer_tokens(gen_length, steps)
        else:
            num_transfer_tokens = self.get_bd3lm_num_transfer_tokens(block_size, steps)

        for block_idx in range(prefill_blocks, num_blocks):
            block_start = block_idx * block_size
            block_end = block_start + block_size
            
            cur_block_embeds = x_embeds[:, block_start:block_end].clone()
            cur_block_ids = x_ids[:, block_start:block_end]
            
            cur_mask = block_diffusion_mask[:, :, block_start:block_end, :block_end]
            cur_pos_ids = position_ids[:, block_start:block_end]
            
            for step in range(steps + 1):
                is_mask = torch.all(torch.abs(cur_block_embeds - mask_embed) < 1e-5, dim=-1)
                if not is_mask.any():
                    _ = self.model(
                        inputs_embeds=cur_block_embeds,
                        attention_mask=cur_mask,
                        position_ids=cur_pos_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True
                    )
                    break

                outputs = self.model(
                    inputs_embeds=cur_block_embeds,
                    attention_mask=cur_mask,
                    position_ids=cur_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False
                )
                logits = self.lm_head(outputs[0]).float()

                top_k = kwargs.get('top_k', 0)
                top_p = kwargs.get('top_p', 1.0)
                x0, x0_p = self._sample_with_temperature_topk_topp(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                remasking_strategy = kwargs.get('remasking_strategy', 'low_confidence_static')
                num_to_transfer = num_transfer_tokens[step].item()

                transfer_mask = torch.zeros_like(x0, dtype=torch.bool, device=device)
                if remasking_strategy == 'low_confidence_static':
                    confidence = torch.where(is_mask, x0_p, -torch.inf)
                    
                    for j in range(confidence.shape[0]):
                        num_masks = is_mask[j].sum().item()
                        k = min(num_to_transfer, num_masks)
                        
                        if k > 0 and not torch.all(torch.isinf(confidence[j])):
                            _, idx = torch.topk(confidence[j], k)
                            transfer_mask[j, idx] = True
                
                elif remasking_strategy == 'low_confidence_dynamic':
                    confidence_threshold = kwargs.get('confidence_threshold', 0.85)
                    confidence = torch.where(is_mask, x0_p, -torch.inf)
                    
                    for j in range(confidence.shape[0]):
                        high_conf_mask = confidence[j] > confidence_threshold
                        num_high_confidence = high_conf_mask.sum().item()
                        
                        if num_high_confidence >= num_to_transfer:
                            transfer_mask[j] = high_conf_mask
                        else:
                            num_masks = is_mask[j].sum().item()
                            k = min(num_to_transfer, num_masks)
                            if k > 0:
                                _, idx = torch.topk(confidence[j], k)
                                transfer_mask[j, idx] = True
                
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")
                
                cur_block_ids = torch.where(transfer_mask, x0, cur_block_ids)
                x0_embeds = self.get_input_embeddings()(x0)
                cur_block_embeds = torch.where(transfer_mask.unsqueeze(-1), x0_embeds, cur_block_embeds)
            
            x_embeds[:, block_start:block_end] = cur_block_embeds
            x_ids[:, block_start:block_end] = cur_block_ids
            
            if block_end > prompt_len:
                gen_start_in_block = max(prompt_len, block_start)
                gen_ids_check = x_ids[:, gen_start_in_block:block_end]
                eot_token_id = 126348
                if (gen_ids_check == eot_token_id).any():
                    break

        return x_ids[:, prompt_len:prompt_len + gen_length]
    
    @staticmethod
    def _top_k_logits(logits, k):
        if k <= 0:
            return logits
        else:
            values, _ = torch.topk(logits, k)
            min_values = values[..., -1, None]
            return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

    @staticmethod
    def _top_p_logits(logits, p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(
            torch.full_like(logits, False, dtype=torch.bool),
            -1, sorted_indices, sorted_mask
        )
        logits = logits.masked_fill(mask_indices, float('-inf'))
        return logits

    def _sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        
        logits_2d = logits.reshape(-1, vocab_size)
        
        probs_original = F.softmax(logits_2d, dim=-1)
        
        if temperature == 0:
            token = torch.argmax(logits_2d, dim=-1, keepdim=True)
        else:
            logits_modified = logits_2d.clone()
            if temperature != 1.0:
                logits_modified = logits_modified / temperature
            if top_k > 0:
                logits_modified = self._top_k_logits(logits_modified, top_k)
            if top_p < 1.0:
                logits_modified = self._top_p_logits(logits_modified, top_p)
            
            probs_modified = F.softmax(logits_modified, dim=-1)
            token = torch.multinomial(probs_modified, num_samples=1)
        
        token_prob = torch.gather(probs_original, -1, token)

        return token.view(*orig_shape), token_prob.view(*orig_shape)
    
    def get_bd3lm_num_transfer_tokens(self, block_length, steps):
        if steps == 0:
            return torch.zeros(0, dtype=torch.int64)
        
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.zeros(steps + 1, dtype=torch.int64) + base
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens
    
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        inputs_embeds=None, 
        **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_grid_thws = kwargs.pop("image_grid_thws", None)
        
        inputs = super().prepare_inputs_for_generation(
            input_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            **kwargs
        )
        
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if image_grid_thws is not None:
            inputs["image_grid_thws"] = image_grid_thws
        
        return inputs


AutoConfig.register("llava_llada_bd3lm", LlavaLladaBD3LMConfig)
AutoModelForCausalLM.register(LlavaLladaBD3LMConfig, LlavaLladaBD3LMForCausalLM)

