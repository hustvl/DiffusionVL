# coding=utf-8
# Copyright 2025 The HustVL Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen2.5-VL (https://github.com/QwenLM/Qwen2.5-VL),
# LLaDA-V (https://github.com/ML-GSAI/LLaDA-V), and
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

"""DiffusionVL-QwenVL model implementation."""

import os
import json
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.configuration_utils import PretrainedConfig as HFPretrainedConfig
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache
from transformers.utils import logging
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention,
    Qwen2_5_VLTextModel as Qwen2_5_VLTextModelOriginal,
    Qwen2_5_VLPreTrainedModel,
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)

from llava.constants import IGNORE_INDEX
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.utils import rank0_print

logger = logging.get_logger(__name__)

class DiffusionVLQwenVLConfig(HFPretrainedConfig):
    """Configuration class for DiffusionVL-QwenVL model."""

    model_type = "diffusionvl_qwenvl"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151655,
        video_token_id=151656,
        enable_bd3lm=False,
        bd3lm_block_size=4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
        if text_config is None:
            text_config = {}

        self.vision_config = Qwen2_5_VLVisionConfig(**vision_config)
        self.text_config = Qwen2_5_VLTextConfig(**text_config)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        self.enable_bd3lm = enable_bd3lm
        self.bd3lm_block_size = bd3lm_block_size

        if self.enable_bd3lm:
            # BD3-LM training parameters
            self.bd3lm_antithetic_sampling = True
            self.bd3lm_sampling_eps_min = 1e-3
            self.bd3lm_sampling_eps_max = 1.0

        for key, value in self.text_config.to_dict().items():
            setattr(self, key, value)

    def to_dict(self):
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        return output


class DiffusionVLQwenVLAttention(Qwen2_5_VLAttention):
    """Attention layer with non-causal mask for BD3-LM diffusion."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        store_kv = kwargs.pop("store_kv", True)
        flash_attn_kwargs = kwargs

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.config.rope_scaling["mrope_section"]
        )

        if past_key_values is not None:
            if store_kv:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                if self.layer_idx < len(past_key_values):
                    past_key_states, past_value_states = past_key_values[self.layer_idx]
                    key_states = torch.cat([past_key_states, key_states], dim=2)
                    value_states = torch.cat([past_value_states, value_states], dim=2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,
            **flash_attn_kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class DiffusionVLQwenVLTextModel(Qwen2_5_VLTextModelOriginal):
    """Text model with diffusion-compatible attention."""

    def __init__(self, config):
        super().__init__(config)

        # Replace attention layers with diffusion-compatible version
        for layer in self.layers:
            original_layer_idx = layer.self_attn.layer_idx
            layer.self_attn = DiffusionVLQwenVLAttention(config, layer_idx=original_layer_idx)

        # BD3-LM extension
        if getattr(config, 'enable_bd3lm', False):
            self._init_bd3lm_components(config)

    def _init_bd3lm_components(self, config):
        """Initialize BD3-LM components."""
        from .bd3lm_utils import LogLinearNoise

        self.noise_scheduler = LogLinearNoise()
        self.mask_token_id = 151671
        self.bd3lm_block_size = config.bd3lm_block_size
        self.antithetic_sampling = getattr(config, 'bd3lm_antithetic_sampling', True)
        self.sampling_eps_min = getattr(config, 'bd3lm_sampling_eps_min', 1e-3)
        self.sampling_eps_max = getattr(config, 'bd3lm_sampling_eps_max', 1.0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sample_mode: Optional[bool] = False,
        store_kv: Optional[bool] = False,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs['store_kv'] = store_kv

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            causal_mask_mapping = {
                "full_attention": attention_mask,
                "sliding_attention": attention_mask if self.has_sliding_layers else None,
            }
        if cache_position is None:
            past_seen_tokens = past_key_values_length
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        if position_ids is not None and position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class DiffusionVLQwenVLForCausalLM_Base(Qwen2_5_VLPreTrainedModel):
    """Base CausalLM model with BD3-LM diffusion logic."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DiffusionVLQwenVLTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _apply_bd3lm_noise_embedding(self, inputs_embeds, labels):
        """Apply BD3-LM noise in embedding space."""
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

        loss_scale, p = self.model.noise_scheduler(t)

        move_probabilities = torch.rand(batch_size, seq_len, device=device)
        move_chance = p
        text_token_mask = (labels != IGNORE_INDEX)
        move_indices = (move_probabilities <= move_chance) & text_token_mask

        mask_embed = self.get_input_embeddings()(torch.tensor([151671], device=device))
        xt_embeds = torch.where(move_indices.unsqueeze(-1), mask_embed, inputs_embeds)

        avg_noise_level = torch.mean(move_chance).item()
        bd3lm_inputs = torch.cat([xt_embeds, inputs_embeds], dim=1)
        return bd3lm_inputs, move_indices, loss_scale, inputs_embeds, avg_noise_level

    def _compute_bd3lm_loss_embedding(self, logits, labels, move_indices, loss_scale):
        """Compute BD3-LM loss."""
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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """BD3-LM forward pass."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Apply BD3-LM noise
        bd3lm_inputs, move_indices, loss_scale, x0_embeds, avg_noise_level = self._apply_bd3lm_noise_embedding(inputs_embeds, labels)

        # Process attention mask
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
            extended_attention_mask = torch.cat([attention_mask, attention_mask], dim=1)
            query_validity_mask = extended_attention_mask.unsqueeze(-1)
            key_validity_mask = extended_attention_mask.unsqueeze(-2)
            combined_padding_mask_2d = query_validity_mask & key_validity_mask
            mask = mask & combined_padding_mask_2d

        attention_mask_4d = torch.zeros(mask.shape, dtype=inputs_embeds.dtype, device=device)
        attention_mask_4d.masked_fill_(~mask, torch.finfo(inputs_embeds.dtype).min)
        attention_mask_4d = attention_mask_4d.unsqueeze(1)

        if position_ids is None:
            pos_ids_part = torch.arange(seq_len, device=device)
            position_ids = torch.cat([pos_ids_part, pos_ids_part], dim=0)

        outputs = self.model(
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

    @torch.no_grad()
    def generate_with_bd3lm(
        self,
        inputs_embeds: torch.FloatTensor,
        steps: int = 4,
        gen_length: int = 128,
        temperature: float = 0.0,
        **kwargs,
    ):
        """BD3-LM inference with KV-Cache and SDAR strategy."""
        from transformers.cache_utils import DynamicCache

        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        prompt_len = inputs_embeds.shape[1]
        block_size = self.model.bd3lm_block_size
        mask_id = 151671

        is_full_diffusion_ablation = block_size >= (prompt_len + gen_length)

        if is_full_diffusion_ablation:
            rank0_print("Full-Diffusion ablation mode enabled.")
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
        block_diffusion_mask_bool = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1).unsqueeze(0)
        block_diffusion_mask = block_diffusion_mask_bool.unsqueeze(1)
        block_diffusion_mask = torch.where(block_diffusion_mask == 0., torch.full_like(block_diffusion_mask, float('-inf')), 0.)
        if is_full_diffusion_ablation:
            block_diffusion_mask = block_diffusion_mask[:, :, :total_length, :total_length]

        position_ids = torch.arange(total_length, device=device).unsqueeze(0).expand(batch_size, -1)

        prefill_blocks = prompt_len // block_size
        prefill_length = prefill_blocks * block_size

        past_key_values = DynamicCache()
        if prefill_length > 0:
            prefill_embeds = x_embeds[:, :prefill_length]
            prefill_mask = block_diffusion_mask[:, :, :prefill_length, :prefill_length]
            prefill_pos_ids = position_ids[:, :prefill_length]
            model_mask = {"full_attention": prefill_mask, "sliding_attention": prefill_mask}

            prefill_outputs = self.model(
                inputs_embeds=prefill_embeds,
                attention_mask=model_mask,
                position_ids=prefill_pos_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True
            )
            past_key_values = prefill_outputs.past_key_values

        num_transfer_tokens = self.get_bd3lm_num_transfer_tokens(block_size, steps)

        for block_idx in range(prefill_blocks, num_blocks):
            block_start = block_idx * block_size
            block_end = block_start + block_size

            cur_block_embeds = x_embeds[:, block_start:block_end].clone()
            cur_block_ids = x_ids[:, block_start:block_end]

            cur_mask = block_diffusion_mask[:, :, block_start:block_end, :block_end]
            cur_pos_ids = position_ids[:, block_start:block_end]
            model_mask = {"full_attention": cur_mask, "sliding_attention": cur_mask}

            for step in range(steps + 1):
                is_mask = torch.all(torch.abs(cur_block_embeds - mask_embed) < 1e-5, dim=-1)
                if not is_mask.any():
                    _ = self.model(
                        inputs_embeds=cur_block_embeds,
                        attention_mask=model_mask,
                        position_ids=cur_pos_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True
                    )
                    break

                outputs = self.model(
                    inputs_embeds=cur_block_embeds,
                    attention_mask=model_mask,
                    position_ids=cur_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False
                )
                logits = self.lm_head(outputs[0]).float()

                top_k = kwargs.get('top_k', 0)
                top_p = kwargs.get('top_p', 1.0)
                x0, x0_p = self._sample_with_temperature_topk_topp(logits, temperature=temperature, top_k=top_k, top_p=top_p)
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
                eos_token_id = 151645
                if eos_token_id in gen_ids_check:
                    break

        return x_ids[:, prompt_len:prompt_len + gen_length]

    @staticmethod
    def _top_k_logits(logits, k):
        if k <= 0:
            return logits
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
        mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask_indices, float('-inf'))
        return logits

    def _sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        """Sample with temperature, top-k, and top-p."""
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits_2d = logits.reshape(-1, vocab_size)

        if temperature == 0:
            token = torch.argmax(logits_2d, dim=-1, keepdim=True)
            probs_original = F.softmax(logits_2d, dim=-1)
            token_prob = torch.gather(probs_original, -1, token)
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
            token_prob = torch.gather(probs_modified, -1, token)

        return token.view(*orig_shape), token_prob.view(*orig_shape)

    def get_bd3lm_num_transfer_tokens(self, block_length, steps):
        if steps == 0:
            return torch.zeros(0, dtype=torch.int64)
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.zeros(steps + 1, dtype=torch.int64) + base
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens

class DiffusionVLQwenVLMultiModalModel(LlavaMetaModel, DiffusionVLQwenVLTextModel):
    """Multimodal model combining vision and text."""
    config_class = DiffusionVLQwenVLConfig

    def __init__(self, config):
        super(DiffusionVLQwenVLMultiModalModel, self).__init__(config)


class DiffusionVLQwenVLForCausalLM(DiffusionVLQwenVLForCausalLM_Base, LlavaMetaForCausalLM):
    """Final multimodal CausalLM model."""
    config_class = DiffusionVLQwenVLConfig

    def __init__(self, config):
        super(DiffusionVLQwenVLForCausalLM, self).__init__(config)
        self.model = DiffusionVLQwenVLMultiModalModel(config)

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
        modalities: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities=modalities,
                image_sizes=image_sizes,
                image_grid_thws=image_grid_thws
            )

        return super(DiffusionVLQwenVLForCausalLM, self).forward(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        image_grid_thws: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = None,
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
            inputs_embeds = self.get_model().embed_tokens(inputs)

        kwargs.pop("input_ids", None)
        return self.generate_with_bd3lm(inputs_embeds=inputs_embeds, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        print(f">>> Loading pre-converted DiffusionVL-QwenVL model from: {pretrained_model_name_or_path}")

        model = super(DiffusionVLQwenVLForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        vision_config_path = os.path.join(pretrained_model_name_or_path, "vision_config.json")
        if os.path.exists(vision_config_path):
            print(">>> Loading pre-converted visual components from .safetensors files...")
            with open(vision_config_path, 'r') as f:
                vision_config_dict = json.load(f)
            model.config.vision_config = PretrainedConfig.from_dict(vision_config_dict)
            model.config.vision_tower_state_dict = load_file(os.path.join(pretrained_model_name_or_path, "vision_tower.safetensors"), device='cpu')
            model.config.projector_state_dict = load_file(os.path.join(pretrained_model_name_or_path, "projector.safetensors"), device='cpu')

        print(">>> DiffusionVL-QwenVL model loaded successfully.")
        return model


# Register the model
AutoConfig.register("diffusionvl_qwenvl", DiffusionVLQwenVLConfig)
AutoModelForCausalLM.register(DiffusionVLQwenVLConfig, DiffusionVLQwenVLForCausalLM)
