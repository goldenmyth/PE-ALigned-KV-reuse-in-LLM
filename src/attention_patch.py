import torch
import torch.nn as nn
import math
import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling

def patched_qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor = None,
    past_key_values = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q_len = hidden_states.shape[1]

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # RoPE
    cos, sin = position_embeddings
    query_states, key_states = qwen2_modeling.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    # GQA
    key_states = qwen2_modeling.repeat_kv(key_states, self.num_key_value_groups)
    value_states = qwen2_modeling.repeat_kv(value_states, self.num_key_value_groups)

    dtype_orig = query_states.dtype
    
    scaling = getattr(self, "scaling", self.head_dim**-0.5)

    use_fp32 = (q_len < 256) and (past_key_values is not None)

    if use_fp32:
        query_states = query_states.to(torch.float32)
        key_states = key_states.to(torch.float32)
        
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    smax_type = torch.float32 if use_fp32 else dtype_orig
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=smax_type).to(dtype_orig)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights

def apply_attention_patch():
    qwen2_modeling.Qwen2Attention.forward = patched_qwen2_attention_forward
