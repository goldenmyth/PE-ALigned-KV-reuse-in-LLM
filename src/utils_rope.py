import torch

def rotate_half(x):
    # GPT-NeoX style
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
   
    return torch.cat((-x2, x1), dim=-1)

def shift_cache(kv_list, offset, model_config):
    sample_tensor = kv_list[0][0]
    device, dtype = sample_tensor.device, sample_tensor.dtype
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    base = getattr(model_config, "rope_theta", 1000000.0)

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim))
    t = torch.tensor([offset], device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    res = []
    for k, v, mask in kv_list:
        k_fp32 = k.to(torch.float32)
        k_rotated = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)

        #k_rotated = torch.clamp(k_rotated, min=-256.0, max=256.0)
        '''# Debug
        norm_before = torch.norm(k, dim=-1).mean()
        norm_after = torch.norm(k_rotated, dim=-1).mean()

        max_val = torch.max(torch.abs(k_rotated))
        if max_val > 100:
            print(f"!!! ПРЕДУПРЕЖДЕНИЕ: Очень высокие значения в К: {max_val:.4f}")
        if torch.abs(norm_before - norm_after) > 1e-4:
            print(f"!!! ОШИБКА: Поворот меняет норму! {norm_before:.4f} -> {norm_after:.4f}")
        if torch.isnan(k_rotated.to(dtype)).any(): print(f"Сдвиг породил NaN на офсете {offset}")
        if torch.isinf(k_rotated.to(dtype)).any(): print(f"Сдвиг породил Inf на офсете {offset}")'''

        res.append((k_rotated.to(dtype), v, mask))
    
    return res

def identity_transform(kv, offset, model_config):
    return kv
    