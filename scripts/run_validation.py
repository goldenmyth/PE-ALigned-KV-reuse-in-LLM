import os
import torch
import torch.nn.functional as F
import logging
import json
import time
from src.config_loader import config
from src.model_engine import load_model
from src.utils_cache import get_kv_cache_list
from src.utils_rope import shift_cache, rotate_half

# Logging
os.makedirs(config.SAVE_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.SAVE_DIR, "validation_log.txt")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Support function

def get_kv_cache_with_pos_ids(model, tokenizer, text, position_ids):
    "Modify func for baseline"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model(inputs.input_ids, position_ids=position_ids, use_cache=True)
    
    kv = outputs.past_key_values
    res = []

    if hasattr(kv, "key_cache"):
        for i in range(len(kv.key_cache)):
            res.append((kv.key_cache[i], kv.value_cache[i], None))
    else:
        for item in kv:
            res.append((item[0], item[1], None))
    return res

def rotate_q_safe(q, pos_offset, cfg, device):
    """ Rotate Query (similar shift_cache for Key)"""
    head_dim = q.shape[-1]
    base = getattr(cfg, "rope_theta", 1000000.0)
    
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim))
    t = torch.tensor([pos_offset], device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    
    emb = torch.cat((freqs, freqs), dim=-1).to(q.dtype)
    cos, sin = emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
    
    return (q * cos) + (rotate_half(q) * sin)

# Main validation

def run_full_validation():
    logger.info("STARTING ROPE VALIDATION")
    model, tokenizer = load_model()
    device = model.device
    model_config = model.config
    
    val_metrics = {}

    # 1. Test: Reversibility
    logger.info("[Step 1] Reversibility Test (Synthetic float32)")
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    num_heads = model_config.num_attention_heads
    k_orig = torch.randn(1, num_heads, 10, head_dim, device=device, dtype=torch.float32)
    v_orig = torch.randn_like(k_orig)
    kv_orig = [(k_orig, v_orig, None)]
    
    offset = 128
    shifted = shift_cache(kv_orig, offset, model_config)
    restored = shift_cache(shifted, -offset, model_config)
    
    rev_diff = torch.abs(k_orig - restored[0][0]).max().item()
    logger.info(f"    Max Diff: {rev_diff:.2e} -> {'PASSED' if rev_diff < 1e-5 else 'FAILED'}")
    val_metrics['reversibility_diff'] = rev_diff

    # 2. Test: Attention Invariance
    logger.info("[Step 2] Attention Invariance Test (Real Text)")
    text = "The quick brown fox jumps over the lazy dog."
    kv_text, seq_len = get_kv_cache_list(model, tokenizer, text)
    
    k_real = kv_text[0][0].to(torch.float32)
    q_real = torch.randn_like(k_real[:, :, -1:, :])
    
    score_0 = torch.einsum("bhid,bhjd->bhij", q_real, k_real[:, :, -1:, :])
    
    shifted_k = shift_cache([(k_real, kv_text[0][1], None)], offset, model_config)
    shifted_q = rotate_q_safe(q_real, offset, model_config, device)
    
    score_offset = torch.einsum("bhid,bhjd->bhij", shifted_q, shifted_k[0][0][:, :, -1:, :])
    
    attn_diff = torch.abs(score_0 - score_offset).max().item()
    logger.info(f"    Score Diff: {attn_diff:.2e} -> {'PASSED' if attn_diff < 1e-5 else 'FAILED'}")
    val_metrics['attention_invariance_diff'] = attn_diff

    # 3. Test: Consistency: Shift vs. Native
    logger.info("[Step 3] Native Consistency Test (Manual Shift vs. Model Internal)")
    
    pos_ids = torch.arange(offset, offset + seq_len, device=device).unsqueeze(0)
    kv_native = get_kv_cache_with_pos_ids(model, tokenizer, text, pos_ids)
    k_native = kv_native[0][0].to(torch.float32)
    
    k_manual = shifted_k[0][0]
    
    abs_diff = torch.abs(k_native - k_manual).max().item()
    cos_sim = F.cosine_similarity(k_native.flatten(), k_manual.flatten(), dim=0).item()
    
    logger.info(f"    Max Abs Diff: {abs_diff:.4f}")
    logger.info(f"    Cosine Similarity: {cos_sim:.10f} -> {'PASSED' if cos_sim > 0.9999 else 'FAILED'}")
    
    val_metrics['consistency_abs_diff'] = abs_diff
    val_metrics['consistency_cosine_sim'] = cos_sim

    with open(os.path.join(config.SAVE_DIR, "validation_results.json"), "w") as f:
        json.dump(val_metrics, f, indent=4)
    logger.info(f"VALIDATION FINISHED. Results saved to {config.SAVE_DIR}")

if __name__ == "__main__":
    run_full_validation()