import torch
from transformers import DynamicCache

def get_kv_cache_list(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    kv = outputs.past_key_values
    res = []
    if hasattr(kv, "key_cache"):
        for i in range(len(kv.key_cache)):
            res.append((kv.key_cache[i], kv.value_cache[i], None))
    else:
        for item in kv:
            res.append((item[0], item[1], None))
    return res, inputs.input_ids.shape[1]

def pack_to_cache(kv_list):
    new_cache = DynamicCache()
    seq_len = kv_list[0][0].shape[-2]
    for i, (k, v, _) in enumerate(kv_list):
        new_cache.update(k, v, i)
    new_cache._seen_tokens = seq_len
    return new_cache

def get_kv_cache_size_mb(cache_obj):
    total_bytes = 0
    if hasattr(cache_obj, "key_cache") and hasattr(cache_obj, "value_cache"):
        for i in range(len(cache_obj.key_cache)):
            total_bytes += cache_obj.key_cache[i].nbytes
            total_bytes += cache_obj.value_cache[i].nbytes
    else:
        for item in cache_obj:
            total_bytes += item[0].nbytes + item[1].nbytes
    return total_bytes / (1024 * 1024)

def precompute_segments(model, tokenizer, text_list):
    cached_data = []
    for txt in text_list:
        kv, length = get_kv_cache_list(model, tokenizer, txt)
        cached_data.append((kv, length))
    return cached_data

def assemble_cache(cached_data, transform_fn, model_config):
    first_kv, first_len = cached_data[0]
    current_kv = [(k.clone(), v.clone(), None) for k, v, _ in first_kv]
    offset = first_len

    for kv_p, len_p in cached_data[1:]:
        kv_to_add = transform_fn(kv_p, offset, model_config)
        for layer in range(len(current_kv)):
            current_kv[layer] = (
                torch.cat([current_kv[layer][0], kv_to_add[layer][0]], dim=2),
                torch.cat([current_kv[layer][1], kv_to_add[layer][1]], dim=2),
                None
            )
        offset += len_p
    return pack_to_cache(current_kv)