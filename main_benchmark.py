import os
import torch
import time
import pandas as pd
from tqdm import tqdm
from src.config_loader import config
from src.model_engine import load_model, run_inference, set_seed
from src.utils_rope import shift_cache, identity_transform
from src.utils_cache import get_kv_cache_list, precompute_segments, assemble_cache, get_kv_cache_size_mb
from src.utils_metrics import calculate_comprehensive_metrics
from src.utils_data import get_chat_parts
from datasets import load_dataset

def main():
    model, tokenizer = load_model()
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_SUBSET, split="validation")
    dataset = dataset.filter(lambda x: len([p for p in x['paragraphs'] if p['is_supporting']]) > 1)
    dataset = dataset.select(range(min(config.NUM_SAMPLES, len(dataset))))

    results_log = []
    
    for idx, sample in enumerate(tqdm(dataset)):
        paragraphs = [p['paragraph_text'] for p in sample['paragraphs'] if p['is_supporting']]
        pre_txt, p_txts, que_txt = get_chat_parts(paragraphs, sample['question'])
        
        get_ids = lambda t: tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        ids_que = get_ids(que_txt)
        full_prompt = torch.cat([get_ids(pre_txt)] + [get_ids(p) for p in p_txts] + [ids_que], dim=1)

        # 1. Baseline
        res_b, logits_b, attn_b = run_inference(model, tokenizer, full_prompt)
        
        # 2. Precompute
        cached_segments = precompute_segments(model, tokenizer, [pre_txt] + p_txts)
        
        # 3. Aligned & Naive
        for strategy, transform in [("Aligned", shift_cache), ("Naive", identity_transform)]:
            cache = assemble_cache(cached_segments, transform, model.config)
            res, logits, attn = run_inference(model, tokenizer, ids_que, cache_obj=cache)
            
            m = calculate_comprehensive_metrics(logits_b, logits, attn_b, attn, res_b, res, sample['answer'], tokenizer)
            m.update({"Strategy": strategy, "Case_ID": idx})
            results_log.append(m)

    # Saving
    df = pd.DataFrame(results_log)
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    df.to_csv(f"{config.SAVE_DIR}/musique_results.csv", index=False)

if __name__ == "__main__":
    set_seed(config.SEED)
    main()