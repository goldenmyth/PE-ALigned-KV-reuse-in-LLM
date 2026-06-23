import os
import torch
import gc
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from src.config_loader import config
from src.model_engine import load_model, run_inference, set_seed
from src.utils_rope import shift_cache, identity_transform
from src.utils_cache import precompute_segments, assemble_cache
from src.utils_metrics import calculate_comprehensive_metrics, calculate_baseline_metrics
from src.utils_data import get_data_for_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    
def main():
    model, tokenizer = load_model()
    set_seed(config.SEED)
    
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    target_debug_ids = [0, 2, 4]    

    for ds_name, ds_cfg in config.get_enabled_datasets().items():
        print(f"\nProcessing: {ds_name.upper()}")
        dataset = load_dataset(ds_cfg['path'], ds_cfg['subset'], split=ds_cfg['split'])
        
        #if ds_name == "musique":
        #    dataset = dataset.filter(lambda x: len([p for p in x['paragraphs'] if p['is_supporting']]) > 1)
        
        dataset = dataset.select(range(min(ds_cfg['num_samples'], len(dataset))))
        ds_results = []

        task_type = ds_cfg.get('task_type', 'qa')
        max_tokens = ds_cfg.get('max_new_tokens', 20)
        compute_attn = ds_cfg.get('compute_attn', True)
        for idx, sample in enumerate(tqdm(dataset, desc=ds_name)):
            try:
                # 1. Data preparation
                pre_txt, p_txts, que_txt, target_answer = get_data_for_dataset(ds_name, sample)
                
                get_ids = lambda t: tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                ids_que = get_ids(que_txt)
                
                # 2. BASELINE
                full_prompt = torch.cat([get_ids(pre_txt)] + [get_ids(p) for p in p_txts] + [ids_que], dim=1)
                res_b, logits_b, attn_b, _ = run_inference(model, tokenizer, full_prompt, max_new=max_tokens, compute_attn=compute_attn)

                b_metrics = calculate_baseline_metrics(target_answer, res_b, task_type)
            
                case_report = {
                    "Case_ID": idx,
                    "Target": target_answer,
                    "Baseline_Output": res_b.replace('\n', ' '),
                    **b_metrics
                }
                
                del full_prompt
                cleanup()

                # 3. PRECOMPUTE
                cached_segments = precompute_segments(model, tokenizer, [pre_txt] + p_txts)

                # 4. STRATEGIES
                for strategy, transform in [("Aligned", shift_cache), ("Naive", identity_transform)]:
                    cache = assemble_cache(cached_segments, transform, model.config)
                    res, logits, attn, _ = run_inference(model, tokenizer, ids_que, cache_obj=cache, max_new=max_tokens, compute_attn=compute_attn) #g_ids
                    
                    try:
                        m = calculate_comprehensive_metrics(
                            logits_b, logits, attn_b, attn, 
                            res_b, res, target_answer, tokenizer, task_type=task_type
                        )
                                    
                        case_report[f"{strategy}_Output"] = res.replace('\n', ' ')
                        for metric_key, val in m.items():
                            case_report[f"{strategy}_{metric_key}"] = val

                    finally:
                        '''if idx in target_debug_ids:
                            if strategy == "Aligned":
                                print(f"IDX: {idx}   Target: {target_answer};  Aligned: {res}")
                            else:
                                print(f" IDX: {idx}  Target: {target_answer};  Naive: {res}")
                            print(f"DEBUG IDs: {g_ids.tolist()}")
                            print(f"DEBUG Tokens: {tokenizer.convert_ids_to_tokens(g_ids)}")

                            for step_idx in range(len(logits)):
                                probs = torch.softmax(logits[step_idx][0], dim=-1)
                                top_k = torch.topk(probs, 3)
                                chosen_token = tokenizer.decode(g_ids[step_idx])
                                print(f"Step {step_idx} | Chosen: '{chosen_token}' (id:{g_ids[step_idx].item()})")
                                for i in range(3):
                                    print(f"Top-{i}: '{tokenizer.decode(top_k.indices[i])}' p={top_k.values[i].item():.4f}")
                        
                        print("="*10 + "\n")'''
                        del cache, res, logits, attn
                        cleanup()
                
                ds_results.append(case_report)

                del res_b, logits_b, attn_b, cached_segments, ids_que
                cleanup()

            except Exception as e:
                print(f"\nError in {ds_name} ID {idx}: {e}")
                cleanup()
                continue

        pd.DataFrame(ds_results).to_csv(f"{config.SAVE_DIR}/{ds_name}_results7B.csv", index=False)
        cleanup()

if __name__ == "__main__":
    
    main()
