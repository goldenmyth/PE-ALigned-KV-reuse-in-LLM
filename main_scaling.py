import os
import gc
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.config_loader import config
from src.model_engine import load_model, run_inference, set_seed
from src.utils_rope import shift_cache, identity_transform
from src.utils_cache import get_kv_cache_list, precompute_segments, assemble_cache, get_kv_cache_size_mb
from src.utils_metrics import normalize_answer
from src.utils_data import get_chat_parts

def run_performance_scaling(model, tokenizer, n_docs):
    # Needle in a Haystack
    filler = "The solar system consists of the Sun and objects that orbit it. " * 20
    target_num = str(random.randint(1000, 9999))
    docs = [filler for _ in range(n_docs)]
    docs[n_docs//2] += f"The secret code is {target_num}."
    
    pre_txt, p_txts, que_txt = get_chat_parts(docs, "What is the secret code?")
    
    get_ids = lambda t: tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    ids_que = get_ids(que_txt)
    full_prompt = torch.cat([get_ids(pre_txt)] + [get_ids(p) for p in p_txts] + [ids_que], dim=1)
    
    results = []
    
    # 1. Baseline
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    res_b, _, _ = run_inference(model, tokenizer, full_prompt, max_new=config.MAX_NEW_SCALING, compute_deep_metrics=False)
    end.record(); torch.cuda.synchronize()
    time_b = start.elapsed_time(end) / 1000

    # 2. Precompute
    cached_p = precompute_segments(model, tokenizer, [pre_txt] + p_txts)
    
    # 3. Aligned & Naive
    for name, transform in [("Aligned", shift_cache), ("Naive", identity_transform)]:
        torch.cuda.synchronize()
        start.record()
        cache = assemble_cache(cached_p, transform, model.config)
        res, _, _ = run_inference(model, tokenizer, ids_que, cache_obj=cache, max_new=config.MAX_NEW_SCALING, compute_deep_metrics=False)
        end.record(); torch.cuda.synchronize()
        time_res = start.elapsed_time(end) / 1000
        
        results.append({
            "Strategy": name, "Time": time_res, "Speedup": time_b / time_res, 
            "EM": 1 if target_num in res else 0, "Ctx": full_prompt.shape[1],
            "Mem_MB": get_kv_cache_size_mb(cache), "num_docs": n_docs
        })
    
    results.append({"Strategy": "Baseline", "Time": time_b, "Speedup": 1.0, "EM": 1 if target_num in res_b else 0, "Ctx": full_prompt.shape[1], "num_docs": n_docs})
    
    del cached_p, full_prompt, res_b, ids_que
    if 'cache' in locals(): del cache
    if 'res' in locals(): del res
    
    gc.collect()
    torch.cuda.empty_cache()
    return results

def main():
    set_seed(config.SEED)
    
    model, tokenizer = load_model()
    all_perf_data = []
    
    for n in tqdm(config.SCALING_DOCS):
        all_perf_data.extend(run_performance_scaling(model, tokenizer, n))
    
    df = pd.DataFrame(all_perf_data)
    df.to_csv(f"{config.SAVE_DIR}/scaling_results.csv", index=False)
    
    # Visualization
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    sns.lineplot(data=df, x='num_docs', y='Time', hue='Strategy', marker='o', ax=axes[0])
    axes[0].set_title("Latency vs Number of Documents", fontsize=14)
    axes[0].set_ylabel("Inference Time (seconds)")
    axes[0].set_xlabel("Number of Documents")

    sns.lineplot(data=df[df['Strategy'] == 'Aligned'], x='Ctx', y='Mem_MB', marker='s', color='green', ax=axes[1])
    axes[1].set_title("KV-Cache Memory Usage", fontsize=14)
    axes[1].set_ylabel("Memory (MB)")
    axes[1].set_xlabel("Context Length (Tokens)")

    tradeoff = df.groupby('Strategy').agg({'EM': 'mean', 'Speedup': 'mean'}).reset_index()
    sns.scatterplot(data=tradeoff, x='Speedup', y='EM', hue='Strategy', s=300, ax=axes[2])
    for i, txt in enumerate(tradeoff['Strategy']):
        axes[2].annotate(txt, (tradeoff.Speedup[i]+0.5, tradeoff.EM[i]), fontsize=12, fontweight='bold')

    axes[2].set_title("Quality vs Speed Trade-off", fontsize=14)
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_xlabel("Average Speedup (x)")
    axes[2].set_ylabel("Accuracy (EM)")

    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, "scaling_plots.png"))
    #print(f"Results saved to {config.SAVE_DIR}")

if __name__ == "__main__":
    main()