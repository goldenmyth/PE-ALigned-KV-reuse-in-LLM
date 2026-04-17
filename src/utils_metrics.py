import re
import string
import torch
import torch.nn.functional as F
from collections import Counter
from scipy.stats import spearmanr

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0: return int(gold_toks == pred_toks)
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)

    return (2 * precision * recall) / (precision + recall)

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def calculate_comprehensive_metrics(base_logits, test_logits, base_attn, test_attn, 
                                   base_gen_text, test_gen_text, ground_truth_text, tokenizer, k=50):
    metrics = {}
    metrics['EM'] = compute_exact(ground_truth_text, test_gen_text)
    metrics['F1'] = compute_f1(ground_truth_text, test_gen_text)

    with torch.no_grad():
        b_log = base_logits[0][0].float() if isinstance(base_logits, tuple) else base_logits[0, -1, :].float()
        t_log = test_logits[0][0].float() if isinstance(test_logits, tuple) else test_logits[0, -1, :].float()
        
        b_p = F.softmax(b_log, dim=-1)
        t_p = F.softmax(t_log, dim=-1)

        metrics['Top1_Agree'] = int(torch.argmax(b_log) == torch.argmax(t_log))
        
        top_k_val, top_k_idx = torch.topk(b_p, k)
        p_sub = b_p[top_k_idx] / b_p[top_k_idx].sum()
        q_sub = t_p[top_k_idx] / t_p[top_k_idx].sum()
        metrics['KL_Div'] = F.kl_div(q_sub.log(), p_sub, reduction='sum').item()

        gt_ids = tokenizer.encode(ground_truth_text, add_special_tokens=False)
        if gt_ids:
            gt_id = gt_ids[0]
            sorted_ids = torch.argsort(t_log, descending=True)
            metrics['GT_Rank'] = (sorted_ids == gt_id).nonzero(as_tuple=True)[0].item() + 1
            metrics['NLL_Loss'] = F.cross_entropy(t_log.unsqueeze(0), torch.tensor([gt_id], device=t_log.device)).item()

        # Attention Correlation
        def get_attn_weights(attn):
            # Extract the last request token, the last layer, and average the heads
            raw = attn[0][-1][0] if isinstance(attn, tuple) else attn[-1][0]
            return raw.mean(dim=0)[-1, :].cpu().float().numpy()

        a_base = get_attn_weights(base_attn)
        a_test = get_attn_weights(test_attn)
        min_l = min(len(a_base), len(a_test))
        metrics['Attn_Corr'], _ = spearmanr(a_base[:min_l], a_test[:min_l])

    return metrics