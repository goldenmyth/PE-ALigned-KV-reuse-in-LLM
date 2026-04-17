import torch
import time
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from src.config_loader import config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to: {seed}")

def load_model():
    bnb_config = None
    if config.LOAD_4BIT:
      bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, cache_dir=config.CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=config.DTYPE,
        device_map="auto",
        attn_implementation=config.ATTN_IMPL,
        cache_dir=config.CACHE_DIR
    ).eval()
    
    return model, tokenizer

def run_inference(model, tokenizer, input_ids, cache_obj=None, max_new=20, compute_deep_metrics=True):
    n_past = cache_obj.get_seq_length() if cache_obj else 0
    n_new = input_ids.shape[1]
    mask = torch.ones((1, n_past + n_new), device=model.device, dtype=torch.long)
    pos = torch.arange(n_past, n_past + n_new, device=model.device) if cache_obj else None

    gen_config = GenerationConfig(
        max_new_tokens=max_new,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        output_attentions=compute_deep_metrics,
        output_logits=compute_deep_metrics,
        return_dict_in_generate=True
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            past_key_values=cache_obj,
            attention_mask=mask,
            cache_position=pos,
            generation_config=gen_config
        )

    gen_text = tokenizer.decode(outputs.sequences[0][n_new:], skip_special_tokens=True).strip()
    logits = outputs.logits if compute_deep_metrics else None
    attentions = outputs.attentions if compute_deep_metrics else None

    return gen_text, logits, attentions