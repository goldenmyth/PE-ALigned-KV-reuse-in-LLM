def get_chat_parts(paragraphs, question):
    system_msg = "Answer accurately using not more than 3 words. Do not use full sentences."
    prefix_text = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\nContext:"
    p_texts = [f"\n{p}" for p in paragraphs]
    suffix_text = f"\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
    
    return prefix_text, p_texts, suffix_text