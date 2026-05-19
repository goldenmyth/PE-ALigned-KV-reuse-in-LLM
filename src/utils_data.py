import re

def get_data_for_dataset(ds_name, sample):
    if ds_name == "musique":
        system_msg = (
        "You are a precise question-answering assistant. "
        "Answer the question using the provided context with a short phrase (1-5 words). "
        "Do not use Markdown, do not provide links, do not use full sentences. "
        "Provide only the factual answer."
        )
        paragraphs = [p['paragraph_text'] for p in sample['paragraphs'] if p['is_supporting']]
        prefix_text = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\nContext:"
        segments = [f"\n{p}" for p in paragraphs]
        suffix_text = f"\n\nQuestion: {sample['question']}<|im_end|>\n<|im_start|>assistant\n"
        answer = sample['answer']

    elif ds_name == "samsum":
        sum_msg = "Summarize the dialogue in one short sentence."
        prefix_text = f"<|im_start|>system\n{sum_msg}<|im_end|>\n<|im_start|>user\nDialogue:"
        
        raw_dialogue = sample['dialogue']
        segments = [f"\n{s.strip()}" for s in sample['dialogue'].split('\n') if s.strip()]
        
        suffix_text = f"\n\nSummary:<|im_end|>\n<|im_start|>assistant\n"
        answer = sample['summary']
    
    elif ds_name == "ruler":
        raw_input = sample['input'].strip()
        
        lines = [l.strip() for l in raw_input.split('\n') if l.strip()]
        
        instruction = lines[0]
        question = lines[2]
        haystack = "\n".join(lines[1])

        system_msg = "You are a precise retrieval assistant. Answer with the MAGIC NUMBER only. DO NOT write sentences, DO NOT explain. Just the digits."
        prefix_text = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{instruction}\n\nContext:"
        
        sentences = re.split(r'(?<=[.!?])\s+', haystack)
        
        num_chunks = 10
        n = len(sentences)
        
        segments = []
        if n >= num_chunks:
            for i in range(num_chunks):
                start_idx = i * n // num_chunks
                end_idx = (i + 1) * n // num_chunks
                chunk_text = " ".join(sentences[start_idx:end_idx])
                if chunk_text.strip():
                    segments.append(f"\n{chunk_text.strip()}")
        else:
            segments = [f"\n{s.strip()}" for s in sentences if s.strip()]
            
        suffix_text = f"\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
        
        answer = str(sample['outputs'][0]) if sample['outputs'] else ""
        
    return prefix_text, segments, suffix_text, answer