import yaml
import torch
import os

class Config:
    def __init__(self, path="config.yaml"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at {path}")

        with open(path, "r") as f:
            c = yaml.safe_load(f)

        # --- Model Configuration ---
        model_cfg = c.get('model', {})
        self.MODEL_NAME = model_cfg.get('name', "Qwen/Qwen2.5-3B-Instruct")
        self.ATTN_IMPL = model_cfg.get('attn_implementation', "eager")
        self.LOAD_4BIT = model_cfg.get('load_in_4bit', False)
        self.DEVICE = model_cfg.get('device', "cuda" if torch.cuda.is_available() else "cpu")
        
        dtype_map = {
            "fp16": torch.float16, 
            "bf16": torch.bfloat16, 
            "fp32": torch.float32
        }
        self.DTYPE = dtype_map.get(model_cfg.get('dtype'), torch.float16)

        # --- Global Configuration ---
        global_cfg = c.get('global', {})
        self.SEED = global_cfg.get('seed', 42)
        self.STRATEGIES = global_cfg.get('strategies', ["Aligned", "Naive"])

        # --- Datasets Configuration ---
        self.DATASETS = c.get('datasets', {})

        # --- Paths & Directories ---
        paths_cfg = c.get('paths', {})
        self.SAVE_DIR = paths_cfg.get('save_dir', "./results")
        self.CACHE_DIR = os.path.expanduser(paths_cfg.get('cache_dir', "./.cache/huggingface"))

        os.makedirs(self.SAVE_DIR, exist_ok=True)

    def get_enabled_datasets(self):
        return {name: cfg for name, cfg in self.DATASETS.items() if cfg.get('enabled', False)}

config = Config()