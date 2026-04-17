import yaml
import torch
import os

class Config:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            c = yaml.safe_load(f)
        
        # Model
        self.MODEL_NAME = c['model']['name']
        self.ATTN_IMPL = c['model']['attn_implementation']
        self.LOAD_4BIT = c['model']['load_in_4bit']
        
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        self.DTYPE = dtype_map.get(c['model']['dtype'], torch.float16)

        # Experiment
        self.SEED = c['experiment']['seed']
        self.NUM_SAMPLES = c['experiment']['num_samples_musique']
        self.SCALING_DOCS = c['experiment']['scaling_doc_counts']
        self.MAX_NEW_QA = c['experiment']['max_new_tokens_musique']
        self.MAX_NEW_SCALING = c['experiment']['max_new_tokens_scaling']

        # Dataset
        self.DATASET_NAME = c['dataset']['name']
        self.DATASET_SUBSET = c['dataset']['subset']

        # Paths
        self.SAVE_DIR = c['paths']['save_dir']
        self.CACHE_DIR = os.path.expanduser(c['paths']['cache_dir'])
        
        os.makedirs(self.SAVE_DIR, exist_ok=True)

config = Config()