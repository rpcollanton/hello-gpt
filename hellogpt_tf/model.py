from .layers import Block, Linear
from .utilities import Config

import math
import tensorflow as tf
from tensorflow.python.keras import Model, Sequential, initializers
from tensorflow.keras.layers import LayerNormalization
from tensorflow.python.keras.layers import Dense, Embedding, Dropout

class GPT(Model):
    @staticmethod 
    def get_default_config() -> Config:
        cfg = Config(
            model_type = None,
            n_embd = None,
            n_layer = None, 
            n_head = None,
            vocab_size = None,
            block_size = None,
            p_drop_embd = 0.1,
            p_drop_attn = 0.1,
            p_drop_resid = 0.1
        )
        return cfg
    
    def __init__(self, cfg: Config):
        super().__init__()

        # hyperparameters
        self._process_config(cfg)
        self.cfg = cfg

        # define layers
        self.embd_tok   = Embedding(cfg.vocab_size, cfg.n_embd, embeddings_initializer=initializers.RandomNormal(std=0.02))
        self.embd_pos   = Embedding(cfg.vocab_size, cfg.n_embd, embeddings_initializer=initializers.RandomNormal(std=0.02))
        self.drop_embd  = Dropout(cfg.p_drop_embd)
        self.blocks     = Sequential(
            *[Block(cfg) for _ in range(cfg.n_layer)]
        )
        self.ln = LayerNormalization()
        self.lm_head = Dense(cfg.vocab_size, bias=False)

    @staticmethod
    def _process_config(cfg: Config):
        d = cfg.to_dict()
        keys = [
            "model_type",
            "n_embd", 
            "n_layer", 
            "n_head",
            "vocab_size", 
            "block_size", 
            "p_drop_embd", 
            "p_drop_attn", 
            "p_drop_resid"
        ]
        for k in keys:
            if k not in d:
                raise ValueError(f"Invalid Config object given to GPT: missing key {k}")
            
        # check if type is given and make sure that the model parameters were not also givne
        type_given = cfg.model_type is not None
        params_given = (cfg.n_embd is not None) or (cfg.n_layer is not None) or (cfg.n_head is not None)
        assert type_given ^ params_given, "Must only specify a model_type or model parameters, not both."
        
        if type_given:
            cfg.merge_from_dict({
                    # names follow the huggingface naming conventions
                    # GPT-1
                    'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                    # GPT-2 configs
                    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                    # Gophers
                    'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                    # (there are a number more...)
                    # Andrej made these tiny models up
                    'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                    'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                    'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[cfg.model_type])

        return 
