from .layers import Block
from .utilities import Config

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class GPT(nn.Module):
    """ A small GPT model that is focused on demonstrating the mathematical structure of the model, rather than absolute speed or flexibility. """

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
    
    @classmethod
    def from_pretrained(cls, model_type: str):
        # only allow model types for which parameters are available
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        # create a GPT model 
        cfg = cls.get_default_config()
        cfg.model_type = model_type
        cfg.vocab_size = 50257 # OpenAI vocabulary size
        cfg.block_size = 1024  # OpenAI context/block size

        model = GPT(cfg)
        sd = model.state_dict()
        
        # load a Hugging Face model
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # transfer parameters from hugging face model to our GPT!
        # uh oh -- are they named and organized right to support this?
        # they are not!! we are going to have to reorganize to match the huggingface/openAI organization :')

    def __init__(self, cfg: Config):
        super().__init__()

        # hyperparameters
        self._process_config(cfg)
        self.cfg = cfg

        # define layers
        self.transformer = nn.ModuleDict(dict(
            embd_tok = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            embd_pos = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            drop_embd = nn.Dropout(cfg.p_drop_embd),
            blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self._initialize_weights()
        print(f"Initialized GPT model with {self.numparam()/1e3:0.2f}K parameters")
  
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

    @staticmethod
    def _init_layer(m: nn.Module):
        """ Initialization, based on the GPT-2 paper -- Radford et al. (2018). """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _initialize_weights(self):
        # apply to all modules and their submodules
        self.apply(self._init_layer)

        # search for the resid_proj 
        for name, param in self.named_parameters():
            if "resid_proj" in name:
                nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * self.cfg.n_layer))

    def numparam(self):
        n = sum(p.numel() for p in self.transformer.parameters())
        n += sum(p.numel() for p in self.lm_head.parameters())
        return n
    
    def forward(self, idx: torch.Tensor, tgt: torch.Tensor = None):
        """ Computes the predicted next token for each of the T tokens in each of the B batches. Attention internal to each batch, and causal (backwards-looking only). """
        B, T = idx.size()
        
        assert T <= self.cfg.block_size, f"Can only support context lengths up to block_size={self.cfg.block_size}, but got a context length of {T}"
        if tgt is not None:
            assert tgt.numel() == B, f"Target size {tgt.numel()} does not match number of inputted batches {B}."

        # position and token embedding, outputting a (B, T, n_embd) tensor
        pos = torch.arange(0,T,dtype=torch.long).view(1,T)
        te = self.transformer.embd_tok(idx)
        pe = self.transformer.embd_pos(pos)
        x = self.transformer.drop_embd(te + pe)

        # pass through transfomer blocks and normalize, outputting a (B, T, n_embd) tensor
        x = self.transformer.blocks(x)
        x = self.transformer.ln(x)

        # project back into the space of the vocabulary, "Language Modelling Head"
        logits = self.lm_head(x)

        # if we are given desired targets, compute a loss
        loss = None
        if tgt is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), tgt.view(-1))
        
        return logits, loss
    
    def generate(self, sequence, max_new_tokens):
        """
        Accept a sequence of size (B, T). T is the length of the embedded sequence, and B is the number of samples to generate.
        Generate a new sequence of length max_new_tokens by progressively sampling probabilities.
        """
        if sequence.dim == 1:
            sequence = sequence.view(1,-1)
        elif sequence.dim > 2: 
            raise ValueError(f"Inputted sequence tensor has unsupported number of dimensions {sequence.dim}.")
        
        for _ in range(max_new_tokens):
            # truncate sequence to block_size
            seq_cond = sequence if sequence.size(1) <= self.cfg.block_size else sequence[:, -self.cfg.block_size:]
            
            # forward through the model and generate probabilities for the next token 
            logits, _ = self(seq_cond)
            logits = logits[:,-1,:] # (All batches, last token, all characters)
            probs = F.softmax(logits,dim=-1)

            # sample probability to get next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # append this to sequence and move on
            sequence = torch.cat((sequence, idx_next), dim=1)
        
        return sequence
