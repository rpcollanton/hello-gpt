from .layers import Block
from .utilities import Config

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class GPT(nn.Module):
    """ A small GPT model that is focused on demonstrating the mathematical structure of the model, rather than absolute speed or flexibility. """

    def __init__(self, cfg: Config):
        super().__init__()

        # hyperparameters
        self._validate_config(cfg)
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
    def _validate_config(cfg: Config):
        d = cfg.to_dict()
        keys = [
            "vocab_size", 
            "n_embd", 
            "n_layer", 
            "n_head", 
            "block_size", 
            "p_drop_embd", 
            "p_drop_attn", 
            "p_drop_resid"
        ]
        for k in keys:
            if k not in d.keys():
                raise ValueError(f"Invalid Config object given to GPT: missing key {k}")
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
