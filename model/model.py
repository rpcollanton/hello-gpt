from layers import Block

import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(self, n_layer, vocab_size, n_embd, n_head, block_size):
        super().__init__()

        # parameters
        self.block_size = block_size

        # layers
        self.transformer = nn.ModuleDict(dict(
            embd_tok = nn.Embedding(vocab_size, n_embd),
            embd_pos = nn.Embedding(vocab_size, n_embd),
            blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in n_layer]),
            ln = nn.LayerNorm(n_embd)
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, tgt: torch.Tensor = None):
        B, T = idx.size()
        
        assert T <= self.block_size, f"Can only support context lengths up to block_size={self.block_size}, but got a context length of {T}"
        if tgt is not None:
            assert tgt.numel() == B, f"Target size {tgt.numel()} does not match number of inputted batches {B}."

        # position and token embedding, outputting a (B, T, n_embd) tensor
        pos = torch.arange(0,T,dtype=torch.long).view(1,T)
        te = self.transformer.embd_tok(idx)
        pe = self.transformer.embd_pos(pos)
        x = te + pe

        # pass through transfomer blocks and normalize, outputting a (B, T, n_embd) tensor
        x = self.transformer.blocks(x)
        x = self.transformer.ln(x)

        # project back into the space of the vocabulary, "Language Modelling Head"
        logits = self.lm_head(x)

        # if we are given desired targets, compute a loss
        loss = None
        if tgt is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), tgt.view(-1))
        
        return logits, loss
    