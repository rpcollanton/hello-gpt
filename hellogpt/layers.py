import math

from .utilities import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

class Skip(nn.Module):
    """ Implements a skip connection by evaluating a layer and projecting it back into the residual pathway. """

    def __init__(self, layer: nn.Module, n_in: int, n_out: int, p_drop: float):
        super().__init__()

        self.layer = layer
        self.resid_proj = nn.Linear(n_in, n_out)
        self.drop_resid = nn.Dropout(p_drop)
    
    def forward(self, x: torch.Tensor):
        return x + self.drop_resid(self.resid_proj(self.layer(x)))

class FeedForward(nn.Module):
    """ A single feed-forward layer followed by an activation function (GELU, here)."""
    
    def __init__(self, n_in, n_out):
        super().__init__()

        self.lin = nn.Linear(n_in, n_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return self.act(self.lin(x))
   
class MultiHeadAttention(nn.Module):
    """ Implements vectorized causal multi-head self-attention. """

    def __init__(self, cfg: Config):
        super().__init__()

        assert cfg.n_embd % cfg.n_head == 0

        # parameters
        self.block_size     = cfg.block_size
        self.n_embd         = cfg.n_embd
        self.n_head         = cfg.n_head
        self.head_size      = cfg.n_embd // cfg.n_head

        # layers
        self.representation = nn.Linear(self.n_embd, 3*self.n_embd)
        self.drop_attn = nn.Dropout(cfg.p_drop_attn)

        # causal mask -- attention only goes to the left 
        self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size, dtype=bool)).view(1, 1, self.block_size, self.block_size))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size() # batch, context/time, embedding/features (n_embd)
        assert C == self.n_embd

        # self.representation has shape (B, T, 3*C)
        # q, k, v have shapes (B, T, C)
        q, k, v = torch.split(self.representation(x), self.n_embd, dim=-1)

        # reshape in this way to get (batch, heads, context/time, head_size)
        # don't do view(B, nh, T, C//nh) directly because this will split up the context dimension in a weird way!
        q = q.view(B, T, self.n_head, self.head_size).transpose(1,2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1,2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1,2)

        # compute scaled query-key similarity
        # scaled to number of elements from k being collapsed into single element of similarity
        # (B, nh, T, hs) x (B, nh, hs, T) = (B, nh, T, T)
        similarity = q@k.transpose(-1,-2) / math.sqrt(k.size(-1))

        # mask it to ensure information only flows forward (attention is only given to past tokens)
        # broadcast across B/nh dimensions
        similarity = similarity.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))

        # apply soft max to convert to a set of weights on past tokens (with weights on future tokens equal to zero, thanks to the mask)
        weights = F.softmax(similarity, dim=-1)

        # dropout during training to reduce overfitting
        weights = self.drop_attn(weights)

        # apply attention weights to values, (B,nh,T,T) x (B,nh,T, hs) => (B, nh, T, hs)
        out = weights @ v
        
        # concatenate output of all heads back into (B,T,C)
        out = out.transpose(1,2).contiguous().view(B, T, C)

        return out

class Block(nn.Module):
    """ Single transformer layer, following the architecture described in Radford et al., 2018. """
    
    def __init__(self, cfg: Config):
        super().__init__()

        # Layers, with the appropriate skip/residual connections
        self.attn   = Skip(MultiHeadAttention(cfg), cfg.n_embd, cfg.n_embd, cfg.p_drop_resid)
        self.ln1    = nn.LayerNorm(cfg.n_embd)
        self.ffwd   = Skip(FeedForward(cfg.n_embd, cfg.n_embd*4), cfg.n_embd*4, cfg.n_embd, cfg.p_drop_resid)
        self.ln2    = nn.LayerNorm(cfg.n_embd)

    def forward(self, x: torch.Tensor):
        x = self.attn(x)
        x = self.ln1(x)
        x = self.ffwd(x)
        x = self.ln2(x)
        return x

