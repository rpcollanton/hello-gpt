import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Skip(nn.Module):
    """ Implements a skip connection by evaluating a layer and projecting it back into the residual pathway. """

    def __init__(self, layer: nn.Module, n_in: int, n_out: int):
        super().__init__()
        self.layer = layer
        self.proj = nn.Linear(n_in, n_out)
    
    def forward(self, x: torch.Tensor):
        return x + self.proj(self.layer(x))

class FeedForward(nn.Module):
    """ A single feed-forward layer followed by an activation function (GELU, here)."""
    
    def __init__(self, n_in, n_out):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)

    def forward(self, x: torch.Tensor):
        return nn.GELU(self.lin(x))
   
class MultiHeadAttention(nn.Module):
    """ Implements vectorized causal multi-head self-attention. """

    def __init__(self, n_embd, n_head, block_size):
        assert n_embd % n_head == 0

        # parameters
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head

        # layers
        self.representation = nn.Linear(n_embd, 3*n_embd)

        # causal mask -- attention only goes to the left 
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size() # batch, context/time, embedding/features (n_embd)
        assert C == self.n_embd

        # self.representation has shape (B, T, 3*C)
        # q, k, v have shapes (B, T, C)
        q, k, v = torch.split(self.representation(x), 3, dim=-1)

        # reshape in this way to get (batch, heads, context/time, head_size)
        # don't do view(B, nh, T, C//nh) directly because this will split up the context dimension in a weird way!
        q.view(B, T, self.n_head, self.head_size).transpose(1,2)
        k.view(B, T, self.n_head, self.head_size).transpose(1,2)
        v.view(B, T, self.n_head, self.head_size).transpose(1,2)

        # compute scaled query-key similarity
        # scaled to number of elements from k being collapsed into single element of similarity
        # (B, nh, T, hs) x (B, nh, hs, T) = (B, nh, T, T)
        similarity = q@k.transpose(-1,-2) / math.sqrt(k.size(-1))

        # mask it to ensure information only flows forward (attention is only given to past tokens)
        # broadcast across B/nh dimensions
        similarity = similarity.masked_fill(self.mask[:,:,:T,:T], float('-inf'))

        # apply soft max to convert to a set of weights on past tokens (with weights on future tokens equal to zero, thanks to the mask)
        weights = F.softmax(similarity, dim=-1)

        # apply attention weights to values, (B,nh,T,T) x (B,nh,T, hs) => (B, nh, T, hs)
        out = weights @ v
        
        # concatenate output of all heads back into (B,T,C)
        out = out.transpose(1,2).view(B, T, C)

        return out

class Block(nn.Module):
    """ Single transformer layer, following the architecture described in Radford et al., 2018. """
    
    def __init__(self, n_embd: int, n_head: int):
        assert n_embd % n_head == 0
        head_size = n_embd // n_head

        # Layers, with the appropriate skip/residual connections
        self.attn = Skip(MultiHeadAttention(n_embd, n_head), n_embd, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ffwd = Skip(FeedForward(n_embd, n_embd*4), n_embd*4, n_embd), 
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor):
        x = self.attn(x)
        x = self.ln1(x)
        x = self.ffwd(x)
        x = self.ln2(x)
        return x

