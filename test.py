from hellogpt.model import GPT
from hellogpt.utilities import Config
import torch

cfg = Config(
    vocab_size = 63,
    n_embd = 12,
    n_layer = 4, 
    n_head = 2,
    block_size = 64,
    p_drop_embd = 0.15,
    p_drop_attn = 0.15,
    p_drop_resid = 0.15
)

mygpt = GPT(cfg)
test = torch.randint(0,63,(1,32))
logits, _ = mygpt(test)

print(logits.shape)