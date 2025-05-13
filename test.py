from hellogpt.model import GPT
from hellogpt.utilities import Config
import torch

cfg = GPT.get_default_config()
cfg.vocab_size = 256
cfg.block_size = 64
cfg.n_embd = 128
cfg.n_layer = 12
cfg.n_head = 8
# cfg.model_type = 'openai-gpt'

mygpt = GPT(cfg)
test = torch.randint(0,63,(1,32))
logits, _ = mygpt(test)

print(logits.shape)