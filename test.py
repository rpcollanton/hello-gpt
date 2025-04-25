from hellogpt.model import GPT
from hellogpt.utilities import Config
import torch

cfg = GPT.get_default_config()
cfg.vocab_size = 20000
cfg.block_size = 512
cfg.model_type = 'openai-gpt'

mygpt = GPT(cfg)
# test = torch.randint(0,63,(1,32))
# logits, _ = mygpt(test)

# print(logits.shape)