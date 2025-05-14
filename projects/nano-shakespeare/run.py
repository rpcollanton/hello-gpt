import sys
import os
sys.path.append(os.path.abspath(os.getcwd() + "/../../"))

import torch
from hellogpt.model import GPT
from hellogpt.trainer import Trainer
from hellogpt.utilities import Config
from hellogpt.data import CharDataset

def cb_print_iter(t: Trainer):
    if t.iter % 10 == 0:
        print(f"Iteration: {t.iter},   Loss = {t.loss.data}")

def cb_generate(t: Trainer):
    if t.iter % 50 == 0:
        t.model.eval()
        with torch.no_grad():
            prompt = "My oh my, what do we have here? Could it be?"
            x = torch.tensor(t.dataset.encode(prompt))
            y = model.generate(x, 500).view(-1)
            response = t.dataset.decode(y)
            print(response)
        t.model.train()

def get_config():
    cfg = Config()
    cfg.model = GPT.get_default_config()
    cfg.model.model_type = "gpt-nano"

    cfg.trainer = Trainer.get_default_config()
    cfg.trainer.learning_rate = 1E-3

    cfg.data = CharDataset.get_default_config()
    cfg.data.block_size = 128

    return cfg

cfg = get_config()
checkpoint_file = "model.pt"

# load dataset
with open('input.txt', 'r') as f:
    text = f.read()
dataset = CharDataset(cfg.data, text)

# build model
cfg.model.block_size = dataset.get_block_size()
cfg.model.vocab_size = dataset.get_vocab_size()
model = GPT(cfg.model)
if os.path.exists(checkpoint_file):
    model.load(checkpoint_file)

# build trainer
cfg.trainer.max_iter = 500
trainer = Trainer(cfg.trainer, model, dataset)
trainer.add_callback("on_batch_end", cb_print_iter)
trainer.add_callback("on_batch_end", cb_generate)

# run trainer
trainer.run()

# save model
model.save("gpt-nano-shakespeare-checkpoint.pt")