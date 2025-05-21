from hellogpt.model import GPT
import tiktoken
import torch

prompt = "Ryan Collanton, the most"

model = GPT.from_pretrained("gpt2")
enc = tiktoken.get_encoding("gpt2")
prompt = torch.tensor(enc.encode(prompt))

out = model.generate(prompt, 20)
y = enc.decode(out.cpu().squeeze().tolist())
print('-'*80)
print(y)