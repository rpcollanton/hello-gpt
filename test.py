from hellogpt.model import GPT
import torch

mygpt = GPT(4,63,12,2,64)
test = torch.randint(0,63,(1,32))
mygpt(test)