import torch
from torch.utils.data import Dataset
from .utilities import Config

class CharDataset(Dataset):

    @staticmethod
    def get_default_config():
        cfg = Config()
        cfg.block_size = 128
        return cfg

    def __init__(self, cfg: Config, data: str):
        self.config: Config = cfg
        self.data = data

        chars = sorted(list(set(data)))
        self.stoi = {ch: i for i,ch in enumerate(chars)}
        self.itos = {i: ch for i,ch in enumerate(chars)}
        self.vocab_size = len(chars)
    
    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        assert idx < self.__len__(), "Out of bounds data access attempt!"
        text = self.data[idx:(idx+self.config.block_size)]
        encoded = self.encode(text)
        x = torch.tensor(encoded[:-1], dtype = torch.long)
        y = torch.tensor(encoded[1:],  dtype = torch.long)
        return x,y

    def encode(self, text):
        return [self.stoi[s] for s in text]
    
    def decode(self, indices):
        return ''.join([self.itos[int(i)] for i in indices])


    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        return self.config.block_size
