import torch
from torch.utils.data import Dataset
from .utilities import Config
from .tokenizer import CharTokenizer

class CharDataset(Dataset):

    @staticmethod
    def get_default_config():
        cfg = Config()
        cfg.block_size = 128
        return cfg

    def __init__(self, cfg: Config, data: str):
        self.config: Config = cfg
        self.data = data
        self.tokenizer = CharTokenizer()
        self.tokenizer.train(data)
    
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
        return self.tokenizer.encode(text)
    
    def decode(self, indices):
        return self.tokenizer.decode(indices)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size
    
    def get_block_size(self):
        return self.config.block_size
