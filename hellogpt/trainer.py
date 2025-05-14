from .utilities import Config 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict


class Trainer:

    @staticmethod
    def get_default_config():
        cfg = Config()
        cfg.max_iter = 10000
        cfg.learning_rate = 1E-3
        cfg.loss_threshold = None
        return cfg
    
    def __init__(self, cfg: Config, model: nn.Module, dataset: Dataset):
        self.config: Config     = cfg
        self.model: nn.Module   = model
        self.dataset: Dataset   = dataset
        self.optimizer          = None
        self.callbacks          = defaultdict(list)

        # configuration parameters
        self.max_iter: int = cfg.max_iter
        self.lr: float = cfg.learning_rate
        self.loss_threshold: float = cfg.loss_threshold if cfg.loss_threshold is not None else float('-inf')

        # logging variables
        self.iter: int = 0
        self.loss: torch.Tensor = None

    def set_callback(self, onevent: str, f: callable):
        self.callbacks[onevent] = [f]
    
    def add_callback(self, onevent: str, f: callable):
        self.callbacks[onevent].append(f)
    
    def trigger_callbacks(self, onevent: str):
        for f in self.callbacks.get(onevent, []):
            f(self)
    
    def run(self):
        
        # configure the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)

        # configure the dataloader
        dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)
        dataiterator = iter(dataloader)

        while True:
            # get next data sample
            try: 
                x,y = next(dataiterator)
            except StopIteration:
                dataiterator = iter(dataloader)
                x,y = next(dataiterator)
            
            # forward pass
            _, self.loss = self.model(x,y)

            # backward pass
            self.model.zero_grad()
            self.loss.backward()

            # update
            self.optimizer.step()
            self.iter += 1

            self.trigger_callbacks("on_batch_end")

            # break conditions
            if self.iter > self.max_iter:
                break
            if self.loss < self.loss_threshold:
                break