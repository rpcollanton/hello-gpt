import torch


class Trainer:
    
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.data = dataset