from abc import ABC, abstractmethod

class Tokenizer(ABC):

    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def decode(self, tokens):
        pass

    @abstractmethod
    def train(self, text, vocab_size=None, verbose=False):
        pass

class CharTokenizer(Tokenizer):

    def __init__(self):
        self.vocab_size = None
        self.stoi = {}
        self.itos = {}

    def train(self, text, vocab_size=None, verbose=False):
        chars = sorted(list(set(text)))
        assert (vocab_size is None) or (vocab_size == len(chars)), "Vocab size can not be independently set in CharTokenizer."
        
        self.vocab_size = len(chars)
        self.stoi = {s: i for i,s in enumerate(chars)}
        self.itos = {i: s for i,s in enumerate(chars)}
        
        if verbose:
            print(f"Vocab constructed with {self.vocab_size} characters.")

    def encode(self, text):
        return [self.stoi[s] for s in text]
    
    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])
