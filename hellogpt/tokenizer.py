from abc import ABC, abstractmethod
from copy import copy

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

class BasicBPETokenizer(Tokenizer):

    def __init__(self):
        self.vocab_size = None
        self.merges = {}
        self.vocab = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, "Desired vocab size can not be less than the number of 8-bit integers."
        n_merges = vocab_size - 256

        tokens = text.encode("utf-8")
        tokens = list(map(int,tokens))
        _, bp_merges, bp_vocab = BasicBPETokenizer._bpe(tokens,n_merges)
        
        print(bp_merges)
        def expand(idx: int):
            if idx in self.vocab:
                return self.vocab[idx]
            elif idx in bp_vocab:
                return expand(bp_vocab[idx][0]) + expand(bp_vocab[idx][1])
            else:
                return bytes([idx])
        
        self.vocab = {}
        self.vocab_size = vocab_size
        self.merges = bp_merges
        self.vocab.update({idx: bytes([idx]) for idx in range(256)})
        self.vocab.update({idx: expand(idx) for idx in bp_vocab.keys()})
        
        if verbose:
            print(f"Constructed vocabulary of length: {len(self.vocab)}")
            for k,v in self.vocab.items():
                print(f"{k}: {v.decode('utf-8', errors='replace')}")
    
    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            counts = BasicBPETokenizer._get_pair_counts(tokens)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = BasicBPETokenizer._merge(tokens, pair, self.merges[pair])
        return tokens

    def decode(self, tokens):
        text = b"".join(self.vocab[idx] for idx in tokens)
        text = text.decode("utf-8", errors="replace")
        return text
    
    @staticmethod
    def _bpe(tokens, n_merges: int):
        out = copy(tokens)
        new_id = max(max(out), 256)
        bp_merges = {}
        bp_vocab = {}
        i = 0
        while i < n_merges:
            counts = BasicBPETokenizer._get_pair_counts(out)
            pair = max(counts, key=counts.get)
            out = BasicBPETokenizer._merge(out, pair, new_id)
            bp_merges[pair] = new_id
            bp_vocab[new_id] = pair
            i+=1
            new_id+=1
        return out, bp_merges, bp_vocab


    @staticmethod
    def _merge(tokens, merge_pair, new_id):
        out = copy(tokens)
        idx = 0
        while (idx+1) < len(out):
            if (out[idx],out[idx+1]) == merge_pair:
                out.pop(idx)
                out.pop(idx)
                out.insert(idx, new_id)
            idx+=1
        return out
    
    @staticmethod
    def _get_pair_counts(tokens):
        counts = {}
        for pair in zip(tokens[:-1],tokens[1:]):
            counts[pair] = counts.get(pair,0) + 1
        return counts