import sys
import os
sys.path.append(os.path.abspath(os.getcwd() + "/../../"))

from hellogpt.tokenizer import BasicBPETokenizer

# load dataset
with open('input.txt', 'r') as f:
    text = f.read()

tokenizer = BasicBPETokenizer()
tokenizer.train(text, 258, verbose=True)

print(tokenizer.decode(tokenizer.encode(text)) == text)