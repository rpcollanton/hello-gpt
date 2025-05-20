# hello-gpt

A purely pedagogical and exploratory re-implementation of the [GPT model](https://github.com/openai/gpt-2), including a short demonstration of how to use it. As much from scratch as possible.

### TO DO
- start small projects -- shakespeare, names, math, arxiv, see [here](https://github.com/niderhoff/nlp-datasets) for more datasets
- clean up generation interface, write clean examples in ipynb
- write byte pair encoder, other encoders/tokenizers (see [karpathy videos](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), [mingpt](https://github.com/karpathy/minGPT.git), and [OpenAI encoder](https://github.com/openai/gpt-2/blob/master/src/encoder.py))
- write [RoPE](https://arxiv.org/pdf/2104.09864), [sinusoidal](https://arxiv.org/pdf/1706.03762)? 
- seek inspiration from deepseek -- what do they do differently? see [here](https://arxiv.org/pdf/2501.12948), and [here](https://arxiv.org/pdf/2412.19437)
- try to load OpenAI parameters... would have to reorganize!
- try an example of fine-tuning starting from OpenAI/HF parameters