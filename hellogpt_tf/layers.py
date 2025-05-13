from .utilities import Config
import math

import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.python.keras.activations import gelu

import numpy as np

class Skip(Model):
    def __init__(self, layer: Layer, n_in: int, n_out: int, p_drop: float):
        super().__init__()
        self.layer = layer
        self.resid_proj = Dense(n_out)
        self.drop_resid = Dropout(p_drop)
    
    def call(self, x: tf.Tensor):
        return x + self.drop_resid(self.resid_proj(x))

class MultiHeadAttention(Layer):
    """ Causal multi-head self-attention in vectorized form. """
    def __init__(self, cfg: Config):
        super().__init__()

        assert cfg.n_embd % cfg.n_head == 0

        # parameters
        self.block_size     = cfg.block_size
        self.n_embd         = cfg.n_embd
        self.n_head         = cfg.n_head
        self.head_size      = cfg.n_embd // cfg.n_head

        # layers
        self.representation = Dense(3 * self.n_embd)
        self.drop_attn = Dropout(cfg.p_drop_attn)

        # causal mask -- information flows from past to present only, not from future to present!
        self.mask = tf.convert_to_tensor(
            np.triu(np.ones((self.block_size,self.block_size),dtype=bool)).resize(1,1,self.block_size,self.block_size)
        )


    def call(self, x: tf.Tensor):
        # extract size (batch, time/context, features/embedding)
        B,T,C = tf.size(x)
        assert C == self.n_embd

        # call representation layer and split into query, key, value arrays
        # q,k,v each have shape (B,T,C)
        # q - what token within batch b at position t is looking for (feature set C)
        # k - what token within batch b at position t has to match to the q of other tokens within batch b (feature set C)
        # v - what token within batch b at position t offers when it is selected
        q, k, v = tf.split(self.representation(x), 3, axis=-1)

        # split into number of heads
        # (B, n_H, T, hs)
        q = tf.transpose(tf.reshape(q, (B, T, self.n_head, self.head_size)), [0,2,1,3])
        k = tf.transpose(tf.reshape(k, (B, T, self.n_head, self.head_size)), [0,2,1,3])
        v = tf.transpose(tf.reshape(v, (B, T, self.n_head, self.head_size)), [0,2,1,3])

        # compute scaled similarity
        # (B, n_h, T, T)
        similarity = tf.linalg.matmul(q, tf.transpose(k, [0,1,3,2])) / math.sqrt(tf.size(k)[-1])

        # mask it to ensure flow of information forwards only
        # (B, n_h, T, T)
        similarity = tf.fill(tf.where(self.mask[:,:,:T,:T], float('-inf')))

        # apply soft max to convert to a set of weights, with future "weights" set to zero
        # (B, n_h, T, T)
        weights = tf.nn.softmax(similarity, axis=-1)

        # drop out during training
        # (B, n_h, T, T)
        weights = self.drop_attn(weights)

        # apply weights to values
        # (B, n_h, T, hs)
        out = tf.linalg.matmul(weights, v)

        # reshape, concatenate output of each head
        out = tf.reshape(tf.transpose(out, [0,2,1,3]), (B,T,C))

        return out
        
    class Block(Layer):
        """ A single transformer block. """

        def __init__(self, cfg: Config):
            super().__init__()

            # Layers with residual connections
            self.attn   = Skip(MultiHeadAttention(cfg), cfg.n_embd, cfg.n_embd, cfg.p_drop_resid)
            self.ln1    = LayerNormalization()
            self.ffwd   = Skip(Dense(4*cfg.n_embd, activation=gelu), 4*cfg.n_embd, cfg.n_embd, cfg.p_drop_resid)
            self.ln2    = LayerNormalization()
        
        def call(self, x: tf.Tensor):
            x = self.attn(x)
            x = self.ln1(x)
            x = self.ffwd(x)
            x = self.ln2(x)
            return x



