import hellogpt_tf.layers as l
from hellogpt.utilities import Config

cfg = Config()
cfg.block_size = 32
cfg.n_embd = 128
cfg.n_head = 4
cfg.p_drop_attn = 0.2

l.MultiHeadAttention(cfg)