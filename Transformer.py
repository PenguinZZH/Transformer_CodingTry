import torch
import torch.nn as nn

from PositionEncoding import PositionalEncoding
from MultiHead_Attention import MultiHeadAttention
from FeedForward_Network import FFN
from Encoder_Layer import EncoderLayer
from Decoder_Layer import DecoderLayer


# Transformer 实现
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)  # 输入序列嵌入
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)  # 输出序列嵌入






















































