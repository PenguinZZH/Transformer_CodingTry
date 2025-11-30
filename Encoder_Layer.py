import torch
import torch.nn as nn

from MultiHead_Attention import MultiHeadAttention
from FeedForward_Network import FFN


# Encoder实现
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff)
        self.norm1 = nn.LayerNorm(d_model)     # 这是归一化层, 不是线性层; 归一化:对输入张量的最后一个维度(大小为d_mdeol)变成均值为0,方差为1的分布;
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Multi Head Attention + 残差 + Norm
        attention_output, _ = self.self_attention(x, x, x, mask)
        attention_output = self.dropout(attention_output)
        x = self.norm1(x + attention_output)    # 残差 + Norm

        # 2. FFN + 残差 + Norm
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.norm2(x + ffn_output)

        return x
    

if __name__ == "__main__":
    batch_size = 5
    d_model = 512
    max_len = 100
    n_heads = 8
    d_ff = 2048

    input_sequence = torch.randn(batch_size, max_len, d_model)
    encoder_layer = EncoderLayer(d_model=d_model, num_heads=n_heads, d_ff=d_ff)
    encoder_output = encoder_layer(input_sequence, mask=None)

    print("编码器层输出形状：", encoder_output.shape)   # (batch_size, max_len, d_model)




























