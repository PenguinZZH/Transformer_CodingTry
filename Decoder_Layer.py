import torch
import torch.nn as nn

from MultiHead_Attention import MultiHeadAttention
from FeedForward_Network import FFN


# 解码器实现: 掩码多头自注意力子层 + 残差_归一化 + 编码器_解码器注意力层 + 残差_归一化 + FFN + 残差_归一化
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)   # 掩码多头自注意力子
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)   # 编码器-解码器注意力
        self.ffn = FFN(d_model=d_model, d_ff=d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. 掩码多头自注意力子层 + 残差_归一化
        self_attn_output, _ = self.masked_self_attention(x,x,x,tgt_mask)
        self_attn_output = self.dropout(self_attn_output)
        x = self.norm1(x + self_attn_output)

        # 2. 编码器_解码器注意力层 + 残差_归一化
        enc_dec_attn_output, _ = self.enc_dec_attention(x, encoder_output, encoder_output, src_mask)
        enc_dec_attn_output = self.dropout(enc_dec_attn_output)
        x = self.norm2(x + enc_dec_attn_output)

        # 3. FFN + 残差_归一化
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.norm3(x + ffn_output)

        return x
    

if __name__ == "__main__":
    # 我应该尝试从input_sequence开始创建
    batch_size = 5
    d_model = 512
    max_len = 100
    n_heads = 8
    dropout = 0.1
    d_ff = 2048

    # 编码器输出: 解码器输入
    encoder_output = torch.randn(batch_size, max_len, d_model)
    decoder_input = torch.randn(batch_size, max_len, d_model)

    # 初始化Decoder_layer
    decoder_layer = DecoderLayer(d_model=d_model, num_heads=n_heads, d_ff=d_ff)

    # 生成掩码(tat_mask: 下三角掩码?)
    tgt_mask = torch.tril(torch.ones(max_len, max_len), diagonal=0).unsqueeze(0).bool()     # (1, max_len, max_len); torch.tril(): 只保留下三角部分,上三角和对角线为0, 即被遮掩。 
    src_mask = None

    # 解码器层计算
    decoder_output = decoder_layer(decoder_input, encoder_output, src_mask, tgt_mask)

    print("解码器层输出形状：", decoder_output.shape)
    print("目标序列掩码（前5x5）：")
    print(tgt_mask[0, :5, :5])
    print(tgt_mask.shape)










