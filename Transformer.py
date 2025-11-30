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

        # 位置编码层
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        # 编码器堆叠(num_layers)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.1) for _ in range(num_layers)])
        # 解码器堆叠(num_layers)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.1) for _ in range(num_layers)])
        # 输出线性层
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """生成输入掩码和目标掩码"""
        # 输入掩码: 屏蔽填充元素(src==0 为填充)
        ## (ser != 0): 将src矩阵变成相同形状、不为0的地方变成True, 为0的地方变成False
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)     # (batch_size, 1, 1, src_len)
        # 目标掩码: 屏蔽填充元素和未来元素
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)     # (batch_size, 1, tgt_len, 1)
        seq_length = tgt.size(1)
        # 下三角掩码
        # triu: Triangle Upper(上三角); 这里使用之前DecoderLayer部分的torch.tril效果是一样的.
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask  # &: AND运算; 就是tgt_mask和nopeak_mask必须同时为真，才为真;

        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        """
        Inputs:
            src: 编码器的初始输入-Token_id; (batch_size, sec_len), 这里sec_len是当前序列长度
            tgt: 解码器的初始输入-Token_id; (batch_size, sec_len)
        """
        # 生成掩码
        src_mask, tgt_mask = self.generate_mask(src=src, tgt=tgt)

        # 1. 编码器处理
        # 词嵌入 + 位置编码 + Dropout
        enc_emb = self.encoder_embedding(src)   # (batch_size, src_len, d_model); 词嵌入层, 将输入数据token_id扩展d_model维度
        enc_pos =  self.positional_encoding(enc_emb)
        src_embedded = self.dropout(enc_pos)
        # 前向传播
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)    # (batch_size, src_len, d_model)

        # 2. 解码器处理
        # 词嵌入 + 位置编码 + Dropout
        dec_emb = self.decoder_embedding(tgt)   # (batch_size, src_len, d_model)
        dec_pos = self.positional_encoding(dec_emb)
        tgt_embedded = self.dropout(dec_pos)
        # 前向传播
        dec_input = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_input, enc_output, src_mask, tgt_mask)

        # 3. 输出线性投影
        output = self.linear(dec_output)    # (batch_size, tgt_len, tgt_vocab_size)
        return output
    

if __name__ == "__main__":
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_len = 100
    dropout = 0.1

    # 初始化Transformer模型
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)

    # 生成随机示例数据(batch_size=5, seq_len=100)
    ## torch.randint(low, high, size): 生成low-high间的随机整数， 形状为size的张量
    Current_Seq_Len=50
    src_data = torch.randint(1, src_vocab_size, (5, Current_Seq_Len))  # 输入序列（避免0，0为填充）
    tgt_data = torch.randint(1, tgt_vocab_size, (5, Current_Seq_Len+10))  # 目标序列

    # 模型前向传播（目标序列右移一位作为输入）
    output = transformer(src_data, tgt_data[:, :-1])    # [:, :-1] 去掉每一行的最后一个元素
    print(f"src_data.shape: {src_data.shape}")  # torch.Size([5, 50])
    print(f"tgt_data[:, :-1].shape: {tgt_data[:, :-1].shape}")  # torch.Size([5, 59])

    print("模型输出形状：", output.shape)  # (batch_size, tgt_len-1, tgt_vocab_size)    torch.Size([5, 59, 5000])
    print("模型参数量：", sum(p.numel() for p in transformer.parameters()) / 1e6, "M")  # 51.823496 M













































