import math
import torch
import torch.nn as nn

# 位置编码的实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 初始化位置编码矩阵(max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 生成位置索引(max_len, 1):: 后续通过获取正弦/余弦函数时 会通过广播扩散1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算频率缩放因子
        ## torch.log(10000): ln(10000);  torch.arange()默认生成整数,后续torch.log()和torch.exp()运算浮点数, 因此需要强制.float()转换。
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.log(torch.tensor(10000.0)) / d_model)     # 维度: (2/d_model) 

        # 偶数维度用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)    # (max_len, 2/d_model)

        # 奇数维度用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加batch维度(1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 注册为缓冲区(不参与梯度更新)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # 嵌入向量 + 位置编码
        x = x + self.pe[:, :x.size(1)]
        return x


if __name__ == "__main__":
    d_model = 512   # 嵌入向量维度
    max_len = 100   # 序列最大长度

    # 初始化位置编码器
    pos_encoder = PositionalEncoding(d_model, max_len)

    # 示例输入序列(batch_size=5, seq_len=100, d_model=512)
    input_sequence = torch.randn(5, max_len, d_model)

    # 应用位置编码
    output = pos_encoder(input_sequence)


    print("位置编码后输出形状: ", output.shape)
    print("前2个位置的编码前5维: ", pos_encoder)




































































