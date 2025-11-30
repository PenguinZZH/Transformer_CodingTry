import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    


if __name__ == "__main__":
    batch_size = 5
    d_model = 512
    d_ff = 2048
    max_len = 100

    input_sequence = torch.randn(batch_size, max_len, d_model)      # torch.randn():从标准正态分布中随机抽取; torch.rand():从标准分布中随机抽取; torch.random():从离散正态分布中抽取;

    ffn = FFN(d_model, d_ff)
    output = ffn(input_sequence)
    print("前馈网络输入形状：", input_sequence.shape)
    print("前馈网络输出形状：", output.shape)
    print("第一层线性变换后形状：", ffn.linear1(input_sequence).shape)























































