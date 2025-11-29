import torch
import torch.nn.functional as F

# 示例输入序列(batch_size=2, seq_len=2, embedding_dim=3)
input_sequence = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.9]])   # 

print(input_sequence.size())

# 生成Q K V矩阵的随机权重
random_weights_query = torch.randn(input_sequence.size(-1), input_sequence.size(-1))
random_weights_key = torch.randn(input_sequence.size(-1), input_sequence.size(-1))
random_weights_value = torch.randn(input_sequence.size(-1), input_sequence.size(-1))

query = torch.matmul(input_sequence, random_weights_query)
key = torch.matmul(input_sequence, random_weights_key)
value = torch.matmul(input_sequence, random_weights_value)
















