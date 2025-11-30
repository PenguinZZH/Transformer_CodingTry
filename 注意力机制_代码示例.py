import torch
import torch.nn.functional as F

# 示例输入序列(batch_size=2, seq_len=2, embedding_dim=3)
input_sequence = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.9]])   # 这里是[2,3], batch_szie被省略了

print(input_sequence.size())

# 生成Q K V矩阵的随机权重
random_weights_query = torch.randn(input_sequence.size(-1), input_sequence.size(-1))
random_weights_key = torch.randn(input_sequence.size(-1), input_sequence.size(-1))
random_weights_value = torch.randn(input_sequence.size(-1), input_sequence.size(-1))

# q k v
query = torch.matmul(input_sequence, random_weights_query)
key = torch.matmul(input_sequence, random_weights_key)
value = torch.matmul(input_sequence, random_weights_value)

# 计算注意力得分(缩放点积)
attention_scores = torch.matmul(query, key.T) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))

# 计算softmax, 获得注意力权重
attention_weights = F.softmax(attention_scores, dim=-1)

# 计算Value向量的加权和
output = torch.matmul(attention_weights, value)


print("自注意力机制后的输出: ", output)
print("注意力权重(每个元素对其他元素的关注程度): ", attention_weights)














