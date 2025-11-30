import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, f"Error: d_model({d_model})必须能被num_heads({num_heads})整除"
        self.depth = d_model//num_heads

        # Q,K,V 的线性投影层
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        # 输出线性投影层
        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # 分割头部: (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, depth)
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. 线性投影: Q K V
        query = self.query_linear(query)    # (b, s, d_model)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # 2. 分割头部
        query = self.split_heads(query)     # (b, n_heads, s, depth)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # 3. 缩放点积注意力-注意力得分
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.depth)    # (b, n_heads, s, s); torch.sqrt(x) 参数x必须是tensor变量, 建议用math.sqrt()
        # 应用掩码(防止关注未来元素或填充元素)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)    # mask矩阵元素=0的地方, 标记为未来token, 填充为负无穷;
        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        # 加权求和
        attention_output = torch.matmul(attention_weights, value)   # (b, n_heads, s, depth)

        # 4. 合并头部
        ## 这里的.contiguous(): .view()是个视图类操作，要求内存的数据是连续的，物理存储顺序必须和逻辑索引顺序一致。而transpose实际上只修改张量的元数据, 虽然张量形式变了，但内存数据的顺序还是原来的顺序。因此需要.contiguous强制执行一次深拷贝, 开辟一块新内存, 按照逻辑上的顺序将数据一个个搬运过去。
        ## 如果不想用.contiguous(), 可以用.reshape()替换.view(); 这里用.view()通常是显式告诉开发者这里存在一次内存整理(拷贝).
        attention_output = attention_output.transpose(1, 2).contiguous()        # (b, s, n_heads, depth)
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # 5. 线性投影
        output = self.output_linear(attention_output)

        return output, attention_weights
    

if __name__ == "__main__":
    batch_size = 5
    d_model = 512   # 隐藏层维度
    num_heads = 8
    max_len = 100   # seq_len: 序列长度

    input_sequence = torch.randn(batch_size, max_len, d_model)
    
    # 初始化多头注意力
    multihead_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    output, attention_weights = multihead_attn(input_sequence, input_sequence, input_sequence)



    



































