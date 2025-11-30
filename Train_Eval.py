import torch
import torch.nn as nn
import torch.optim as optim

from Transformer import Transformer


# 模型初始化
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_len = 100
dropout = 0.1
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss(ignore_index=0)     # 交叉损失函数, 忽略填充元素
optimizer = optim.Adam(
    transformer.parameters(),
    lr=1e-4,
    betas=(0.9, 0.98),  # Adam优化器参数
    eps=1e-9             # 数值稳定性参数: 防止除零操作
)

# 生成训练数据
src_train = torch.randint(1, src_vocab_size, (5, max_len-10))
tgt_train = torch.randint(1, tgt_vocab_size, (5, max_len+10))   # (batch, tgt_len)

# 训练循环
epochs = 10 # 训练参数
transformer.train()     # 模型设置为训练模式
for epoch in range(epochs):
    optimizer.zero_grad()   # 清空梯度

    # 模型前向传播(目标序列右移一位作为输入)
    output = transformer(src_train, tgt_train[:, :-1])  # (batch_size, tgt_len-1, tgt_vocab_size) 解码后各个词的得分，还没有经过softmax(即未转化为概率)

    # 计算损失
    ## .contiguous(): 通过深度拷贝修复内存与索引的顺序
    loss = criterion(
        output.contiguous().view(-1, tgt_vocab_size),   # (batch*(tgt_len-1), vocab_size)
        tgt_train[:, 1:].contiguous().view()            # (batch*(tgt_len-1), )
    )

    # 反向传播和参数更新
    loss.backward()
    optimizer.step()

    # 计算困惑度
    perplexity = torch.exp(loss)

    # 打印训练信息
    print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {loss.item():.4f} | Perplexity: {perplexity.item():.2f}")


print("-"*50)
# 生成评估数据
src_test = torch.randint(1, src_vocab_size, (5, max_len))
tgt_test = torch.randint(1, tgt_vocab_size, (5, max_len))

# 模型评估
transformer.eval()  # 模型设置为评估模式
with torch.no_grad():
    test_output = transformer(src_test, tgt_test[:, :-1])
    test_loss = criterion(
        test_output.contiguous().view(-1, tgt_vocab_size),
        tgt_train[:, 1:].contiguous().view()
    )
    test_perplexity = torch.exp(test_loss)

print("\n评估结果：")
print(f"Test Loss: {test_loss.item():.4f} | Test Perplexity: {test_perplexity.item():.2f}")




























