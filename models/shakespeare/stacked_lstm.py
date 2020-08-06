import torch
from torch import nn


class Model(nn.Module):

    def __init__(self, seq_len, num_classes, num_hidden):
        super(Model, self).__init__()
        self.input_shape = [80]
        self.input_type = 'index'  # 输入是 index
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        # 用于将文本数据转换为对应的词向量, sent 的实验中使用训练好的 glove 词向量
        # emv 有一个记录权重的矩阵 [num_vocabulary, 8]. num_vocabulary 是句子中的字符的数量
        # 输入 [*], 索引, 输出 [*, embedding_dim]
        # TODO sparse 参数可以影响梯度的计算, 请注意
        self.embedding_layer = nn.Embedding(num_embeddings=seq_len, embedding_dim=8, sparse=False)
        torch.nn.init.xavier_uniform(self.embedding_layer.weight)
        # 输入: (seq_len, batch, input_size), hx(2,batch,hidden_size)
        # 输出: (seq_len, batch, num_directions * hidden_size), 如果 batch_first == True, 交换 0, 1
        self.stacked_lstm = nn.LSTM(input_size=8,
                                    hidden_size=num_hidden,
                                    num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=self.num_hidden, out_features=num_classes)

    def forward(self, inputs):
        x = inputs
        x = self.embedding_layer(x)
        # 将 embedding 前任嵌入的数据转换, 这里不传入 hidden, LSTM 自动处理
        x, _ = self.stacked_lstm(x)
        # 预测是那个人物, 用最后一句话?
        x = x[:, -1, :]
        x = self.fc(x)
        return x