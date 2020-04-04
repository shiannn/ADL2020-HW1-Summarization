import torch.nn as nn
import torch.nn.functional as F
from seq2seq.EncoderRNN import hidden_size

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, pretrained_weight, batch_Size):
        super(DecoderRNN, self).__init__()
        self.batch_Size = batch_Size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        if (pretrained_weight is not 'none'):
            self.embedding.weight.data.copy_(pretrained_weight)
        # output_size 是 output language 的 vocab 數量
        # embedding 是 output lang 的 embedding
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        # 每次丟好幾個batch的第一個字進來 -1 表示他們都有一個詞向量
        output = self.embedding(input).view(1, self.batch_Size, -1)
        #output = self.embedding(input)
        #print('embedded.shape', output.shape)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch_Size, self.hidden_size, device=device)