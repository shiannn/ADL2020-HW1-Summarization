import torch.nn as nn
import torch

hidden_size = 300
BATCH_SIZE = 32

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pretrained_weight):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        if (pretrained_weight is not 'none'):
            self.embedding.weight.data.copy_(pretrained_weight)
        #print(self.embedding)
        #print(self.embedding.weight)
        #dummyInput = torch.tensor([[0,1],[2,3]], dtype=torch.long)
        dummyInput = torch.tensor([0], dtype=torch.long)
        dummyEmbed = self.embedding(dummyInput)
        #print('dummyEmbed shape', dummyEmbed.shape)
        dummyHidden = torch.zeros(1, 1, self.hidden_size)
        #print('hidden shape', dummyHidden.shape)
        self.gru = nn.GRU(hidden_size, hidden_size)
        dummyGRUout, dummyHidden = self.gru(dummyEmbed.view(1,1,-1), dummyHidden)
        #print('dummyGRUout shape', dummyGRUout.shape)
        #print('dummyGRUHidden shape', dummyHidden.shape)

    def forward(self, input, hidden):
        # 此 view 為 view(seq字數(pad到多少), batch_Size(32), feature數(glove 300維))
        # 每次丟好幾個batch的第一個字進來 -1 表示他們都有一個詞向量
        embedded = self.embedding(input).view(1, BATCH_SIZE, -1)
        #print('input.shape', input.shape)
        #embedded = self.embedding(input)
        #embedded = self.embedding(input)
        #print('embedded.shape', embedded.shape)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # hidden (,, feature數(glove 300維))
        return torch.zeros(1, BATCH_SIZE, self.hidden_size, device=device)