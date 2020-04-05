import torch.nn as nn
import torch

hidden_size = 300
directions = 2

class AttnEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pretrained_weight, batch_Size):
        super(AttnEncoderRNN, self).__init__()
        self.batch_Size = batch_Size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.num_directions = directions

        self.embedding = nn.Embedding(input_size, hidden_size)
        if (pretrained_weight is not 'none'):
            self.embedding.weight.data.copy_(pretrained_weight)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        # 此 view 為 view(seq字數(pad到多少), batch_Size(32), feature數(glove 300維))
        # 每次丟好幾個batch的第一個字進來 -1 表示他們都有一個詞向量
        #embedded = self.embedding(input).view(1, self.batch_Size, -1)
        embedded = self.embedding(input)
        #print('input.shape', input.shape)
        #embedded = self.embedding(input)
        #embedded = self.embedding(input)
        #print('embedded.shape', embedded.shape)
        output = embedded
        output, hidden = self.gru(output, hidden)
        ### output (seq_len, batch, direct* hidden_size)
        ### hidden (direct*layers, batch, hidden_size)
        return output, hidden

    def initHidden(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # hidden (,, feature數(glove 300維))
        # num_layers*num_directions, batch_size, hidden_size
        return torch.zeros(self.num_layers*self.num_directions, self.batch_Size, self.hidden_size, device=device)