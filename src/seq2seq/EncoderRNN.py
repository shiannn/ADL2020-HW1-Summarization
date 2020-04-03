import torch.nn as nn
import torch

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
        embedded = self.embedding(input).view(1,1,-1)
        #embedded = self.embedding(input)
        #print(embedded)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, 1, self.hidden_size, device=device)

hidden_size = 300