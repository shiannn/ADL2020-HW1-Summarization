import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from seq2seq.EncoderRNN import hidden_size

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, pretrained_weight, batch_Size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.batch_Size = batch_Size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(2*self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, self.batch_Size, -1)
        embedded = self.dropout(embedded)
        ### batch_size*(embed_size+hidden_size)
        output = F.relu(embedded)
        output, hidden = self.gru(output, hidden)
        ### hidden (1, batch_size, hidden_size)
        ### out (seq_len, batch_size, hidden_size)
        ### attn_applied (batch_size, 1, hidden_size)

        #output_hidden = torch.cat((output[0], hidden[0]), 1)
        ### using output to match all the encoder_outputs
        ### output[0] (batch_size, hidden_size) hidden[0] (batch_size, hidden_size)
        ### output_hidden (batch_size, 2*hidden_size)
        cos = CosineSimilarity(dim=2, eps=1e-7)
        similar = cos(output.repeat(self.max_length, 1,1), encoder_outputs)
        ### out (seq_len, batch_size, hidden_size)
        ### encoder out_put (seq_len, batch_size, hidden_size)
        #print('similar.shape', similar.shape)
        #print('similar', similar)
        ### similar (max_len, hidden)
        #attn_weights = F.softmax(
        #    self.attn(similar), dim=1)
        attn_weights = F.softmax(similar, dim=0)
        #print('attn_weights.shape', attn_weights)
        attn_weights = torch.t(attn_weights)
        #print('attn_weights.shape', attn_weights)
        ### attn_weights (batch_size*max_length)
        threeDattn_weights = attn_weights.unsqueeze(0)
        ### threeDattn_weights (1, 3, 300) (1, batch_size, max_length)
        #threeDencoder_outputs = encoder_outputs.unsqueeze(0)
        ### threeDencoder_outputs (300, 3, 300) (maxLen, batch_size, hidden_size)
        threeDattn_weights = threeDattn_weights.permute(1,0,2)
        threeDencoder_outputs = encoder_outputs
        threeDencoder_outputs = threeDencoder_outputs.permute(1,0,2)

        attn_applied = torch.bmm(threeDattn_weights, threeDencoder_outputs)
        ### (3,1,300)x(3,300,300) = (3,1,300)
        ### embedded (seq_len, batch_size, feature)

        
        IntoLinear = torch.cat((output[0], attn_applied.squeeze(1)), 1)
        ### output[0] (batch_size, hidden_size) attn_applied (batch_size, hidden_size)
        #IntoLinear (batch_size, 2*hidden_size)

        output = F.log_softmax(self.out(IntoLinear), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, self.batch_Size, self.hidden_size, device=device)