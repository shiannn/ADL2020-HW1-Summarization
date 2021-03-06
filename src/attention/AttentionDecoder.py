import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, self.batch_Size, -1)
        embedded = self.dropout(embedded)

        ### batch_size*(embed_size+hidden_size)
        embedded_hidden = torch.cat((embedded[0], hidden[0]), 1)
        #print('embedded_hidden.shape',embedded_hidden.shape)

        attn_weights = F.softmax(
            self.attn(embedded_hidden), dim=1)
        #print('attn_weights.shape', attn_weights.shape)
        ### batch_size*max_length
        threeDattn_weights = attn_weights.unsqueeze(0)
        ### threeDattn_weights (1, 3, 300) (1, batch_size, max_length)
        #threeDencoder_outputs = encoder_outputs.unsqueeze(0)
        ### threeDencoder_outputs (300, 3, 300) (maxLen, batch_size, hidden_size)
        threeDattn_weights = threeDattn_weights.permute(1,0,2)
        threeDencoder_outputs = encoder_outputs
        threeDencoder_outputs = threeDencoder_outputs.permute(1,0,2)
        #print('threeDattn_weights.shape', threeDattn_weights.shape)
        #print('threeDencoder_outputs.shape', threeDencoder_outputs.shape)

        attn_applied = torch.bmm(threeDattn_weights, threeDencoder_outputs)
        #print('attn_applied.shape', attn_applied.shape)

        ### embedded (seq_len, batch_size, feature)
        ### attn_applied (batch_size, 1, hidden_size)
        output = torch.cat((embedded[0], attn_applied.squeeze(1)), 1)
        #print('output.shape', output.shape)
        ### out (batch_size, (feature+hidden_size))

        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, self.batch_Size, self.hidden_size, device=device)