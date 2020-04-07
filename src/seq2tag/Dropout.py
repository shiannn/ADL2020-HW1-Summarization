import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim1, hidden_dim2, vocab_size, tagset_size, pretrained_weight):
        super(LSTMTagger, self).__init__()
        """
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = 30
        self.hidden_dim4 = 10
        self.hidden_dim5 = 5
        """
        self.hidden_dim = 15

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if (pretrained_weight is not 'none'):
            self.word_embeddings.weight.data.copy_(pretrained_weight)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        """
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2*hidden_dim1, hidden_dim2, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(2*hidden_dim2, self.hidden_dim3, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(2*self.hidden_dim3, self.hidden_dim4, batch_first=True, bidirectional=True)
        self.lstm5 = nn.LSTM(2*self.hidden_dim4, self.hidden_dim5, batch_first=True, bidirectional=True)
        """
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=5, batch_first=True, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2*self.hidden_dim, tagset_size)
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        """
        lstm1_out, _ = self.lstm1(embeds)
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm3_out, _ = self.lstm3(lstm2_out)
        lstm4_out, _ = self.lstm4(lstm3_out)
        lstm5_out, _ = self.lstm5(lstm4_out)
        """
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        #tag_scores = F.log_softmax(tag_space, dim=1)
        tag_scores = tag_space
        return tag_scores

EMBEDDING_DIM = 300
HIDDEN_DIM1 = 100
HIDDEN_DIM2 = 60