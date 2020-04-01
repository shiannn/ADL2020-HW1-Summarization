import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim1, hidden_dim2, vocab_size, tagset_size, pretrained_weight):
        super(LSTMTagger, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if (pretrained_weight != 'none'):
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight))

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim1, dropout=0.2)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, dropout=0.2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim2, tagset_size)
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm1_out, _ = self.lstm1(embeds.view(len(sentence), 1, -1))
        lstm2_out, _ = self.lstm2(lstm1_out.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm2_out.view(len(sentence), -1))
        #tag_scores = F.log_softmax(tag_space, dim=1)
        tag_scores = tag_space
        return tag_scores

EMBEDDING_DIM = 300
HIDDEN_DIM1 = 100
HIDDEN_DIM2 = 50