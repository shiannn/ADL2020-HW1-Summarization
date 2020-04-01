import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import sys
#from RNN import LSTMTagger
#from RNN import HIDDEN_DIM, EMBEDDING_DIM
from BidirectionLstm import LSTMTagger
from BidirectionLstm import HIDDEN_DIM, EMBEDDING_DIM
from usedataset import SeqTaggingDataset
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TARGETONE = torch.tensor([0,1], dtype=torch.float64).to(device)
TARGETZERO = torch.tensor([1,0], dtype=torch.float64).to(device)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else 0 for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def postprocessing(tag_scores, test):
    output = {}
    #print(tag_scores)
    #print(test)
    wordList = test[0]
    splitInterval = test[1]
    ratio = []
    predict_sent_idx = []
    loss_function = nn.BCEWithLogitsLoss()

    for i in range(len(splitInterval)-1):
        #the i'th sentence
        wordInSent = splitInterval[i+1] - splitInterval[i]
        #print('wordInSent',wordInSent)
        wordselected = 0
        for word_score in tag_scores[splitInterval[i]:splitInterval[i+1]]:
            #print('loss on [0,1]',loss_function(word_score, TARGETONE))
            #print('loss on [1,0]',loss_function(word_score, TARGETZERO))
            if loss_function(word_score, TARGETONE).item() < loss_function(word_score, TARGETZERO).item():
                wordselected += 1
            #elif loss_function(word_score, [0,1]) > loss_function(word_score, [1,0])::
            """
            if word_score[0] < word_score[1]:
                #predict_sent_idx.append(i)
                wordselected += 1
            """
        ratio.append(wordselected/wordInSent)
    predict_sent_idx.append(ratio.index(max(ratio)))
    return predict_sent_idx

if __name__ == '__main__':
    if(len(sys.argv)!=4):
        print('usage: python3 predict.py model.pt TestingData.npy predict.jsonl')
        exit(0)
    
    # sys.argv[1] embedding (TrainingData.npy)
    # sys.argv[2] TestX (TestingData.npy)
    # sys.argv[3] TestY predict result (predict.jsonl)

    testing_data = np.load(sys.argv[2], allow_pickle=True)
    seqTaggingDataset = SeqTaggingDataset(testing_data)
    #model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), 2, 'none') # yes/no 2
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(seqTaggingDataset.word2idx), 2, 'none') # yes/no 2
    model.load_state_dict(torch.load(sys.argv[1]))
    model = model.to(device)
    model.eval()

# 15779 出問題

    with torch.no_grad():
        with open(sys.argv[3],'w') as f_predict:
            for cnt,test in enumerate(testing_data):
                Id = test[2]
                toWrite = {}
                inputs = prepare_sequence(test[0], seqTaggingDataset.word2idx)
                inputs = inputs.to(device)
                print(inputs.shape)
                tag_scores = model(inputs)
                #print(tag_scores)
                predict_sent_idx = postprocessing(tag_scores, test)
                print(Id, predict_sent_idx)
                toWrite["id"] = Id
                toWrite["predict_sentence_index"] = predict_sent_idx
                json.dump(toWrite, f_predict)
                f_predict.write("\n")
                """
                try:
                    inputs = prepare_sequence(test[0], word2idx)
                    inputs = inputs.to(device)
                    #print(test[0])
                    tag_scores = model(inputs)
                    #print(tag_scores)
                    predict_sent_idx = postprocessing(tag_scores, test)
                    print(Id, predict_sent_idx)
                    toWrite["id"] = Id
                    toWrite["predict_sentence_index"] = predict_sent_idx
                    json.dump(toWrite, f_predict)
                    f_predict.write("\n")
                except KeyboardInterrupt:
                    print('Interrupted')
                    exit(0)
                except:
                    toWrite["id"] = Id
                    toWrite["predict_sentence_index"] = []
                    json.dump(toWrite, f_predict)
                    f_predict.write("\n")
                """