import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import sys
import pickle
#from seq2tag.DoubleLSTM import LSTMTagger
#from seq2tag.DoubleLSTM import EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2
from seq2tag.Dropout import LSTMTagger
from seq2tag.Dropout import EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2
#from seq2tag.BidirectionLSTM import LSTMTagger
#from seq2tag.BidirectionLSTM import HIDDEN_DIM, EMBEDDING_DIM
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
from torch.nn import Sigmoid
from dataset import SeqTaggingDataset
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TARGETONE = torch.tensor([0,1], dtype=torch.float64).to(device)
TARGETZERO = torch.tensor([1,0], dtype=torch.float64).to(device)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else 0 for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def postprocessing(tag_scores,sent_range):
    tag_scores = tag_scores.squeeze()
    #label = label.squeeze()
    #print(tag_scores)
    #print(label)
    #loss_function = nn.BCEWithLogitsLoss()
    #loss = loss_function(tag_scores, torch.tensor())
    oneS = torch.ones(tag_scores.shape).to(device)
    zeroS = torch.zeros(tag_scores.shape).to(device)
    #print(oneS)
    #print(zeroS)
    f = Sigmoid()
    predictLabel = torch.where(f(tag_scores)>0.5,oneS,zeroS)
    #print(predictLabel)
    #print(sent_range)
    
    maxRatio = -1
    predict_sent_idx = -1
    for i in range(len(sent_range[0])):
        #the i'th sentence
        st = sent_range[0][i][0]
        ed = sent_range[0][i][1]
        wordInSent = ed - st
        #print('wordInSent',wordInSent)
        wordselected = (predictLabel[st:ed]==1).sum()
        #print('wordselected',wordselected)
        ratio = wordselected / wordInSent
        if ratio > maxRatio:
            predict_sent_idx = i
            maxRatio = ratio
    return [predict_sent_idx]

if __name__ == '__main__':
    if(len(sys.argv)!=4):
        print('usage: python3 plotingRelative.py model.pt embedding.pkl TestingData.pkl')
        exit(0)
    
    modelName = sys.argv[1]
    embeddingName = sys.argv[2]
    testDataName = sys.argv[3]
    # sys.argv[1] embedding (TrainingData.npy)
    # sys.argv[2] TestX (TestingData.npy)
    # sys.argv[3] TestY predict result (predict.jsonl)

    with open(testDataName,"rb") as FileTesting:
        #print(sys.argv[1])
        testingData = pickle.load(FileTesting)

    testingData = SeqTaggingDataset(testingData)

    BATCH_SIZE = 1
    loader = Data.DataLoader(
        dataset=testingData,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=False,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        collate_fn=testingData.collate_fn
    )

    with open(embeddingName, 'rb') as f:
        embedding = pickle.load(f)
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2, len(embedding.vocab), 1, embedding.vectors) # yes/no 2
    #model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(embedding.vocab), 1, 'none') # yes/no 2
    model.load_state_dict(torch.load(modelName))
    model = model.to(device)
    model.eval()

    histList = []
    with torch.no_grad():
        for cnt,batch in enumerate(loader):
            try:
                print('counting {}/{}'.format(cnt, len(loader.dataset)))
                #print(batch.keys())
                X = batch['text']
                Y = batch['label']
                sentRange = batch['sent_range']
                Id = batch['id'][0]
                numSentence = len(sentRange[0])
                X = X.to(device, dtype=torch.long)
                #print(X.shape)
                tag_scores = model(X)
                #print(tag_scores)
                predict_sent_idx = postprocessing(tag_scores, sentRange)
                print('predict_sent_idx', predict_sent_idx)
                
                ratio = predict_sent_idx[0] / numSentence
                histList.append(ratio)
            except KeyboardInterrupt:
                print('Interrupted')
                exit(0)
            except:
                print('error')
    plt.hist(histList)
    plt.savefig("hist.png")