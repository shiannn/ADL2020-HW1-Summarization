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
from predictSeq2Tag import postprocessing

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    BATCH_SIZE = 40
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
            print("{}/{}".format((cnt+1)*BATCH_SIZE, len(loader.dataset)))
            X = batch['text']
            Y = batch['label']
            sentRange = batch['sent_range']
            Id = batch['id']
            #print('Id',Id)
            #print(X,Y,Id)
            X = X.to(device, dtype=torch.long)
            tag_scores = model(X)
            
            for i in range(BATCH_SIZE):
                predict_sent_idx = postprocessing(tag_scores[i], sentRange[i])
                numSentence = len(sentRange[i])
                #print('numSentence', numSentence)
                ratios = [idx/numSentence for idx in predict_sent_idx]
                histList.extend(ratios)
                #print('ratios', ratios)
                #print('predict_sent_idx', predict_sent_idx)
        #binNum = 10
        binEdge = np.arange(0, 1, step=0.05)
        plt.hist(histList, binEdge, density=False, facecolor='blue', alpha=0.5, ec="k")
        plt.xticks(binEdge, rotation=45)
        plt.xlabel('relative position of summary')
        plt.ylabel('number of occurence')
        #plt.xticks(range(binNum))
        plt.tight_layout()
        plt.savefig("hist.png")