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

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else 0 for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def postprocessing(tag_scores,sent_range):
    #tag_scores = tag_scores.squeeze()
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
    for i in range(len(sent_range)):
        #the i'th sentence
        st = sent_range[i][0]
        ed = sent_range[i][1]
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
    if(len(sys.argv)!=5):
        print('usage: python3 predict.py model.pt embedding.pkl TestingData.pkl predict.jsonl')
        exit(0)
    
    modelName = sys.argv[1]
    embeddingName = sys.argv[2]
    testDataName = sys.argv[3]
    predictName = sys.argv[4]
    # sys.argv[1] embedding (TrainingData.npy)
    # sys.argv[2] TestX (TestingData.npy)
    # sys.argv[3] TestY predict result (predict.jsonl)

    with open(testDataName,"rb") as FileTesting:
        #print(sys.argv[1])
        testingData = pickle.load(FileTesting)

    testingData = SeqTaggingDataset(testingData)

    BATCH_SIZE = 20
    loader_valid = Data.DataLoader(
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

    with torch.no_grad():
        with open(predictName,'w') as f_predict:
            for cnt,batch in enumerate(loader_valid):
                """
                try:
                    X = batch['text']
                    Y = batch['label']
                    sentRange = batch['sent_range']
                    Id = batch['id'][0]
                    #print(X,Y,Id)
                    X = X.to(device, dtype=torch.long)
                    print(X.shape)
                    tag_scores = model(X)
                    ### tag_scores (1,296,1)
                    print(tag_scores)
                    print(sentRange)
                    predict_sent_idx = postprocessing(tag_scores, sentRange)
                    print('predict_sent_idx {}/{}'.format(cnt, len(loader_valid.dataset)), predict_sent_idx)
                    toWrite = {}
                    toWrite["id"] = Id
                    toWrite["predict_sentence_index"] = predict_sent_idx
                except KeyboardInterrupt:
                    print('Interrupted')
                    exit(0)
                except:
                    toWrite["id"] = Id
                    toWrite["predict_sentence_index"] = []
                json.dump(toWrite, f_predict)
                f_predict.write("\n")
                
                """
                X = batch['text']
                Y = batch['label']
                sentRange = batch['sent_range']
                Id = batch['id']
                #print('Id',Id)
                #print(X,Y,Id)
                X = X.to(device, dtype=torch.long)
                tag_scores = model(X)
                #print(tag_scores.shape)
                #print(sentRange)
                #print(tag_scores)
                ### tag_score (3, 296, 1)
                #tag_scores = torch.squeeze(tag_scores, 2)
                #exit(0)
                for i in range(BATCH_SIZE):
                    toWrite = {}
                    predict_sent_idx = postprocessing(tag_scores[i], sentRange[i])
                    #print('predict_sent_idx', predict_sent_idx)
                    try:
                        toWrite["predict_sentence_index"] = predict_sent_idx
                    except KeyboardInterrupt:
                        print('Interrupted')
                        exit(0)
                    except:
                        toWrite["predict_sentence_index"] = []
                    toWrite["id"] = Id[i]
                    json.dump(toWrite, f_predict)
                    f_predict.write("\n")
                    
                print('predict_sent_idx {}/{}'.format((cnt+1)*BATCH_SIZE, len(loader_valid.dataset)), predict_sent_idx)