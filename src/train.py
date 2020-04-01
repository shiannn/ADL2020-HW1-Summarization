import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import json
#from RNN import LSTMTagger
#from RNN import HIDDEN_DIM, EMBEDDING_DIM
from seq2tag.DoubleLSTM import LSTMTagger
from seq2tag.DoubleLSTM import EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2
#from Dropout import LSTMTagger
#from Dropout import EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2
#from seq2tag.BidirectionLSTM import LSTMTagger
#from seq2tag.BidirectionLSTM import HIDDEN_DIM, EMBEDDING_DIM
import matplotlib.pyplot as plt
from dataset import SeqTaggingDataset
import torch.utils.data as Data
import pickle

PADDINGWORD = 111359
PADDINGTARG = [-100,-100]

def countClassNum(training):
    zeroNum = 0
    oneNum = 0
    for td in trainingData:
        zeroNum += td['label'].count(0)
        oneNum += td['label'].count(1)
    return zeroNum, oneNum

def getMaxLen(training_data):
    maxLen = -1
    for i in range(len(training_data)):
        temp = len(training_data[i][0])
        if temp > maxLen:
            maxLen = temp
    return maxLen

def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        Low = w.lower()
        if Low in to_ix:
            idxs.append(to_ix[Low])
        else:
            idxs.append(0)
    """
    ### padding
    for i in range(len(seq), maxLen):
        idxs.append(PADDINGWORD)
    #return torch.tensor(idxs, dtype=torch.long)
    """
    return idxs

def prepare_target(tags):
    #print(tags)
    targets = []
    for tag in tags:
        if tag == 0:
            targets.append([1,0])
        elif tag == 1:
            targets.append([0,1])
    """
    for i in range(len(tags), maxLen):
        targets.append(PADDINGTARG)
    """
    return targets

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: python3 train.py train.pkl embedding.pkl loadModel.pt')
        exit(0)
    
    trainingName = sys.argv[1]
    embeddingName = sys.argv[2]
    modelName = sys.argv[3]

    with open(trainingName,"rb") as FileTraining:
        #print(sys.argv[1])
        trainingData = pickle.load(FileTraining)
    
    trainingData = SeqTaggingDataset(trainingData)
    BATCH_SIZE=32
    EPOCH = 20
    stEPOCH = 1
    
    loader = Data.DataLoader(
        dataset=trainingData,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        collate_fn=trainingData.collate_fn
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    zeroNum, oneNum = countClassNum(trainingData)
    print(zeroNum, oneNum)
    # pos_weight should be negative/positive
    pos_weight_cal = torch.tensor(zeroNum / oneNum, dtype=torch.float)
    pos_weight_cal = pos_weight_cal.to(device)

    #glove = np.loadtxt('../myEmbedding.txt', dtype='str', comments=None)
    #words = glove[:, 0]
    #vectors = glove[:, 1:].astype('float')
    #Load embedding
    with open(embeddingName, 'rb') as f:
        embedding = pickle.load(f)

    #print(len(embedding.vocab))
    #print(embedding.vocab[:10])
    #model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(embedding.vocab), 1, 'none') # yes/no 2
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2, len(embedding.vocab), 1, embedding.vectors) # yes/no 2
    if modelName != 'none':
        model.load_state_dict(torch.load(modelName))
    model = model.to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight_cal, reduction='none')
    #optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.LBFGS(model.parameters(), lr=0.1)
    
    plotList = []
    for epoch in range(stEPOCH,EPOCH+1):  # again, normally you would NOT do 300 epochs, it is toy data
        lossValueEachEpoch = []
        for i, batch in enumerate(loader):
            #print(batch.keys())
            try:
                X = batch['text']
                Y = batch['label']
                meanInBatch = []
                model.zero_grad()
                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentence_in = X.to(device, dtype=torch.long)
                targets = Y.to(device, dtype=torch.float64)
                #targets = targets.unsqueeze(2)
                # Step 3. Run our forward pass.
                scores = model(sentence_in)
                scores = torch.squeeze(scores, 2)
                #print(scores.shape)
                #print(targets.shape)
                loss = loss_function(scores, targets)
                
                #loss = torch.where((targets<0), loss, loss)
                loss = loss[targets>=0].mean()
                #loss = loss.view(-1,1)[targets.view(-1,1)>0].mean()
                lossValueEachEpoch.append(loss)

                print('epoch:{}/{} {}/{} loss:{}'.format(epoch, EPOCH, i, len(loader.dataset), loss))
                loss.backward()
                optimizer.step()
            except KeyboardInterrupt:
                print('Interrupted')
                exit(0)
        plotList.append(sum(lossValueEachEpoch)/len(lossValueEachEpoch))
        plt.plot(plotList)
        plt.savefig('figplot/'+'1'+'-'+'1'+'withDataloader.png')    
        torch.save(model.state_dict(), 'checkpoint/'+'BCEloader'+str(epoch)+'.pt')