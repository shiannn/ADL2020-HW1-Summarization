import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import json
#from RNN import LSTMTagger
#from RNN import HIDDEN_DIM, EMBEDDING_DIM
#from DoubleLstm import LSTMTagger
#from DoubleLstm import EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2
#from Dropout import LSTMTagger
#from Dropout import EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2
from seq2tag.BidirectionLSTM import LSTMTagger
from seq2tag.BidirectionLSTM import HIDDEN_DIM, EMBEDDING_DIM
import matplotlib.pyplot as plt
#from SeqTagDataSet import SeqTaggingDataset
import pickle

PADDINGWORD = 111359
PADDINGTARG = [-100,-100]

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
    if len(sys.argv) != 3:
        print('usage: python3 train.py train.pkl loadModel.pt')
        exit(0)
    with open(sys.argv[1],"rb") as FileTraining:
        #print(sys.argv[1])
        trainingData = pickle.load(FileTraining)
    
    print(trainingData)

    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    EPOCH = 10
    stEPOCH = 1

    BATCHSIZE = 32
    seqTaggingDataset = SeqTaggingDataset()
    zeroNum, oneNum = seqTaggingDataset.countClassNum(seqTaggingDataset.training_data)
    print(zeroNum, oneNum)
    # pos_weight should be negative/positive
    pos_weight_cal = torch.tensor([1, zeroNum / oneNum], dtype=torch.float)
    pos_weight_cal = pos_weight_cal.to(device)

    #glove = np.loadtxt('../myEmbedding.txt', dtype='str', comments=None)
    #words = glove[:, 0]
    #vectors = glove[:, 1:].astype('float')
    
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(seqTaggingDataset.word2idx), 2, 'none') # yes/no 2
    #model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2, len(word2idx), 2, vectors) # yes/no 2
    if sys.argv[2] != 'none':
        model.load_state_dict(torch.load(sys.argv[2]))
    model = model.to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight_cal, reduction='none')
    #optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.LBFGS(model.parameters(), lr=0.1)
    
    ### get DataLoader
    train_loader = DataLoader(dataset=seqTaggingDataset,
                batch_size=BATCHSIZE,
                shuffle=True,
                collate_fn=seqTaggingDataset.collate_fn)

    plotList = []
    for epoch in range(stEPOCH,EPOCH+1):  # again, normally you would NOT do 300 epochs, it is toy data
        lossValueEachEpoch = []
        for i, batch in enumerate(train_loader):
            #print(torch.tensor(batch['X']), torch.tensor(batch['Y']))
            try:
                Xlist = batch['X']
                Ylist = batch['Y']
                if Xlist == []:
                    continue
                meanInBatch = []
                model.zero_grad()
                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentence_in = torch.tensor(Xlist, dtype=torch.long).to(device)
                targets = torch.tensor(Ylist, dtype=torch.float64).to(device)
                # Step 3. Run our forward pass.
                tag_scores = model(sentence_in)

                #print('targets',targets)
                loss = loss_function(tag_scores, targets)
                loss = loss.view(-1,1)[targets.view(-1,1)>=0].mean()
                lossValueEachEpoch.append(loss)

                print('epoch:{}/{} batch:{}/{} loss:{}'.format(epoch, EPOCH, i, len(train_loader), loss))
                loss.backward()
                optimizer.step()
            except KeyboardInterrupt:
                print('Interrupted')
                exit(0)
        plotList.append(sum(lossValueEachEpoch)/len(lossValueEachEpoch))
        plt.plot(plotList)
        plt.savefig('figplot5/'+'1'+'-'+'1'+'withDataloader.png')    
        torch.save(model.state_dict(), 'checkpoint5/'+'BCEloader'+str(epoch)+'-'+str(i)+'.pt')
        """