import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import json
#from RNN import LSTMTagger
#from RNN import HIDDEN_DIM, EMBEDDING_DIM
#from seq2tag.DoubleLSTM import LSTMTagger
#from seq2tag.DoubleLSTM import EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2
from seq2tag.Dropout import LSTMTagger
from seq2tag.Dropout import EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2
#from seq2tag.BidirectionLSTM import LSTMTagger
#from seq2tag.BidirectionLSTM import HIDDEN_DIM, EMBEDDING_DIM
import matplotlib.pyplot as plt
from dataset import SeqTaggingDataset
import torch.utils.data as Data
import pickle
from predictSeq2Tag import postprocessing
from multiprocessing import Pool, cpu_count
from rouge_score.rouge_scorer import RougeScorer

ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']
USE_STEMMER = False

def countClassNum(training):
    zeroNum = 0
    oneNum = 0
    for td in trainingData:
        zeroNum += td['label'].count(0)
        oneNum += td['label'].count(1)
    return zeroNum, oneNum

def validError(predicts, answers):
    target = []
    prediction = []
    for id, p in predicts.items():
        a = answers.get(id, None)
        if a is None:
            raise Exception(f"Cannot find answer to Prediction ID: {id}")
        else:
            sent_bounds = {i: bound for i, bound in enumerate(a['sent_bounds'])}
            predict_sent = ''
            for sent_idx in p['predict_sentence_index']:
                start, end = sent_bounds.get(sent_idx, (0, 0))
                predict_sent += a['text'][start:end]
        target.append(a['summary'])
        prediction.append(predict_sent)        

    rouge_scorer = RougeScorer(ROUGE_TYPES, use_stemmer=USE_STEMMER)
    with Pool(cpu_count()) as pool:
        scores = pool.starmap(rouge_scorer.score,
                              [(t, p) for t, p in zip(target, prediction)])
    r1s = np.array([s['rouge1'].fmeasure for s in scores])
    r2s = np.array([s['rouge2'].fmeasure for s in scores])
    rls = np.array([s['rougeL'].fmeasure for s in scores])
    scores = {
        'mean': {
            'rouge-1': r1s.mean(),
            'rouge-2': r2s.mean(),
            'rouge-l': rls.mean()
        },
        'std': {
            'rouge-1': r1s.std(),
            'rouge-2': r2s.std(),
            'rouge-l': rls.std()
        },
    }
    print(json.dumps(scores, indent='    '))
    return [r1s.mean(),r1s.std(),
            r2s.mean(), r2s.std(),
            rls.mean(), rls.std()]

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('usage: python3 train.py train.pkl valid.pkl embedding.pkl loadModel.pt')
        exit(0)
    
    trainingName = sys.argv[1]
    validName = sys.argv[2]
    embeddingName = sys.argv[3]
    modelName = sys.argv[4]

    with open(trainingName,"rb") as FileTraining:
        #print(sys.argv[1])
        trainingData = pickle.load(FileTraining)
    
    with open(validName,"rb") as FileValidating:
        #print(sys.argv[1])
        validData = pickle.load(FileValidating)

    with open("data/valid.jsonl","r") as f:
        answers = [json.loads(line) for line in f]
        answers = {a['id']: a for a in answers}    
    
    trainingData = SeqTaggingDataset(trainingData)
    validData = SeqTaggingDataset(validData)
    BATCH_SIZE=32
    BATCH_Valid = 20
    EPOCH = 20
    stEPOCH = 1
    
    loader = Data.DataLoader(
        dataset=trainingData,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        collate_fn=trainingData.collate_fn
    )
    loader_valid = Data.DataLoader(
        dataset=validData,      # torch TensorDataset format
        batch_size=BATCH_Valid,      # mini batch size
        shuffle=False,               # 要不要打乱数据 (打乱比较好)
        collate_fn=validData.collate_fn
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
    M1 = []
    S1 = []
    M2 = []
    S2 = []
    Ml = []
    Sl = []
    for epoch in range(stEPOCH,EPOCH+1):  # again, normally you would NOT do 300 epochs, it is toy data
        trainingLoss = []
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
                trainingLoss.append(loss)

                print('epoch:{}/{} {}/{} loss:{}'.format(epoch, EPOCH, (i+1)*BATCH_SIZE, len(loader.dataset), loss))
                loss.backward()
                optimizer.step()
            except KeyboardInterrupt:
                print('Interrupted')
                exit(0)
        with torch.no_grad():
            predicts = {}
            for cnt,batch in enumerate(loader_valid):
                X = batch['text']
                Y = batch['label']
                sentRange = batch['sent_range']
                Id = batch['id']
                #print('Id',Id)
                #print(X,Y,Id)
                X = X.to(device, dtype=torch.long)
                tag_scores = model(X)
                #print(tag_scores.shape)
                print('len(sentRange)',len(sentRange))
                #print(tag_scores)
                ### tag_score (3, 296, 1)
                #tag_scores = torch.squeeze(tag_scores, 2)
                #exit(0)
                for i in range(BATCH_Valid):
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
                    #json.dump(toWrite, f_predict)
                    #f_predict.write("\n")
                    toWrite["id"] = Id[i]
                    predicts[Id[i]] = toWrite
                    
                print('predict_sent_idx {}/{}'.format((cnt+1)*BATCH_Valid, len(loader_valid.dataset)), predict_sent_idx)
        [m1,s1,m2,s2,ml,sl] = validError(predicts, answers)
        M1.append(m1)
        M2.append(m2)
        S1.append(s1)
        S2.append(s2)
        Ml.append(ml)
        Sl.append(sl)
        plotList.append(sum(trainingLoss)/len(trainingLoss))
        plt.plot(plotList)
        plt.plot(M1)
        plt.plot(S1)
        plt.plot(M2)
        plt.plot(S2)
        plt.plot(Ml)
        plt.plot(Sl)

        plt.savefig('figplot/'+'1'+'-'+'1'+'withDataloader.png')    
        torch.save(model.state_dict(), 'checkpoint/'+'BCEloader'+str(epoch)+'.pt')