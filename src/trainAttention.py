import sys
from dataset import Seq2SeqDataset
import torch.utils.data as Data
import torch
from torch import optim
import torch.nn as nn
import pickle
import json
from seq2seq.AttentionDecoder import AttnDecoderRNN
from seq2seq.AttentionEncoder import AttnEncoderRNN, hidden_size
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 1
EOS_token = 2
BATCH_SIZE = 32

def tensor2word(tns, embedding):
    words = [embedding.vocab[a] for a in tns]
    return words

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, embedding):
    input_tensor = torch.t(input_tensor)
    target_tensor = torch.t(target_tensor)
    #print('input_tensor', input_tensor)
    #print('target_tensor', target_tensor)
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    ### for the longest sentence
    encoder_outputs = torch.zeros(max_length, BATCH_SIZE, encoder.hidden_size, device=device)

    TotalLoss = 0

    # input_tensor (seq_len, batch_size)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        ### encoder_output (seq_len, batch_size, hidden_size)
        encoder_outputs[ei] = encoder_output[0]
    """
    encoder_output, encoder_hidden = encoder(
        input_tensor, encoder_hidden)
    """
    ### encoder_outputs (maxLen, batch, hidden_size)

    #decoder_input = torch.tensor([[SOS_token]]* BATCH_SIZE, device=device)
    decoder_input = torch.tensor([SOS_token]* BATCH_SIZE, device=device)

    decoder_hidden = encoder_hidden
    # (num_layers*num_directions, batch_size, hidden_size)

    for di in range(target_length):
        #decoder_output, decoder_hidden, decoder_attention = decoder(
        #    decoder_input, decoder_hidden, encoder_outputs)
        #decoder_output, decoder_hidden, decoder_attention = decoder(
        #    decoder_input, decoder_hidden)
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        #print('decoder_output')
        #print(decoder_output)
        #print('decoder_output.shape', decoder_output.shape)
        values, indices = torch.topk(decoder_output,k=1,dim=1)
        #print('indices', indices)
        words = tensor2word(indices, embedding)
        #print(words)
        #print('target_tensor[di]', target_tensor[di])
        #print('target_tensor[di].shape', target_tensor[di].shape)
        loss = criterion(decoder_output, target_tensor[di])
        TotalLoss += loss[target_tensor[di]>0].mean()
        decoder_input = target_tensor[di]  # Teacher forcing

    TotalLoss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return TotalLoss.item() / target_length

if __name__ == '__main1__':
    if len(sys.argv) != 5:
        print('usage: python3 train.py train.pkl valid.pkl embedding.pkl loadModel.pt')
        exit(0)
    trainingName = sys.argv[1]
    validName = sys.argv[2]
    embeddingName = sys.argv[3]
    modelName = sys.argv[4]

    with open(embeddingName, 'rb') as f:
        embedding = pickle.load(f)
    A = torch.tensor([1,2,3,4,5,6,7,8,9,10])
    tensor2word(A, embedding)

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

    """
    with open("data/valid.jsonl","r") as f:
        answers = [json.loads(line) for line in f]
        answers = {a['id']: a for a in answers}    
    """
    
    trainingData = Seq2SeqDataset(trainingData)
    validData = Seq2SeqDataset(validData)

    with open(embeddingName, 'rb') as f:
        embedding = pickle.load(f)
    
    with open('../datasets/seq2seq/config.json', 'r') as f:
        config = json.load(f)
    
    maxTextLen = config.get('max_text_len')
    maxSummaryLen = config.get('max_summary_len')
    print(maxTextLen, maxSummaryLen)
    maxLength = max(maxTextLen, maxSummaryLen)

    encoder = AttnEncoderRNN(len(embedding.vocab), hidden_size, embedding.vectors, BATCH_SIZE).to(device)
    #decoder = DecoderRNN(hidden_size, len(embedding.vocab), embedding.vectors, BATCH_SIZE).to(device)
    decoder = AttnDecoderRNN(hidden_size, len(embedding.vocab), embedding.vectors, BATCH_SIZE, maxLength, dropout_p=0.1).to(device)

    loader = Data.DataLoader(
        dataset=trainingData,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        #num_workers=1,              # 多线程来读数据
        drop_last=True,
        collate_fn=trainingData.collate_fn,
    )

    
    learning_rate=0.001
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(reduction='none')

    print('hello')
    epochError = []
    tempError = []
    EPOCH = 10
    for epoch in range(EPOCH):
        for i, batch in enumerate(loader):
            #input_tensor = batch['text'].reshape(-1,1).to(device)
            #target_tensor = batch['summary'].reshape(-1,1).to(device)
            input_tensor = batch['text'].to(device)
            target_tensor = batch['summary'].to(device)
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, maxLength, embedding)
            print('training loss {} epoch {}/{} {}/{}'.format(loss, epoch, EPOCH, i*BATCH_SIZE, len(loader.dataset)))
            tempError.append(loss)
        epochError.append(sum(tempError)/len(tempError))
        tempError = []

        plt.plot(epochError)
        plt.savefig('figplot_attention/'+'1'+'-'+'1'+'.png')    
        torch.save(encoder.state_dict(), 'checkpoint_attention/'+'encoder'+str(epoch)+'.pt')
        torch.save(decoder.state_dict(), 'checkpoint_attention/'+'decoder'+str(epoch)+'.pt')