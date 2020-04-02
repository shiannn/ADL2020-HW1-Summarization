import sys
from dataset import Seq2SeqDataset
import torch.utils.data as Data
import torch
from torch import optim
import torch.nn as nn
import pickle
import json
from seq2seq.DecoderRNN import DecoderRNN
from seq2seq.EncoderRNN import EncoderRNN
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 1
EOS_token = 2

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    #print('input_tensor', input_tensor)
    #print('input_tensor', target_tensor)
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    teacher_forcing_ratio = 0.5
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            #decoder_output, decoder_hidden, decoder_attention = decoder(
            #    decoder_input, decoder_hidden, encoder_outputs)
            #decoder_output, decoder_hidden, decoder_attention = decoder(
            #    decoder_input, decoder_hidden)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            #decoder_output, decoder_hidden, decoder_attention = decoder(
            #    decoder_input, decoder_hidden, encoder_outputs)
            #decoder_output, decoder_hidden, decoder_attention = decoder(
            #    decoder_input, decoder_hidden)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

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

    BATCH_SIZE=1
    hidden_size = 256

    with open(embeddingName, 'rb') as f:
        embedding = pickle.load(f)

    encoder = EncoderRNN(len(embedding.vocab), hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, len(embedding.vocab)).to(device)

    loader = Data.DataLoader(
        dataset=trainingData,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        collate_fn=trainingData.collate_fn
    )
    
    #maxSummaryLen = max([len(sample['summary']) for sample in trainingData])
    #maxTextLen = max([len(sample['text']) for sample in trainingData])
    with open('../datasets/seq2seq/config.json') as f:
        config = json.load(f)
    
    maxTextLen = config.get('max_text_len')
    maxSummaryLen = config.get('max_summary_len')
    print(maxTextLen, maxSummaryLen)
    
    learning_rate=0.01
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    max_length = max(maxTextLen, maxSummaryLen)
    for i, batch in enumerate(loader):
        input_tensor = batch['text'].reshape(-1,1).to(device)
        target_tensor = batch['summary'].reshape(-1,1).to(device)        
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)
        print('{}/{} loss:{}'.format(i, len(loader.dataset), loss))