import sys
import json
import torch
from dataset import Seq2SeqDataset
import torch.utils.data as Data
import torch
from torch import optim
import torch.nn as nn
from seq2seq.DecoderRNN import DecoderRNN
from seq2seq.EncoderRNN import EncoderRNN, hidden_size
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 1
EOS_token = 2

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def evaluate(encoder, decoder, input_tensor, max_length):
    with torch.no_grad():
        #input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            #encoder_output, encoder_hidden = encoder(input_tensor[ei],
            #                                         encoder_hidden)
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            #decoder_output, decoder_hidden, decoder_attention = decoder(
            #    decoder_input, decoder_hidden, encoder_outputs)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            #decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                #decoded_words.append(output_lang.index2word[topi.item()])
                decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        #return decoded_words, decoder_attentions[:di + 1]
        return decoded_words

def postprocessing(indexes, embedding):
    print(indexes)
    sentence = [embedding.vocab[idx] if isinstance(idx, int) else embedding.vocab[2] for idx in indexes]
    return sentence

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('usage: python3 predict.py encoder.pt decoder.pt embedding.pkl TestingData.pkl predict.jsonl')
        exit(0)

    encoderName = sys.argv[1]
    decoderName = sys.argv[2]
    embeddingName = sys.argv[3]
    testDataName = sys.argv[4]
    predictName = sys.argv[5]

    with open(embeddingName, 'rb') as f:
        embedding = pickle.load(f)
    encoder1 = EncoderRNN(len(embedding.vocab), hidden_size, embedding.vectors).to(device)
    encoder1.load_state_dict(torch.load(encoderName))
    decoder1 = DecoderRNN(hidden_size, len(embedding.vocab), embedding.vectors).to(device)
    decoder1.load_state_dict(torch.load(decoderName))

    encoder1 = encoder1.to(device)
    encoder1.eval()
    decoder1 = decoder1.to(device)
    decoder1.eval()

    with open('../datasets/seq2seq/config.json') as f:
        config = json.load(f)
    
    maxTextLen = config.get('max_text_len')
    maxSummaryLen = config.get('max_summary_len')

    with open(testDataName,"rb") as FileTesting:
        #print(sys.argv[1])
        testingData = pickle.load(FileTesting)

    testingData = Seq2SeqDataset(testingData)

    BATCH_SIZE = 1
    loader = Data.DataLoader(
        dataset=testingData,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        collate_fn=testingData.collate_fn
    )

    max_length = max(maxTextLen, maxSummaryLen)
    with torch.no_grad():
        with open(predictName,'w') as f_predict:
            for cnt,batch in enumerate(loader):
                print(batch.keys())
                X = batch['text'].reshape(-1,1).to(device)
                print(X.shape)
                outputWord = evaluate(encoder1, decoder1, X, max_length)
                sent = postprocessing(outputWord, embedding)
                print(sent)
                break