import sys
import json
import torch
from dataset import Seq2SeqDataset
import torch.utils.data as Data
import torch
from torch import optim
import torch.nn as nn
#from seq2seq.AttentionDecoder import AttnDecoderRNN
#from attention.AttentionDecoderCat import AttnDecoderRNN
#from attention.AttentionDecoderCat2 import AttnDecoderRNN
#from attention.AttentionEncoder import AttnEncoderRNN, hidden_size
from attention.AttentionEncoderBi import AttnEncoderRNN, hidden_size
from attention.AttentionDecoderCosBi import AttnDecoderRNN
import pickle

BATCH_SIZE = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 1
EOS_token = 2

def tensor2word(tns, embedding):
    words = [embedding.vocab[a] for a in tns]
    return words

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def evaluate(encoder, decoder, input_tensor, max_length):
    #input_tensor = tensorFromSentence(input_lang, sentence)
    input_tensor = torch.t(input_tensor)
    #print(input_tensor)
    input_length = input_tensor.size()[0]
    batchSize = input_tensor.size()[1]
    
    encoder_hidden = encoder.initHidden()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    print('input_length', input_length)
    encoder_output, encoder_hidden = encoder(
        input_tensor, encoder_hidden)
    
    if max_length-encoder_output.shape[0] > 0:
        ZERO = torch.zeros(max_length-encoder_output.shape[0], BATCH_SIZE, hidden_size).to(device)
        encoder_outputs = torch.cat((encoder_output, ZERO),0)
    else:
        encoder_outputs = encoder_output
    """
    for ei in range(input_length):
        #encoder_output, encoder_hidden = encoder(input_tensor[ei],
        #                                         encoder_hidden)
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                    encoder_hidden)
        #encoder_outputs[ei] += encoder_output[0, 0]
    """

    decoder_input = torch.tensor([SOS_token]* BATCH_SIZE, device=device)

    decoder_hidden = encoder_hidden
    #print('decoder_hidden', decoder_hidden)

    decoded_words = [[] for i in range(BATCH_SIZE)]
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        #decoder_output, decoder_hidden, decoder_attention = decoder(
        #    decoder_input, decoder_hidden, encoder_outputs)
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        #print('decoder_output', decoder_output)
    
        #decoder_attentions[di] = decoder_attention.data
        #print('decoder_output', decoder_output)
        #print('decoder_output.shape', decoder_output.shape)
        #print('decoder_output.data', decoder_output.data)
        values, indices = torch.topk(decoder_output,k=1,dim=1)
        #print('indices', indices)
        words = tensor2word(indices, embedding)
        #print('words', words)
        #topv, topi = decoder_output.data.topk(1)
        for j in range(BATCH_SIZE):
            if indices[j].item() == EOS_token:
                decoded_words[j].append('<EOS>')
            else:
                #decoded_words.append(output_lang.index2word[topi.item()])
                decoded_words[j].append(indices[j].item())

        decoder_input = indices.squeeze().detach()
        #decoder_input = indices[0]
    #return decoded_words, decoder_attentions[:di + 1]
    #print('decoder_words', decoded_words)
    return decoded_words
    

def postprocessing(indexes, embedding):
    #print(indexes)
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

    with open('../datasets/seq2seq/config.json') as f:
        config = json.load(f)
    
    maxTextLen = config.get('max_text_len')
    maxSummaryLen = config.get('max_summary_len')
    maxLength = max(maxTextLen, maxSummaryLen)

    encoder1 = AttnEncoderRNN(len(embedding.vocab), hidden_size, embedding.vectors, BATCH_SIZE).to(device)
    encoder1.load_state_dict(torch.load(encoderName))
    decoder1 = AttnDecoderRNN(hidden_size, len(embedding.vocab), embedding.vectors, BATCH_SIZE, maxLength, dropout_p=0.1).to(device)
    decoder1.load_state_dict(torch.load(decoderName))

    encoder1 = encoder1.to(device)
    encoder1.eval()
    decoder1 = decoder1.to(device)
    decoder1.eval()

    with open(testDataName,"rb") as FileTesting:
        #print(sys.argv[1])
        testingData = pickle.load(FileTesting)

    testingData = Seq2SeqDataset(testingData)

    def pad_to_len(seqs, to_len, padding=0):
        paddeds = []
        for seq in seqs:
            paddeds.append(
                seq[:to_len] + [padding] * max(0, to_len - len(seq))
            )

        return paddeds


    def attention_collate_fn(samples):
        batch = {}
        for key in ['id', 'len_text', 'len_summary']:
            batch[key] = [sample[key] for sample in samples]

        for key in ['text', 'summary', 'attention_mask']:
            #to_len = max([len(sample[key]) for sample in samples])
            to_len = maxLength
            padded = pad_to_len(
                [sample[key] for sample in samples], to_len, 0
            )
            batch[key] = torch.tensor(padded)

        return batch

    loader = Data.DataLoader(
        dataset=testingData,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=False,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        collate_fn=attention_collate_fn
    )

    #max_length = max(maxTextLen, maxSummaryLen)
    max_length = maxSummaryLen
    with torch.no_grad():
        with open(predictName,'w') as f_predict:
            for cnt,batch in enumerate(loader):
                print("{}/{}".format(cnt*BATCH_SIZE, len(loader.dataset)))
                toWrite = {}
                #print(batch.keys())
                X = batch['text'].to(device)
                idx = batch['id']
                print(idx)
                outputWordBatch = evaluate(encoder1, decoder1, X, max_length)
                for dataId, outputWord in enumerate(outputWordBatch):
                    toWrite["id"] = idx[dataId]
                    try:
                        #evaluate(encoder1, decoder1, X, max_length)
                        sent = postprocessing(outputWord, embedding)
                        sentWithoutUnk = [a for a in sent if (a != '<unk>')]
                        #print(idx)
                        #print(sentWithoutUnk)
                        joinSentence = sentWithoutUnk[1]
                        quote = 0
                        for i, word in enumerate(sentWithoutUnk[2:], 2):
                            if word == '</s>':
                                break
                            word = word.strip()
                            if (word == "'s" 
                            or sentWithoutUnk[i-1]=="\u00a3"
                            or sentWithoutUnk[i-1]=="-"
                            or sentWithoutUnk[i-1][-1]=="-"
                            or sentWithoutUnk[i-1]=="("
                            or (sentWithoutUnk[i-1]=='"' and quote==1)
                            or word == ")"
                            or word == "-"
                            or word == ","
                            or word == "."
                            or word == ":" 
                            or word == "?"
                            or word == "!"
                            or word == "$"
                            or word == "'"
                            or (word == "m" and len(word)==1)):
                                joinSentence += word
                            elif(word == '"'):
                                if quote == 0:
                                    joinSentence += ' '
                                    joinSentence += word
                                    quote = 1
                                elif quote == 1:
                                    joinSentence += word
                                    quote = 0
                            else:
                                joinSentence += ' '
                                joinSentence += word
                        #joinSentence = ' '.join()
                        print(joinSentence)
                        toWrite["predict"] = joinSentence
                    
                    except KeyboardInterrupt:
                        print('Interrupted')
                        exit(0)
                    except:
                        toWrite = {
                            "id":idx[dataId],
                            "predict":""
                        }
                    json.dump(toWrite, f_predict)
                    f_predict.write("\n")