import sys
import json
import torch
from attention.AttentionEncoderBi import AttnEncoderRNN, hidden_size
from attention.AttentionDecoderCosBi import AttnDecoderRNN
import pickle
from dataset import Seq2SeqDataset
import torch.utils.data as Data
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SOS_token = 1
EOS_token = 2
BATCH_SIZE = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor2word(tns, embedding):
    words = [embedding.vocab[a] for a in tns]
    return words

def list2word(wordlist, embedding):
    words = []
    for a in wordlist:
        if isinstance(a, int):
            words.append(embedding.vocab[a])
        else:
            break
    return words

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

    decoder_input = torch.tensor([SOS_token]* BATCH_SIZE, device=device)

    decoder_hidden = encoder_hidden
    #print('decoder_hidden', decoder_hidden)

    decoded_words = [[] for i in range(BATCH_SIZE)]
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        
        #print('decoder_attention.data.shape', decoder_attention.data.shape)
        decoder_attentions[di] = decoder_attention.data

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
    return decoded_words, decoder_attentions

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('usage: python3 predict.py encoder.pt decoder.pt embedding.pkl TestingData.pkl')
        exit(0)

    encoderName = sys.argv[1]
    decoderName = sys.argv[2]
    embeddingName = sys.argv[3]
    testDataName = sys.argv[4]

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

    max_length = max(maxTextLen, maxSummaryLen)
    #max_length = maxSummaryLen
    #iterLoader = iter(loader)
    with torch.no_grad():
        for cnt,dataInstance in enumerate(loader):
            if (dataInstance['len_text'][0] == 27):
                X = dataInstance['text'].to(device)
                output_words, attentions = evaluate(encoder1, decoder1, X, max_length)
                #print(attentions)
                #print(attentions.shape)
                #plt.matshow(attentions.numpy())
                output_word = output_words[0]
                summaryLen = 0
                for i, a in enumerate(output_word):
                    if a == '<EOS>':
                        summaryLen = i
                        break
                input_text = X[0]                
                textLen = 0
                for i, a in enumerate(input_text):
                    if a == 0:
                        textLen = i
                        break
                inputWord = tensor2word(input_text , embedding)
                outputWord = list2word(output_word , embedding)
                #print(inputWord)
                #print(outputWord)
                inputWord = [a for a in inputWord if (a != '<s>' and a != '<\s>' and a != '<pad>')]
                outputWord = [a for a in outputWord if (a != '<s>' and a != '<\s>' and a != '<pad>')]
                print('inputWord', inputWord)
                print('outputWord', outputWord)
                #print('textLen', textLen)
                visMatrix = attentions.numpy()

                fig = plt.figure()
                ax = fig.add_subplot(111)
                #print('len(inputWord)', len(inputWord))
                #print(visMatrix.shape)
                #print(type(visMatrix))
                todraw = visMatrix[:len(outputWord),:len(inputWord)]
                todraw = softmax(todraw, axis=1)
                #print(todraw)
                #print(todraw.shape)
                cax = ax.matshow(todraw, cmap='bone')
                fig.colorbar(cax)

                ax.set_xticklabels([''] + inputWord + ['<EOS>'], rotation=90)
                ax.set_yticklabels([''] + outputWord)

                # Show label at every tick
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

                #plt.matshow(visMatrix[:summaryLen][:textLen])
                fig.savefig('visAttention.png')
                """
                #print('dataInstance', dataInstance)
                try:
                    dataInstance = next(iterLoader)
                    #print('dataInstance', dataInstance)
                except:
                    print('iter error')
                    exit(0)
                """
                exit(0)