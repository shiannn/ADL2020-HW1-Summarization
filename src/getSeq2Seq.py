import argparse
import logging
import os
import json
import pickle
import sys
from pathlib import Path
from utils import Tokenizer, Embedding
from dataset import Seq2SeqDataset
from tqdm import tqdm

CONFIG = 'config/seq2seqConfig.json'
ENBEDDINT_NAME = 'seq2seqEmbedding/embedding.pkl'
def main(argv):
    with open(CONFIG, 'r') as f:
        config = json.load(f)

    # loading datasets from jsonl files
    testName = argv[1]
    with open(testName, 'r') as f:
        test = [json.loads(line) for line in f]

    tokenizer = Tokenizer(lower=config['lower_case'])

    logging.info('Loading embedding...')
    with open(ENBEDDINT_NAME, 'rb') as f:
        embedding = pickle.load(f)

    tokenizer.set_vocab(embedding.vocab)

    logging.info('Creating test dataset...')
    create_seq2seq_dataset(
        process_samples(tokenizer, test), 'testSeq2Seq.pkl', config, 
            tokenizer.pad_token_id
    )


def process_samples(tokenizer, samples):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    processeds = []
    for sample in tqdm(samples):
        processed = {
            'id': sample['id'],
            'text': tokenizer.encode(sample['text']) + [eos_id],
        }
        if 'summary' in sample:
            processed['summary'] = (
                [bos_id]
                + tokenizer.encode(sample['summary'])
                + [eos_id]
            )
        processeds.append(processed)

    return processeds


def create_seq2seq_dataset(samples, save_path, config, padding=0):
    dataset = Seq2SeqDataset(
        samples, padding=padding,
        max_text_len=config.get('max_text_len') or 300,
        max_summary_len=config.get('max_summary_len') or 80
    )
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('output_dir', type=Path,
                        help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #args = _parse_args()
    if len(sys.argv) != 2:
        print('usage: python3 getSeq2Seq testDataPath')
        exit(0)
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(sys.argv)
