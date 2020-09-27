import argparse
import json
import logging
import os
import sys
import pickle
from pathlib import Path
from typing import Iterable

from dataset import SeqTaggingDataset
from tqdm import tqdm
from utils import Tokenizer, Embedding

CONFIG = 'config/seq2tagConfig.json'
ENBEDDINT_NAME = 'seq2tagEmbedding/embedding.pkl'
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
    create_seq_tag_dataset(
        process_seq_tag_samples(tokenizer, test),
        'testSeq2Tag.pkl', config,
        tokenizer.pad_token_id
    )


def process_seq_tag_samples(tokenizer, samples):
    processeds = []
    for sample in tqdm(samples):
        if not sample['sent_bounds']:
            continue
        processed = {
            'id': sample['id'],
            'text': tokenizer.encode(sample['text']),
            'sent_range': get_tokens_range(tokenizer, sample)
        }

        if 'extractive_summary' in sample:
            label_start, label_end = processed['sent_range'][sample['extractive_summary']]
            processed['label'] = [
                1 if label_start <= i < label_end else 0
                for i in range(len(processed['text']))
            ]
            assert len(processed['label']) == len(processed['text'])
        processeds.append(processed)
    return processeds


def get_tokens_range(tokenizer,
                     sample) -> Iterable:
    ranges = []
    token_start = 0
    for char_start, char_end in sample['sent_bounds']:
        sent = sample['text'][char_start:char_end]
        tokens_in_sent = tokenizer.tokenize(sent)
        token_end = token_start + len(tokens_in_sent)
        ranges.append((token_start, token_end))
        token_start = token_end
    return ranges


def create_seq_tag_dataset(samples, save_path, config, padding=0):
    dataset = SeqTaggingDataset(
        samples, padding=padding,
        max_text_len=config.get('max_text_len') or 300,
    )
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #args = _parse_args()
    if len(sys.argv) != 2:
        print('usage: python3 getSeq2Tag testDataPath')
        exit(0)
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(sys.argv)
