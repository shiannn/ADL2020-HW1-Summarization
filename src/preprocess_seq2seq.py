import argparse
import logging
import os
import json
import pickle
from pathlib import Path
from utils import Tokenizer, Embedding
from dataset import Seq2SeqDataset
from tqdm import tqdm


def main(args):
    with open(args.output_dir / 'config.json') as f:
        config = json.load(f)

    # loading datasets from jsonl files
    with open(config['train']) as f:
        train = [json.loads(line) for line in f]
    with open(config['valid']) as f:
        valid = [json.loads(valid) for valid in f]
    with open(config['test']) as f:
        test = [json.loads(line) for line in f]

    logging.info('Collecting documents...')
    documents = (
        [sample['text'] for sample in train]
        + [sample['summary'] for sample in train]
        + [sample['text'] for sample in valid]
        + [sample['text'] for sample in test]
    )

    logging.info('Collecting words in documents...')
    tokenizer = Tokenizer(lower=config['lower_case'])
    words = tokenizer.collect_words(documents)

    logging.info('Loading embedding...')
    embedding = Embedding(config['embedding'], words=words)
    with open(args.output_dir / 'embedding.pkl', 'wb') as f:
        pickle.dump(embedding, f)

    tokenizer.set_vocab(embedding.vocab)

    logging.info('Creating train dataset...')
    create_seq2seq_dataset(
        process_samples(tokenizer, train),
        args.output_dir / 'train.pkl', config,
        tokenizer.pad_token_id
    )
    logging.info('Creating valid dataset...')
    create_seq2seq_dataset(
        process_samples(tokenizer, valid),
        args.output_dir / 'valid.pkl', config,
        tokenizer.pad_token_id
    )
    logging.info('Creating test dataset...')
    create_seq2seq_dataset(
        process_samples(tokenizer, test),
        args.output_dir / 'test.pkl', config,
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
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)
