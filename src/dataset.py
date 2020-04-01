import torch
from torch.utils.data import Dataset
from utils import pad_to_len


class Seq2SeqDataset(Dataset):
    def __init__(self, data, padding=0,
                 max_text_len=300, max_summary_len=80):
        self.data = data
        self.padding = padding
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'id': self.data[index]['id'],
            'text': self.data[index]['text'][:self.max_text_len],
            'summary': self.data[index]['summary'][:self.max_summary_len],
            'len_text': len(self.data[index]['text']),
            'len_summary': len(self.data[index]['summary']),
            'attention_mask': [True] * min(len(self.data[index]['text']),
                                           self.max_text_len)
        }

    def collate_fn(self, samples):
        batch = {}
        for key in ['id', 'len_text', 'len_summary']:
            batch[key] = [sample[key] for sample in samples]

        for key in ['text', 'summary', 'attention_mask']:
            to_len = max([len(sample[key]) for sample in samples])
            padded = pad_to_len(
                [sample[key] for sample in samples], to_len, self.padding
            )
            batch[key] = torch.tensor(padded)

        return batch


class SeqTaggingDataset(Seq2SeqDataset):
    ignore_idx = -100

    def __getitem__(self, index):
        sample = self.data[index]
        instance = {
            'id': sample['id'],
            'text': sample['text'][:self.max_text_len],
            'sent_range': sample['sent_range']
        }
        if 'label' in sample:
            instance['label'] = sample['label'][:self.max_text_len]
        return instance

    def collate_fn(self, samples):
        batch = {}
        for key in ['id', 'sent_range']:
            batch[key] = [sample[key] for sample in samples]

        for key in ['text', 'label']:
            if any(key not in sample for sample in samples):
                continue
            to_len = max([len(sample[key]) for sample in samples])
            padded = pad_to_len(
                [sample[key] for sample in samples], to_len,
                self.padding if key != 'label' else SeqTaggingDataset.ignore_idx
            )
            batch[key] = torch.tensor(padded)

        return batch
