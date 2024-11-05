import os
import json
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset


class ClassificationDataset(Dataset):

    def __init__(self, ids, sentences, labels, tokenizer, max_length):
        self.ids = ids
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        _id_ = self.ids[idx]
        sentence = self.sentences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'id': _id_,
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
