import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from data_models.data_qa import train_data_preprocess, preprocess_validation_examples


class RoBInDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_seq_length=512, stride=128, mode="train"):
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.stride = stride
        self.class_dict = {'UNCLEAR': 0, 'HIGH': 0, 'LOW': 1}

        if self.mode == "train":
            # sampling
            self.dataset = dataset["train"]
            self.data = self.dataset.map(train_data_preprocess,
                                         fn_kwargs={"tokenizer": self.tokenizer,
                                                    "stride": self.stride,
                                                    "max_length": self.max_seq_length},
                                         batched=True,
                                         remove_columns=dataset["train"].column_names)

        elif self.mode == 'validate':
            self.dataset = dataset["validation"]
            self.data = self.dataset.map(train_data_preprocess,
                                         fn_kwargs={"tokenizer": self.tokenizer,
                                                    "stride": self.stride,
                                                    "max_length": self.max_seq_length},
                                         batched=True, remove_columns=dataset["validation"].column_names,
                                         )
        else:
            self.dataset = dataset["validation"]
            self.data = self.dataset.map(preprocess_validation_examples,
                                         fn_kwargs={"tokenizer": self.tokenizer,
                                                    "stride": self.stride,
                                                    "max_length": self.max_seq_length},
                                         batched=True, remove_columns=dataset["validation"].column_names,
                                         )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out = {}
        example = self.data[idx]
        out['input_ids'] = torch.tensor(example['input_ids'])
        out['attention_mask'] = torch.tensor(example['attention_mask'])

        if self.mode == "train" or self.mode == 'validate':
            out['labels'] = torch.tensor(self.class_dict[self.dataset['label'][example['overflow_to_sample_mapping']]],
                                         dtype=torch.float)
            out['start_positions'] = torch.unsqueeze(torch.tensor(example['start_positions']), dim=0)
            out['end_positions'] = torch.unsqueeze(torch.tensor(example['end_positions']), dim=0)
        else:
            out['labels'] = [torch.tensor(self.class_dict[row['label']], dtype=torch.float)
                             for row in self.dataset if row['id'] == example['base_id']][0]

        return out
