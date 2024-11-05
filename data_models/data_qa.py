import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset


def train_data_preprocess(examples, tokenizer, max_length, stride):
    """
    generate start and end indexes of answer in context
    """
    def find_context_start_end_index(sequence_ids):
        """
        returns the token index in which context starts and ends
        """
        token_idx = 0
        while sequence_ids[token_idx] != 1:  # means its special tokens or tokens of question
            token_idx += 1  # loop only break when context starts in tokens
        context_start_idx = token_idx

        while sequence_ids[token_idx] == 1:
            token_idx += 1
        context_end_idx = token_idx - 1
        return context_start_idx, context_end_idx

    questions = [q.strip() for q in examples["question"]]
    context = examples["context"]
    answers = examples["answers"]

    inputs = tokenizer(
        questions,
        context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,  # returns id of base context
        return_offsets_mapping=True,  # returns (start_index,end_index) of each token
        padding="max_length"
    )

    start_positions = []
    end_positions = []

    for i, mapping_idx_pairs in enumerate(inputs['offset_mapping']):
        context_idx = inputs['overflow_to_sample_mapping'][i]

        # from main context
        answer = answers[context_idx]
        answer_start_char_idx = answer['answer_start'][0]
        answer_end_char_idx = answer_start_char_idx + len(answer['text'][0].strip())

        # now we have to find it in sub contexts
        tokens = inputs['input_ids'][i]
        sequence_ids = inputs.sequence_ids(i)

        # finding the context start and end indexes wrt sub context tokens
        context_start_idx, context_end_idx = find_context_start_end_index(sequence_ids)

        # if the answer is not fully inside context label it as (0,0)
        # starting and end index of the character of full context text
        context_start_char_index = mapping_idx_pairs[context_start_idx][0]
        context_end_char_index = mapping_idx_pairs[context_end_idx][1]

        # If the answer is not fully inside the context, label is (0, 0)
        if (context_start_char_index > answer_start_char_idx) or (
                context_end_char_index < answer_end_char_idx):
            start_positions.append(0)
            end_positions.append(0)

        else:

            # else its start and end token positions
            # here idx indicates index of token
            idx = context_start_idx
            while idx <= context_end_idx and mapping_idx_pairs[idx][0] <= answer_start_char_idx:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end_idx
            while idx >= context_start_idx and mapping_idx_pairs[idx][1] > answer_end_char_idx:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples, tokenizer, max_length, stride):
    """
    preprocessing validation data
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs["overflow_to_sample_mapping"]

    base_ids = []

    for i in range(len(inputs["input_ids"])):
        # take the base id (ie in cases of overflow happens we get base id)
        base_context_idx = sample_map[i]
        base_ids.append(examples["id"][base_context_idx])

        # sequence id indicates the input. 0 for first input and 1 for second input
        # and None for special tokens by default
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        # for Question tokens provide offset_mapping as None
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["base_id"] = base_ids

    return inputs


class QADataset(Dataset):

    def __init__(self, dataset, tokenizer, max_seq_length=512, stride=128, mode="train"):
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.stride = stride

        if self.mode == "train":
            # sampling
            self.dataset = dataset["train"]
            self.data = self.dataset.map(train_data_preprocess,
                                         fn_kwargs={"tokenizer": self.tokenizer, "stride": self.stride,
                                                    "max_length": self.max_seq_length},
                                         batched=True,
                                         remove_columns=dataset["train"].column_names,
                                         )
        else:
            self.dataset = dataset["validation"]
            self.data = self.dataset.map(preprocess_validation_examples,
                                         fn_kwargs={"tokenizer": self.tokenizer, "stride": self.stride,
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

        if self.mode == "train":
            out['start_positions'] = torch.unsqueeze(torch.tensor(example['start_positions']), dim=0)
            out['end_positions'] = torch.unsqueeze(torch.tensor(example['end_positions']), dim=0)

        return out
