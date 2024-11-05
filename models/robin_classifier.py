import os
import json
import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, PreTrainedModel, PretrainedConfig
from torchvision.ops import sigmoid_focal_loss


class QASoftMaxAttention(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.0):
        super(QASoftMaxAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_size = hidden_size
        self.scale_factor = 1 / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32))

    def forward(self, sequence_output, attention_mask):
        query = sequence_output * attention_mask.unsqueeze(-1)
        keys = sequence_output * attention_mask.unsqueeze(-1)
        attention = torch.bmm(query, keys.transpose(-1, -2)) * self.scale_factor
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        return torch.bmm(attention, sequence_output)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.2):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class RoBInConfig(PretrainedConfig):

    def __init__(self, model_name, sep_token_id, lambda_p=0.5, pos_weight=1.0, dropout=0.2, qa_attention=False,
                 loss_fn_cls='bce', mlp_classifier=False, mlp_hidden_layer=512, use_peft=False, **kwargs):
        super(PretrainedConfig, self).__init__()
        self.model_name = model_name
        self.sep_token_id = sep_token_id
        self.lambda_p = lambda_p
        self.pos_weight = pos_weight
        self.dropout = dropout
        self.qa_attention = qa_attention
        self.loss_fn_cls = loss_fn_cls
        self.mlp_classifier = mlp_classifier
        self.mlp_hidden_layer = mlp_hidden_layer
        self.use_peft = use_peft
        self.model_type = "FEATURE_EXTRACTION"

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "sep_token_id": self.sep_token_id,
            "lambda_p": self.lambda_p,
            "pos_weight": self.pos_weight,
            "dropout": self.dropout,
            "qa_attention": self.qa_attention,
            "loss_fn_cls": self.loss_fn_cls,
            "mlp_classifier": self.mlp_classifier,
            "mlp_hidden_layer": self.mlp_hidden_layer,
            "use_peft": self.use_peft,
            "model_type": self.model_type
        }

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return cls(**config_dict)

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        config_file_path = os.path.join(save_directory, "config.json")
        config = {
            "model_name": self.model_name,
            "sep_token_id": self.sep_token_id,
            "lambda_p": self.lambda_p,
            "pos_weight": self.pos_weight,
            "dropout": self.dropout,
            "qa_attention": self.qa_attention,
            "loss_fn_cls": self.loss_fn_cls,
            "mlp_classifier": self.mlp_classifier,
            "mlp_hidden_layer": self.mlp_hidden_layer,
            "model_type": "FEATURE_EXTRACTION"
        }
        with open(config_file_path, 'w') as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        config_file_path = os.path.join(path, "config.json")
        with open(config_file_path) as f:
            config = json.load(f)
        return cls(**config)


class RoBInClassifier(PreTrainedModel):
    def __init__(self, config):
        super(RoBInClassifier, self).__init__(config)
        self.config = config
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(p=config.dropout)

        if config.mlp_classifier:
            self.classifier = MLP(self.qa_model.config.hidden_size, config.mlp_hidden_layer, 1, dropout_p=config.dropout)
        else:
            self.classifier = nn.Linear(self.qa_model.config.hidden_size, 1)

        self.qa_attention = QASoftMaxAttention(self.qa_model.config.hidden_size, dropout_p=config.dropout) if config.qa_attention else None

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None, labels=None):

        qa_outputs = self.qa_model(input_ids=input_ids, attention_mask=attention_mask,
                                   start_positions=start_positions, end_positions=end_positions,
                                   output_hidden_states=True)

        loss_qa = qa_outputs.loss if start_positions is not None and end_positions is not None else 0
        sequence_output = qa_outputs.hidden_states[-1]

        mask = torch.zeros(sequence_output.size(0), sequence_output.size(1), dtype=torch.float32,
                           device=sequence_output.device)

        if start_positions is None and end_positions is None:
            start_logits = qa_outputs.start_logits
            end_logits = qa_outputs.end_logits
            start_positions = torch.argmax(start_logits, dim=1)
            end_positions = torch.argmax(end_logits, dim=1)

        for i in range(mask.size(0)):
            if end_positions[i] > start_positions[i]:
                mask[i, start_positions[i]:end_positions[i] + 1] = 1.0
            elif start_positions[i] == 0 and end_positions[i] == 0:
                mask[i, :] = 1.0
                if labels is not None:
                    labels[i] = 0

        if self.qa_attention is None:
            # The mask is used to calculate the pooled_output by taking the mean of the hidden states
            # focusing only on the question and the answer parts of the sequence
            # masked_sequence_output = sequence_output * mask.unsqueeze(-1)

            pooled_output = sequence_output.mean(dim=1)
        else:
            pooled_output = self.qa_attention(sequence_output, mask).mean(dim=1)

        norm_layer = nn.LayerNorm(pooled_output.size()[1:], elementwise_affine=True, device=pooled_output.device)
        sentence_representation = norm_layer(pooled_output)
        logits = self.classifier(self.dropout(sentence_representation))

        loss_cls = 0
        if labels is not None:
            if self.config.loss_fn_cls == 'bce':
                pos_weight = torch.tensor(self.config.pos_weight, device=labels.device)
                loss_fn_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss_cls = loss_fn_cls(logits.view(-1), labels.view(-1))
            elif self.config.loss_fn_cls == 'focal':
                loss_cls = sigmoid_focal_loss(logits.view(-1), labels.view(-1), reduction='mean')

        # _lambda_ = self.config.lambda_p
        # total_loss = (_lambda_ * loss_qa) + ((1 - _lambda_) * loss_cls) if loss_qa and loss_cls else (1 - _lambda_) * loss_cls
        total_loss = loss_qa + loss_cls if loss_qa and loss_cls else loss_cls

        return {
            'loss': total_loss,
            'start_logits': qa_outputs.start_logits,
            'end_logits': qa_outputs.end_logits,
            'logits': logits
        }

    def save_pretrained(self, save_directory, **kwargs):

        os.makedirs(save_directory, exist_ok=True)
        model_file_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_file_path)
        self.config.save_pretrained(save_directory)

        if hasattr(self, 'peft_model'):
            self.peft_model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, save_directory, **kwargs):
        config = RoBInConfig.from_pretrained(save_directory)
        model = cls(config)

        model_file_path = os.path.join(save_directory, "pytorch_model.bin")
        model_state_dict = torch.load(model_file_path, map_location='cpu')
        model.load_state_dict(model_state_dict)

        peft_model_path = os.path.join(save_directory, "peft_model.json")
        if os.path.exists(peft_model_path):
            from peft import get_peft_model, LoraConfig
            peft_config = LoraConfig.from_pretrained(save_directory)
            model.peft_model = get_peft_model(model, peft_config)
            model.peft_model.load_state_dict(torch.load(os.path.join(save_directory, "peft_model.bin")))

        return model
