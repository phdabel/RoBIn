import os
import json
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, PretrainedConfig, PreTrainedModel


class MLPConfig(PretrainedConfig):

    def __init__(self, model_name, num_classes=2, pos_weight=1.0, dropout=0.2, loss_fn_cls='bce', mlp_hidden_layer=512):
        super(PretrainedConfig, self).__init__()
        self.model_name = model_name
        self.pos_weight = pos_weight
        self.dropout = dropout
        self.mlp_hidden_layer = mlp_hidden_layer
        self.loss_fn_cls = loss_fn_cls
        self.model_type = "SENTENCE_CLASSIFICATION"
        self.num_classes = num_classes

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "pos_weight": self.pos_weight,
            "dropout": self.dropout,
            "mlp_hidden_layer": self.mlp_hidden_layer,
            "loss_fn_cls": self.loss_fn_cls,
            "num_classes": self.num_classes
        }

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return cls(**config_dict)

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        config_file_path = os.path.join(save_directory, "config.json")
        config = {
            "model_name": self.model_name,
            "pos_weight": self.pos_weight,
            "dropout": self.dropout,
            "mlp_hidden_layer": self.mlp_hidden_layer,
            "loss_fn_cls": self.loss_fn_cls,
            "num_classes": self.num_classes
        }
        with open(config_file_path, 'w') as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        config_file_path = os.path.join(path, "config.json")
        with open(config_file_path) as f:
            config = json.load(f)
        return cls(**config)


class MLPClassifier(PreTrainedModel):
    def __init__(self, config):
        super(MLPClassifier, self).__init__(config)

        self.config = config
        self.encoder = AutoModelForSequenceClassification.from_pretrained(config.model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer1 = nn.Linear(self.encoder.roberta.encoder.config.hidden_size, config.mlp_hidden_layer)
        self.layer2 = nn.Linear(config.mlp_hidden_layer, config.mlp_hidden_layer)
        self.layer3 = nn.Linear(config.mlp_hidden_layer, config.mlp_hidden_layer)
        self.linear = nn.Linear(config.mlp_hidden_layer, config.num_classes)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        sequence_output = outputs.hidden_states[-1][:, 0, :]
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.dropout(self.relu(self.layer1(sequence_output)))
        sequence_output = self.dropout(self.relu(self.layer2(sequence_output)))
        sequence_output = self.dropout(self.relu(self.layer3(sequence_output)))
        logits = self.linear(sequence_output)

        loss = None
        if labels is not None:
            if self.config.loss_fn_cls == 'bce':
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.config.pos_weight))
                labels = labels.unsqueeze(-1)
                labels = torch.cat((labels, 1 - labels), dim=1)
                loss = loss_fct(logits, labels.float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))

        return loss, logits

    def save_pretrained(self, save_directory, **kwargs):

        os.makedirs(save_directory, exist_ok=True)
        model_file_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_file_path)
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, save_directory, **kwargs):
        config = MLPConfig.from_pretrained(save_directory)
        model = cls(config)

        model_file_path = os.path.join(save_directory, "pytorch_model.bin")
        model_state_dict = torch.load(model_file_path, map_location='cpu')
        model.load_state_dict(model_state_dict)

        return model
