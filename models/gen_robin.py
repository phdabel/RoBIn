import os
import json
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartPreTrainedModel, BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput


class GenerativeRoBInConfig(BartConfig):

    def __init__(self, model_name, lambda_p=0.5, pos_weight=1.0, dropout=0.2, **kwargs):
        super(BartConfig, self).__init__()
        self.model_name = model_name
        self.lambda_p = lambda_p
        self.pos_weight = pos_weight
        self.dropout = dropout
        self.model_type = "FEATURE_EXTRACTION"

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "lambda_p": self.lambda_p,
            "pos_weight": self.pos_weight,
            "dropout": self.dropout,
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
            "lambda_p": self.lambda_p,
            "pos_weight": self.pos_weight,
            "dropout": self.dropout,
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


class GenerativeRoBInClassifier(BartPreTrainedModel):

    def __init__(self, config, **kwargs):
        super(GenerativeRoBInClassifier, self).__init__(config)
        self.config = config
        self.model = BartForConditionalGeneration.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.model.config.d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None, **kwargs):
        # Geração de evidências
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                             labels=decoder_input_ids, output_hidden_states=True)

        # Loss de QA (para geração de evidências)
        loss_qa = outputs.loss if decoder_input_ids is not None else None

        hidden_states = outputs.decoder_hidden_states[-1]
        pooled_output = hidden_states.mean(dim=1)

        norm_layer = nn.LayerNorm(pooled_output.size()[1:], elementwise_affine=True, device=pooled_output.device)
        sentence_representation = norm_layer(pooled_output)
        logits = self.classifier(self.dropout(sentence_representation))

        # Loss de classificação de viés
        loss_cls = None
        if labels is not None:
            # computing weights based on the number of samples in the dataset (hard coded)
            # HIGH_UNCLEAR = 1183 + 1654
            # LOW = 7973
            # TOTAL_SAMPLES = HIGH_UNCLEAR + LOW
            # NUM_CLASSES = 2
            # WEIGHT_HIGH_UNCLEAR = TOTAL_SAMPLES / (HIGH_UNCLEAR * NUM_CLASSES)
            # WEIGHT_LOW = TOTAL_SAMPLES / (LOW * NUM_CLASSES)
            # weights = torch.tensor([WEIGHT_HIGH_UNCLEAR, WEIGHT_LOW], device=labels.device)

            pos_weight = torch.tensor(self.config.pos_weight, device=labels.device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_cls = loss_fn(logits.view(-1), labels.float().view(-1))

        # Combinar perdas
        total_loss = loss_qa + loss_cls if loss_qa is not None and loss_cls is not None else loss_qa if loss_qa is not None else loss_cls

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions
        ), {
            "classifier_logits": logits
        }

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        model_file_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_file_path)
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        config = GenerativeRoBInConfig.from_pretrained(path)
        model = cls(config)
        model_file_path = os.path.join(path, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_file_path))
        return model

    def generate(self, input_ids, attention_mask, **kwargs):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def predict(self, input_ids, attention_mask, **kwargs):
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                      num_beams=1,
                                      output_hidden_states=True, return_dict_in_generate=True, output_scores=True,
                                      **kwargs)


        decoder_hidden_states = outputs.decoder_hidden_states
        final_hidden_states = []
        for step_hidden_states in decoder_hidden_states[1:]:
            # reshape last layer hidden states
            step_hidden_states = step_hidden_states[-1].view(-1, step_hidden_states[-1].size(-1))
            final_hidden_states.append(step_hidden_states)

        hidden_states = torch.stack(final_hidden_states)
        hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states.mean(dim=1)
        norm_layer = nn.LayerNorm(hidden_states.size()[1:], elementwise_affine=True, device=hidden_states.device)
        sentence_representation = norm_layer(hidden_states)
        logits = self.classifier(self.dropout(sentence_representation))


        return {
            "logits": logits,
            "prediction": self.sigmoid(logits),
            "sequences": outputs.sequences,
            "scores": outputs.scores
        }
