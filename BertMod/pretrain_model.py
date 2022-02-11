import torch
import torch.nn as nn
from bert_transformers.modeling_bert import (BertModel, )


class PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = PredictionHeadTransform(config)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = LMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class PretrainBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model = BertModel(self.config)

        self.decoder = MLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, label_ids):
        hidden_states = self.model(input_ids).last_hidden_state
        logits = self.decoder(hidden_states)

        loss = self.loss_fct(logits.view(-1, self.config.vocab_size), label_ids.view(-1))
        return loss