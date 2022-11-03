from transformers import BertModel
import torch.nn as nn


class BertUncasedModel(nn.Module):
    def __init__(self, bert_type, drop_rate):
        super(BertUncasedModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_type)
        self.drop = nn.Dropout(drop_rate)
        self.out = nn.Linear(768, 1)

    def forward(self, input):
        x = self.bert(**input).pooler_output
        return self.out(self.drop(x))
