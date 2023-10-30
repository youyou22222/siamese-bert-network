
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class SiameseClassificationBERT(nn.Module):
    def __init__(self, num_classes, model_name_or_path='./pretrained_models'):
        super(SiameseClassificationBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.fc = nn.Linear(self.bert.config.hidden_size*2, num_classes)

    def forward_one(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output

    def forward(self, input1, input2):
        input_ids_1, attention_mask_1 = input1['input_ids'], input1['attention_mask']
        input_ids_2, attention_mask_2 = input2['input_ids'], input2['attention_mask']

        output1 = self.forward_one(input_ids_1, attention_mask_1)
        output2 = self.forward_one(input_ids_2, attention_mask_2)

        concatenated = torch.cat((output1, output2), 1)
        logits = self.fc(concatenated)
        return logits
