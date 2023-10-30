#--encoding:utf-8--
# author:jiayawei
"""
the dataset for finetune, the data format is organized as follows:
(sentence1, sentence2, label)
where setence1 is the user profile keywords, sentence2 is the job keywords, label is the score
"""

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class SentencePairDataset(Dataset):
    def __init__(self, data, max_seq_length=128):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained('chinese-roberta-wwm-ext')
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def _tokenize(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = tokens[:self.max_seq_length - 2]
        return tokens

    def _get_input_features(self, sentence1, sentence2):
        tokens1 = self._tokenize(sentence1)
        tokens2 = self._tokenize(sentence2)

        input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]'])
        attention_mask = [1] * len(input_ids)

        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        attention_mask += [0] * padding_length

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }

    def __getitem__(self, idx):
        sentence1, sentence2, label = self.data[idx]
        input_features = self._get_input_features(sentence1, sentence2)
        return input_features, torch.tensor(label, dtype=torch.long)
