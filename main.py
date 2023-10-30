# -*- coding: utf-8 -*-
# author: jiayawei
"""
the main function for trainning a siamese bert model
"""
import torch
import argparse
import json
import logging
import torch.nn as nn
from dataset import SentencePairDataset
from bert import SiameseClassificationBERT
from torch.utils.data import DataLoader
from train import train_model_with_adamw, evaluate_model
from transformers import AdamW, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser("Finetune bert")
# get logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def read_config(config_file):
    with open(config_file) as inf:
        config = json.load(inf)
    return config


def load_data(filepath):
    # the file format is organized as follows: sentence1 \t sentence2 \t label
    data = []
    with open(filepath) as inf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            sentence1, sentence2, label = line.split("\t")
            data.append((sentence1, sentence2, label))
    return SentencePairDataset(data)


def main():
    parser.add_argument("--config_file", type=str, required=True, help="config file path for training model")
    args = parser.parse_args()

    config = read_config(args.config_file)
    num_classes = config.num_classes
    train_file = config.train_file
    val_file = config.val_file
    lr = config.lr
    model_path = config.model_path
    logger_path = config.logger_path
    num_epochs = config.num_epochs
    patience = config.patience

    # set logger path
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    train_data = load_data(train_file)
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_data = load_data(val_file)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

    total_steps = len(train_dataloader) * num_epochs

    model = SiameseClassificationBERT(num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to("cuda")

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()

    train_model_with_adamw(model, train_dataloader, criterion, optimizer, scheduler,
                           num_epochs=num_epochs,
                           validation_dataloader=val_dataloader,
                           patience=patience, logger=logger, model_path=model_path)


if __name__ == "__main__":
    main()