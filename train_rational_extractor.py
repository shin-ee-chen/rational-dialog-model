"""
File to train a language model.
"""

import argparse

import torch

from utils.parse_config import parse_config_RE, parse_config

parser = argparse.ArgumentParser(description='Train a language model')

parser.add_argument('--config', type=str, default="configs/dialoGPT_dailyDialog_RE_config.yml",
                    help='path to the config')

args = parser.parse_args()

config_ref = args.config

parameters = parse_config(config_ref)

device = "cuda" if torch.cuda.is_available() else "cpu"

trainer = parameters["trainer"]

model = parameters["lightning_language_model"]

dataloader_train = parameters["dataloader_train"]
dataloader_test = parameters["dataloader_test"]

trainer.fit(model, dataloader_train, dataloader_test)

# Next we save the language model
