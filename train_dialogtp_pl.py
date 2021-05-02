'''
Trains a language model on the daily dialog set.
'''
import datasets
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from daily_dialog.CLMDataset import CLMDataset
from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer
# from modules.LanguageModels.LstmLanguageModel import LSTMLM
# from modules.LanguageModels.LanguageModelPL import LMPL
# from daily_dialog.callbacks import FinishSentenceCallback, ReshuffleDatasetCallback
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_checkpoint = 'microsoft/DialoGPT-small'
fine_tune_dataset = 'daily_dialog'
max_epochs = 3

datasets = load_dataset(fine_tune_dataset)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def tokenize_function(examples):
    dialogues = ['[SEP]'.join(dialog) for dialog in examples["dialog"]]
    return tokenizer(dialogues)

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["dialog"])
block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

training_args = TrainingArguments(
    "logs",
    learning_rate=1e-3,
    weight_decay=0.01,
    num_train_epochs=max_epochs,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"]
)

trainer.train()