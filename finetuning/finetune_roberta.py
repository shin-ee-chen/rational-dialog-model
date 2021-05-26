import datasets
import torch
# from torch.utils.data import DataLoader
# import pytorch_lightning as pl

from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling

import os
import math
import argparse


def tokenize_daily_dialog(dataset, tokenizer):
    def tokenize_function(examples):
        # note that gpt2 does't have sep_token
        dialogues = [tokenizer.sep_token.join(dialog) for dialog in examples["dialog"]]
        return tokenizer(dialogues)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["dialog", 'emotion', 'act'])
    return tokenized_datasets


def group_texts(examples, block_size=128):
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


def finetune_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint, use_fast=True)
    dataset = datasets.load_dataset(config.fine_tune_dataset)

    if config.fine_tune_dataset == "daily_dialog":
        tokenized_dataset = tokenize_daily_dialog(dataset, tokenizer)
    
    lm_datasets = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
        )

    training_args = TrainingArguments(
        "logs",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=config.max_epochs,
        load_best_model_at_end=True,
    )
    # training_args.device = "cpu" if not torch.cuda.is_available() else "cuda"

    if config.model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(config.model_checkpoint)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
            )

    elif config.model_type == "masked":
        model = AutoModelForMaskedLM.from_pretrained(config.model_checkpoint)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
            data_collator=data_collator,
            )

    trainer.train()
    
    path = os.path.join(config.save_path, config.fine_tune_dataset,
                        config.model_checkpoint+"_"+config.model_type)
    trainer.save_model(path)
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")



if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--model_checkpoint', default="roberta-base", type=str,
                        help='name of the pretrained LM model')
    parser.add_argument('--fine_tune_dataset', default="daily_dialog", type=str,
                        help='dataset used for fine-tuning')

    parser.add_argument('--model_type', default="causal", type=str,
                        help='causal or masked model')
    parser.add_argument('--max_epochs', default=4, type=int,

                        help='max epochs for fine-tuning')                 
    parser.add_argument('--save_path', default="saved_models/roberta-daily_dialog-model", type=str,
                        help='path to save fine-tuned models') 
    args = parser.parse_args()
    finetune_model(args)



    





