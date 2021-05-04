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
from daily_dialog.callbacks import FinishSentenceCallback, ReshuffleDatasetCallback
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_checkpoint = 'microsoft/DialoGPT-small'
fine_tune_dataset = 'daily_dialog'
max_epochs = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

datasets = load_dataset(fine_tune_dataset)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def tokenize_function(examples):
    dialogues = [(tokenizer.eos_token).join(dialog) for dialog in examples["dialog"]]
    return tokenizer(dialogues)

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["dialog", "emotion", "act"])

print(tokenized_datasets["train"][1])

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
    #import pdb; pdb.set_trace()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=256,
    num_proc=4,
)

print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))

model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)

training_args = TrainingArguments(
    "dailoGPT_model",
    do_train= True,
    do_eval= True,
    evaluation_strategy="steps",
    learning_rate=1e-3,
    weight_decay=0.01,
    #max_steps=10, #only for debugging
    num_train_epochs=max_epochs,
    load_best_model_at_end = True
)
class CompleteDialogueCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        chat_history_ids = None
        sentence = ('Believe it or not, I can do 30 push-ups a minute.' + tokenizer.eos_token)
        print('[START] ' + sentence)
        for step in range(4):
            #encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
            #append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
            #generated a response while limiting the total chat history to 1000 tokens, 
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            #pretty print last ouput tokens from bot
            print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    callbacks=[CompleteDialogueCallback]
)

trainer.train()
trainer.save_model()

#evaluate model
eval_results = trainer.evaluate()
trainer.save_metrics('eval', eval_results)