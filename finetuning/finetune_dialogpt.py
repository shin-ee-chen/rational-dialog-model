'''
Trains DialoGPT on the daily dialog set.
'''
#template: https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb
import torch
import argparse

from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def tokenize_daily_dialog(dataset, tokenizer):
    def tokenize_function(examples):
        # note that gpt2 does't have sep_token
        dialogues = [tokenizer.eos_token.join(dialog) for dialog in examples['dialog']]
        return tokenizer(dialogues)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['dialog', 'emotion', 'act'])
    return tokenized_datasets

def tokenize_mutual_friends(dataset, tokenizer):
    def tokenize_function(examples):
        # note that gpt2 does't have sep_token
        data_messages = [filter(lambda x: x != '', event['data_messages']) for event in examples['events']]
        dialogues = [tokenizer.eos_token.join(dialog) for dialog in data_messages]
        return tokenizer(dialogues)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['events', 'agents', 'outcome_reward', 'scenario_alphas', 'scenario_attributes', 'scenario_kbs', 'scenario_uuid', 'uuid'])
    return tokenized_datasets

def group_texts(dialogues):
    #block_size = tokenizer.model_max_length
    block_size = 128 #https://github.com/microsoft/DialoGPT/issues/34
    # Concatenate all texts.
    concatenated_dialogues = {k: sum(dialogues[k], []) for k in dialogues.keys()}
    total_length = len(concatenated_dialogues[list(dialogues.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_dialogues.items()
    }
    result["labels"] = result["input_ids"].copy() #The model of the 🤗 Transformers library apply the shifting to the right, so we don't need to do it manually.
    return result

class CompleteDialogueCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        chat_history_ids = None
        sentence = 'Believe it or not, I can do 30 push-ups a minute.'
        print('Input: ' + sentence)
        sentence += tokenizer.eos_token #add separation token
        for step in range(3):
            #encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
            #append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
            #generated a response while limiting the total chat history to 1000 tokens,
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            #pretty print last ouput tokens from bot
            try:
                print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
            except:
                #UnicodeEncodeError: 'latin-1' codec can't encode character '\u2019' in position 21: ordinal not in range(256)
                print("DialoGPT: [can't generate response!]")

def finetune_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    datasets = load_dataset(config.fine_tune_dataset)
    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

    if config.fine_tune_dataset == "daily_dialog":
        tokenized_datasets = tokenize_daily_dialog(datasets, tokenizer)
    elif config.fine_tune_dataset == "mutual_friends":
        tokenized_datasets = tokenize_mutual_friends(datasets, tokenizer)

    print(tokenized_datasets["train"][0])

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=config.batch_size,
        num_proc=4,
    )

    print(tokenizer.decode(lm_datasets["train"][0]["input_ids"]))

    model = AutoModelForCausalLM.from_pretrained(config.model_checkpoint).to(device)
    #model.resize_token_embeddings(len(tokenizer)) #because we added special tokens

    training_args = TrainingArguments(
        config.save_path,
        do_train= True,
        do_eval= True,
        evaluation_strategy="epoch", #steps or epoch
        learning_rate=1e-3,
        #max_steps=2, #only for debugging
        num_train_epochs=config.max_epochs,
        load_best_model_at_end = True,
        save_strategy='epoch',
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        callbacks=[]#[CompleteDialogueCallback]
    )

    trainer.train()
    trainer.save_model()

    #evaluate model
    eval_results = trainer.evaluate()
    trainer.save_metrics('eval', eval_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--model_checkpoint', default="microsoft/DialoGPT-small", type=str,
                        help='name of the pretrained LM model')
    parser.add_argument('--fine_tune_dataset', default="daily_dialog", type=str,
                        help='dataset used for fine-tuning')
    parser.add_argument('--max_epochs', default=3, type=int,
                        help='max epochs for fine-tuning')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='batch size to be used for fine-tuning')                  
    parser.add_argument('--save_path', default="dialoGPT_dailyDialog_model", type=str,
                        help='path to save fine-tuned models') 

    args = parser.parse_args()
    finetune_model(args)