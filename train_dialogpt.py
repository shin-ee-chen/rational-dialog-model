'''
Trains DialoGPT on the daily dialog set.
'''
import torch

from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_checkpoint = 'microsoft/DialoGPT-small'
fine_tune_dataset = 'daily_dialog'
max_epochs = 10
batch_size = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

datasets = load_dataset(fine_tune_dataset)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(dialogues):
    dialogues = [(tokenizer.eos_token).join(dialog) for dialog in dialogues["dialog"]]
    return tokenizer(dialogues)

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["dialog", "emotion", "act"])

print(tokenized_datasets["train"][0])

#block_size = tokenizer.model_max_length
block_size = 128 #https://github.com/microsoft/DialoGPT/issues/34

def group_texts(dialogues):
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

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=batch_size,
    num_proc=4,
)

print(tokenizer.decode(lm_datasets["train"][0]["input_ids"]))

model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)

training_args = TrainingArguments(
    "dialoGPT_dailyDialog_model",
    do_train= True,
    do_eval= True,
    evaluation_strategy="epoch", #steps or epoch
    learning_rate=1e-3,
    #max_steps=2, #only for debugging
    num_train_epochs=max_epochs,
    load_best_model_at_end = True,
    save_strategy='epoch',
    save_total_limit=5
)
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

#push to lisa: rsync -av . lcur0343@lisa.surfsara.nl:~/rational-dialog-model
#pull logs from lisa: rsync -av lcur0343@lisa.surfsara.nl:~/rational-dialog-model/runs .
#Tutorial: https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb