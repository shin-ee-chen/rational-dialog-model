import torch
from modules.LanguageModels.LstmLanguageModel import LSTMLM
from modules.LanguageModels.LanguageModelPL import LMPL
from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer
from utils import get_lastest_model_name

save_path = r'./saved_models/'
device = "cuda" if torch.cuda.is_available() else "cpu"
loss_module = torch.nn.CrossEntropyLoss(ignore_index=0)
learning_rate = 1e-3
hparams = {"learning_rate": learning_rate}
model_name = get_lastest_model_name(save_path)
language_model = LSTMLM.load(model_name).to(device)
my_tokenizer = get_daily_dialog_tokenizer(tokenizer_location='./daily_dialog/tokenizer.json', )
model = LMPL(language_model, my_tokenizer, loss_module=loss_module, hparams=hparams)

print("=" * 60)
print("Petrained_model: ", model_name)
print("Device:          ", device)
print("=" * 60, "\n")
print("Type your question or statement after the '>'.")
print("The model's response appears after the '<'.\n")

# Let's chat :)

question = input("> ")
context = question + '[SEP]'
while len(question) > 0:
    answer = model.next_utterance(context + '[SEP]', sep_token=2)
    print("< " + answer)
    question = input("> ") 
    context += answer + '[SEP]' + question + '[SEP]'
print("< Bye")