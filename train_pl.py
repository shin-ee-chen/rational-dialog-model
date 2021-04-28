'''
Trains a language model on the daily dialog set.
'''
import datasets
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer
from modules.LanguageModels.LstmLanguageModel import LSTMLM
from modules.LanguageModels.LanguageModelPL import LMPL
from daily_dialog.callbacks import FinishSentenceCallback


from utils import collate_fn

save_path = './small_lm_pretrained_2.pt'
load_pretrained = False
batch_first = True
max_epochs = 250
batch_size = 128
embedding_dim = 128

learning_rate = 1e-3

hparams = {
    "learning_rate": learning_rate
}

my_tokenizer = get_daily_dialog_tokenizer(tokenizer_location='./daily_dialog/tokenizer.json', )

dataset_train = datasets.load_dataset("daily_dialog", split="train", )
dataset_test = datasets.load_dataset("daily_dialog", split="test", )

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=collate_fn)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate_fn)
device = "cuda"

if load_pretrained:
    print("load pretrained_model")
    language_model = LSTMLM.load(save_path).to(device)
else:
    print("load fresh model")
    language_model = LSTMLM(my_tokenizer.get_vocab_size(), embedding_dim=embedding_dim,).to(device)

callbacks = [
    FinishSentenceCallback(["[START] How are you doing today?", "[START] What are you upto? "])
]


hparams = {'learning_rate': learning_rate}

loss_module = torch.nn.CrossEntropyLoss(ignore_index=0)

model = LMPL(language_model, my_tokenizer, loss_module, hparams=hparams)

trainer = pl.Trainer(default_root_dir='logs',
                     checkpoint_callback=False,

                     gpus=1 if torch.cuda.is_available() else 0,
                     max_epochs=max_epochs,
                     log_every_n_steps=1,
                     progress_bar_refresh_rate=1,
                    callbacks=callbacks,
                     gradient_clip_val=2.0
                     )
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

trainer.fit(model, dataloader_train, dataloader_test)

print("save language model")
language_model.save(save_path)
