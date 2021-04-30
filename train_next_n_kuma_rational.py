'''
Trains a language model on the daily dialog set.
'''

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from daily_dialog.PredictionDataset import PredictionDataset
from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer

from daily_dialog.callbacks import FinishSentenceRationalizedCallback
from modules.LanguageModels.LstmLanguageModel import LSTMLM
from modules.PredictionLMPL import PredictionLMPL
from modules.RationalExtractor import RationalExtractor
from modules.kurmaswamy.KumaRationalExtractor import KumaRationalExtractor

save_path = './small_lm_pretrained.pt'
load_pretrained = True
size = int(5e2)
test_size = int(1e2)

max_epochs = 50
batch_size = 32
embedding_dim = 128
learning_rate = 1e-3

hparams = {
    "learning_rate": learning_rate
}

my_tokenizer = get_daily_dialog_tokenizer(tokenizer_location='./daily_dialog/tokenizer.json', )

train_dataset = PredictionDataset(my_tokenizer, size=size)
dataloader_train = DataLoader(train_dataset, shuffle=True)

test_dataset = PredictionDataset(my_tokenizer, size=test_size, split="test")
dataloader_test = DataLoader(test_dataset, )

device = "cuda"

if load_pretrained:
    print("load pretrained_model")

    language_model = LSTMLM.load(save_path).to(device)
else:
    print("load fresh model")
    language_model = LSTMLM(my_tokenizer.get_vocab_size(), embedding_dim=embedding_dim).to(device)

callbacks = [
    FinishSentenceRationalizedCallback(["[START] How ", "[START] What are you upto? "])
]

hparams = {'learning_rate': learning_rate}

loss_module = torch.nn.CrossEntropyLoss()

rational_extractor = KumaRationalExtractor(embedding_dim)

model = PredictionLMPL(language_model, rational_extractor, my_tokenizer, loss_module, hparams=hparams)

trainer = pl.Trainer(default_root_dir='logs',
                     checkpoint_callback=False,
                     gpus=1 if torch.cuda.is_available() else 0,
                     max_epochs=max_epochs,
                     log_every_n_steps=1,
                     callbacks=callbacks,
                     progress_bar_refresh_rate=1,
                     gradient_clip_val=2.0
                     )
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

trainer.fit(model, dataloader_train, dataloader_test)
