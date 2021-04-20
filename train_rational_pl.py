''''
Should train a rational language model
Does not work yet :(
'''
import datasets
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer
from daily_dialog.RationalLanguageModelPL import PackedRationalLSTMLM, RationalLMPL
from daily_dialog.callbacks import FinishSentenceCallback
from utils import collate_fn
torch.autograd.set_detect_anomaly(True)
batch_size = 32

learning_rate = 1e-3

hparams = {
    "learning_rate": learning_rate
}

my_tokenizer = get_daily_dialog_tokenizer(tokenizer_location='./daily_dialog/tokenizer.json', )

dataset_train = datasets.load_dataset("daily_dialog", split="train", )
dataset_test = datasets.load_dataset("daily_dialog", split="test", )

dataloader_train = DataLoader(dataset_train, batch_size=32, collate_fn=collate_fn)
dataloader_test = DataLoader(dataset_test, batch_size=32, collate_fn=collate_fn)
device = "cuda"

language_model = PackedRationalLSTMLM(my_tokenizer.get_vocab_size()).to(device)

callbacks = [
    FinishSentenceCallback(["[START] How ", "[START] What are you upto? "])
]

max_epochs = 30
hparams = {'learning_rate': learning_rate}

loss_module = torch.nn.CrossEntropyLoss()

model = RationalLMPL(language_model, my_tokenizer, loss_module, hparams=hparams)

# images = torch.stack([mnist_train[i][0] for i in range(8)])
#
# callbacks = [
#     ShowReconstructedCallback(images)
# ]

trainer = pl.Trainer(default_root_dir='logs',
                     checkpoint_callback=False,
                     # checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                     gpus=1 if torch.cuda.is_available() else 0,
                     max_epochs=max_epochs,
                     log_every_n_steps=1,
                     progress_bar_refresh_rate=1,
                     callbacks=callbacks
                     )
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

trainer.fit(model, dataloader_train, dataloader_test)
