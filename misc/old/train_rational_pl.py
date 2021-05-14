''''
Should train a rational language model
Does not work yet :(
'''
import datasets
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer

from old.RationalLanguageModelPL import RationalLMPL
from utils.callbacks import FinishDialogueCallback
from old.language_model import PackedLSTMLM
from modules.RationalExtractor import PackedRationalExtractor
from utils import collate_fn
torch.autograd.set_detect_anomaly(True)

save_path = '../small_lm.pt'
load_pretrained = False
max_epochs = 50
batch_size = 32
embedding_dim=256

learning_rate = 1e-3

hparams = {
    "learning_rate": learning_rate
}

my_tokenizer = get_daily_dialog_tokenizer(tokenizer_location='./daily_dialog/tokenizer.json', )

dataset_train = datasets.load_dataset("../daily_dialog", split="train", )
dataset_test = datasets.load_dataset("../daily_dialog", split="test", )

dataloader_train = DataLoader(dataset_train, batch_size=32, collate_fn=collate_fn)
dataloader_test = DataLoader(dataset_test, batch_size=32, collate_fn=collate_fn)
device = "cuda"

if load_pretrained:
    print("load pretrained_model")

    language_model = PackedLSTMLM.load(save_path).to(device)
else:
    print("load fresh model")
    language_model = PackedLSTMLM(my_tokenizer.get_vocab_size()).to(device)

callbacks = [
    FinishDialogueCallback(["[START] How ", "[START] What are you upto? "])
]


rational_extractor = PackedRationalExtractor(embedding_dim)


loss_module = torch.nn.CrossEntropyLoss()

model = RationalLMPL(language_model, rational_extractor, my_tokenizer, loss_module, hparams=hparams)

trainer = pl.Trainer(default_root_dir='../logs',
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
