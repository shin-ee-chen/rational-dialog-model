'''
Train the highest number prediction with the rational extractor.
'''
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from HighestNumber.HighestDataset import HighestDataset
from HighestNumber.RationalExtractor import RationalExtractorGumbell, RationalExtractor
from HighestNumber.model import LstmPL
from HighestNumber.reinforce_rational_extraction import ReinforceRationalExtractor, HighestNumberLSTM, ReinforceModelPL

max_epochs = 3

dataset_train = HighestDataset(n_examples=int(10e3))
dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=8)

dataset_test = HighestDataset(n_examples=int(10e2))
dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=8)

loss_module = torch.nn.CrossEntropyLoss()
model = HighestNumberLSTM()

RE = ReinforceRationalExtractor()

hparams = {
    "learning_rate": 1e-3
}

model_pl_pretrain = ReinforceModelPL(model, RE, loss_module, hparams=hparams, pretrain=True)
trainer = pl.Trainer(default_root_dir='logs',
                     checkpoint_callback=False,
                     # checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                     gpus=1 if torch.cuda.is_available() else 0,
                     max_epochs=5,
                     log_every_n_steps=1,
                     progress_bar_refresh_rate=1,
                     gradient_clip_val=2,
                     )
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

trainer.fit(model_pl_pretrain, dataloader_train, dataloader_test)
loss_module = torch.nn.CrossEntropyLoss(reduce=False)
model_pl = ReinforceModelPL(model, RE, loss_module, hparams=hparams, pretrain=False)
trainer = pl.Trainer(default_root_dir='logs',
                     checkpoint_callback=False,
                     # checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                     gpus=1 if torch.cuda.is_available() else 0,
                     max_epochs=40,
                     log_every_n_steps=1,
                     progress_bar_refresh_rate=1,
                     gradient_clip_val=2,
                     )
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

trainer.fit(model_pl, dataloader_train, dataloader_test)
