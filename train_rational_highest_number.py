'''
Train the highest number prediction with the rational extractor.
'''
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from HighestNumber.HighestDataset import HighestDataset
from daily_dialog.RationalExtractor import RationalExtractor
from HighestNumber.model import LstmPL
torch.autograd.set_detect_anomaly(True)
max_epochs = 10

dataset_train = HighestDataset(n_examples=int(10e3))

dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=32)


dataset_test = HighestDataset(n_examples=int(10e3))

dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=32)


loss_module = torch.nn.CrossEntropyLoss()

model = RationalExtractor()

hparams = {
    "learning_rate": 1e-3
}

model_pl = LstmPL(model, loss_module, hparams=hparams)
trainer = pl.Trainer(default_root_dir='logs',
                     checkpoint_callback=False,
                     # checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                     gpus=1 if torch.cuda.is_available() else 0,
                     max_epochs=max_epochs,
                     log_every_n_steps=1,
                     progress_bar_refresh_rate=1,
                     gradient_clip_val=2,
                     )
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

trainer.fit(model_pl, dataloader_train, dataloader_test)
