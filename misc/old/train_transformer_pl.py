'''
Trains a tranformer language model on the daily dialog set.
'''
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from misc.old.CLMDataset import CLMDataset
from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer
from utils.callbacks import FinishDialogueCallback, ReshuffleDatasetCallback
from modules.pytorch_lightning.TranformerLanguageModelPL import TransformerLMPL
from modules.LanguageModels.TransformerLanguageModel import TransformerLM

save_path = '../../transformer_lm_pretrained_2.pt'
load_pretrained = False
max_epochs = 50
batch_size = 128
embedding_dim = 128
num_head = 2
num_hid = 3
num_layers = 3
dropout = 0.25
size = None
learning_rate = 1e-3
training_steps = max_epochs * 700
hparams = {
    "learning_rate": learning_rate,
    "num_training_steps": training_steps,
    "num_warmup_steps": int(training_steps / 5)
}

my_tokenizer = get_daily_dialog_tokenizer(tokenizer_location='./daily_dialog/tokenizer.json', )

dataset_train = CLMDataset(my_tokenizer, split="train", size=size)
dataset_test = CLMDataset(my_tokenizer, split="test", size=size)

dataloader_train = DataLoader(dataset_train, )
dataloader_test = DataLoader(dataset_test, )
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if load_pretrained:
    print("load pretrained_model")
    language_model = TransformerLM.load(save_path).to(device)
else:
    print("load fresh model")
    language_model = TransformerLM(my_tokenizer.get_vocab_size(), embedding_dim=embedding_dim, num_head=num_head,
                                   num_hid=num_hid, num_layers=num_layers, dropout=dropout).to(device)
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks = [
    FinishDialogueCallback(["[START] How are you doing today?", "[START] What are you upto? "]),
    lr_monitor,
    ReshuffleDatasetCallback(dataset_train) # To reshuffle the dataset.
]


loss_module = torch.nn.CrossEntropyLoss(ignore_index=0)

model = TransformerLMPL(language_model, my_tokenizer, loss_module, hparams=hparams)

trainer = pl.Trainer(default_root_dir='../../logs',
                     checkpoint_callback=False,
                     gpus=1 if torch.cuda.is_available() else 0,
                     max_epochs=max_epochs,
                     log_every_n_steps=1,
                     progress_bar_refresh_rate=1,
                     callbacks=callbacks,
                     gradient_clip_val=2.0
                     )
trainer.logger._default_hp_metric = None  # Optional logging argument that we do n't need

trainer.fit(model, dataloader_train, dataloader_test)

print("save language model")
language_model.save(save_path)
