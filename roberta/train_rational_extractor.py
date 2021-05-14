from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from daily_dialog.NextNPredictionDataset import NextNPredictionDataset
from utils.callbacks import FinishDialogueRationalizedCallback
from roberta.wrapper import PretrainedWrapper

from modules.pytorch_lightning.PredictionLMPL import PredictionLMPL
from modules.RationalExtractor import RationalExtractor

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
language_model = AutoModelForCausalLM.from_pretrained("distilroberta-base")
lm = PretrainedWrapper(language_model)

teacher_forcing = True
size = int(1e2)
test_size = int(1e2)
embedding_dim = lm.embedding.embedding_dim

max_epochs = 50
batch_size = 32

learning_rate = 5e-5

hparams = {
    "learning_rate": learning_rate,
    "teacher_forcing": teacher_forcing,
    "freeze_language_ml": False
}

callbacks = [
    FinishDialogueRationalizedCallback(["How ", "What are you upto? "]),
]

train_dataset = NextNPredictionDataset(tokenizer, size=size)
dataloader_train = DataLoader(train_dataset, shuffle=True)

test_dataset = NextNPredictionDataset(tokenizer, size=test_size, split="test")
dataloader_test = DataLoader(test_dataset, )

device = "cuda"

weights = torch.ones(tokenizer.vocab_size)

#Makes sure to ignore certain indices.
weights[0:10] = 0
weights[17487] = 0
loss_module = torch.nn.CrossEntropyLoss(weight=weights)

rational_extractor = RationalExtractor(embedding_dim)

model = PredictionLMPL(lm, rational_extractor, tokenizer, loss_module, hparams=hparams)

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
