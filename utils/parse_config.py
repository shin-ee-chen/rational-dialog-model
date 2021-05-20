import torch
import yaml
from torch.utils.data import DataLoader

from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer
from daily_dialog.UtterancesDataset import UtterancesDataset
from modules.LanguageModels.LstmLanguageModel import LSTMLanguageModel
from modules.LanguageModels.PretrainedLanguageModel import PretrainedLanguageModel
from modules.RationalExtractors.PolicyBasedRationalExtractor import PolicyBasedRationalExtractor
from modules.pytorch_lightning.LightningLanguageModel import LightningLanguageModel
import pytorch_lightning as pl
from transformers import AutoTokenizer
from tokenizers import Tokenizer

from modules.pytorch_lightning.LightningReinforceRationalizedLanguageModel import LightingReinforceRationalizedLanguageModel
from utils.callbacks import FinishDialogueCallback


def parse_config_lm(config_ref):
    with open(config_ref, 'r') as f:
        config = yaml.load(f)

    # First we load the tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])

    datasets = get_datasets(config["dataset"], tokenizer)

    language_model = get_language_model(config["language_model"], tokenizer)

    hparams = config["hparams"]

    loss_module = get_loss_module(config["loss_module"], tokenizer)

    # Load the pytorch lightning module
    lightning_language_model = LightningLanguageModel(language_model, tokenizer, loss_module=loss_module,
                                                      hparams=hparams)

    trainer = get_trainer(config["trainer"])

    return {"tokenizer": tokenizer, **datasets, "language_model": language_model, "hparams": hparams,
            "lightning_language_model": lightning_language_model, "trainer": trainer, "config": config}


def parse_config_RE(config_ref):
    with open(config_ref, 'r') as f:
        config = yaml.load(f)

    # First we load the tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])

    datasets = get_datasets(config["dataset"], tokenizer)

    language_model = get_language_model(config["language_model"], tokenizer)

    RE = get_rational_extractor(config["rational_extractor"], tokenizer)

    hparams = config["hparams"]

    loss_module = get_loss_module(config["loss_module"], tokenizer)

    # Load the pytorch lightning module
    lightning_language_model = LightingReinforceRationalizedLanguageModel(language_model, RE, tokenizer,
                                                                     loss_module=loss_module,
                                                                     hparams=hparams)

    trainer = get_trainer(config["trainer"])

    return {"tokenizer": tokenizer, **datasets, "language_model": language_model, "hparams": hparams,
            "lightning_language_model": lightning_language_model, "trainer": trainer, "config": config}


def get_tokenizer(tokenizer_config):
    if tokenizer_config["type"] == "transformers":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["checkpoint"])
    elif tokenizer_config["type"] == "daily_dialogue":
        print(tokenizer_config["link"])
        tokenizer = get_daily_dialog_tokenizer(tokenizer_location=tokenizer_config["link"], )
    else:
        raise ValueError("type not found", tokenizer_config["type"])
    return tokenizer


def get_datasets(config, tokenizer):
    if config["type"] == 'daily_dialogue':
        dataset_train = UtterancesDataset(tokenizer, subsets="start", split="train", 
                                          size=config["size_train"], remove_top_n=config["remove_top_n"])
        dataset_test = UtterancesDataset(tokenizer, subsets="start", split="test", 
                                         size=config["size_test"], remove_top_n=config["remove_top_n"])

        dataloader_train = DataLoader(dataset_train, batch_size=config["batch_size"],
                                      collate_fn=UtterancesDataset.get_collate_fn())
        dataloader_test = DataLoader(dataset_test, batch_size=config["batch_size"],
                                     collate_fn=UtterancesDataset.get_collate_fn())

        return {"dataloader_train": dataloader_train, "dataloader_test": dataloader_test}
    else:
        raise ValueError("type not found", config["type"])


def get_language_model(config, tokenizer):
    if config["type"] == "LSTM":
        if config["pretrained"]:
            model_name = config["save_location"]
            print("load pretrained_model: ", model_name)
            language_model = LSTMLanguageModel.load(model_name)
        else:
            print("load fresh model")
            language_model = LSTMLanguageModel(
                tokenizer.get_vocab_size(),
                embedding_dim=config['embedding_dim'],
                num_layers=config['num_layers'],
                hidden_state_size=config['hidden_state_size']
            )
    elif config["type"] == "transformers":
        language_model = PretrainedLanguageModel(pretrained_model=config['checkpoint'], tokenizer=tokenizer)
       
    else:
        raise ValueError("type not found", config["type"])
    return language_model


def get_loss_module(config, tokenizer):
    # TODO make sure we exclude the padding (Is now set 2 as a default)
    pad_id = 2
    # weight = torch.ones(tokenizer.get_vocab_size())
    weight = torch.ones(len(tokenizer))
    weight[pad_id] = 0
    return torch.nn.CrossEntropyLoss(weight=weight)


def get_rational_extractor(config, tokenizer):
    if config["type"] == "policy_based":
        # Mask token is at the moment 2
        if type(tokenizer) == Tokenizer: #nltk tokenizer
            return PolicyBasedRationalExtractor(tokenizer.get_vocab_size(), mask_token=4)
        else: #transformers tokenizer
            return PolicyBasedRationalExtractor(len(tokenizer), mask_token=4)

def get_trainer(config):
    # TODO add callbacks somehow
    if config["type"] == "normal":
        callbacks = [
            FinishDialogueCallback(["How are you doing today? [SEP]", "What are you upto? [SEP]"]),
        ]
        trainer = pl.Trainer(
            default_root_dir='logs',
            checkpoint_callback=False,
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=config["max_epochs"],
            log_every_n_steps=1,
            progress_bar_refresh_rate=1,
            callbacks=callbacks

        )
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        return trainer
    elif config["type"] == "policy":
        callbacks = [
            FinishDialogueCallback(["[START] How are you doing today?", "[START] What are you upto? "]),
        ]
        trainer = pl.Trainer(
            default_root_dir='logs',
            checkpoint_callback=False,
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=config["max_epochs"],
            log_every_n_steps=1,
            progress_bar_refresh_rate=1,
            callbacks=callbacks

        )
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        return trainer