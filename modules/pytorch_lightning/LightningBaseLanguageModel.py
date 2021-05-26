'''
Pytorch lightning version for training a language model.
'''
import pytorch_lightning as pl
import torch

from utils.token_utils import get_token
from utils.utils import decode
from tokenizers import Tokenizer

class LightningBaseLanguageModel(pl.LightningModule):

    def __init__(self, language_model, tokenizer,  hparams=None):
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer

        self.log_list = [
            "loss", "perplexity", "acc"
        ]
        self.hparams = hparams

    def complete_dialogues(self, sentences, max_length):
        return [self.complete_dialogue(sentence, max_length) for sentence in sentences]

    def complete_dialogue(self, context, max_length):
        encoding = self.tokenizer.encode(context)
        if type(self.tokenizer) == Tokenizer:
            ids_tensor = torch.tensor(encoding.ids).to(self.device)
        else:
            ids_tensor = torch.tensor(encoding).to(self.device)

        completed_dialogue_tokens = self.language_model.complete_dialogue(ids_tensor, max_length)
        completed_dialogue = decode(self.tokenizer, completed_dialogue_tokens)
        return completed_dialogue.encode('utf8')

    def next_utterance(self, context, sep_token):
        encoding = self.tokenizer.encode(context)
        if type(self.tokenizer) == Tokenizer:
            ids_tensor = torch.tensor(encoding.ids).to(self.device)
        else:
            ids_tensor = torch.tensor(encoding).to(self.device)

        next_utterance_tokens = self.language_model.next_utterance(ids_tensor, sep_token)
        new_sentence = str(self.tokenizer.decode(
            next_utterance_tokens,
            skip_special_tokens=False
        )).replace(" #", "").replace("#", "")
        sep = new_sentence.find(get_token(self.tokenizer, "sep_token"))
        return new_sentence[:sep]

    def batch_to_out(self, batch):
        '''
        Should return a dictionary with at least the loss inside
        :param batch:
        :param batch_idx:
        :return: dict with {"loss": loss} and other values one finds relevant
        '''
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        result = self.batch_to_out(batch)
        self.log_results(result)

        return result["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        result = self.batch_to_out(batch)
        self.log_results(result, prepend="val ")

    def configure_optimizers(self,):
        parameters = list(self.language_model.parameters())
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.hparams['learning_rate']
        )
        return optimizer

    def log_results(self, result, prepend=""):
        for k in self.log_list:
            self.log(prepend + k, result[k], on_step=True, on_epoch=True)
