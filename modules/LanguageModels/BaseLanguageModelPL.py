'''
Pytorch lightning version for training a language model.
'''
import pytorch_lightning as pl
import torch

from utils import to_packed_sequence


class BaseLanguageModelPL(pl.LightningModule):

    def __init__(self, language_model, tokenizer, loss_module, hparams=None, ):
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.loss_module = loss_module
        self.log_list = [
            "loss", "perplexity"
        ]
        self.hparams = hparams

    def complete_dialogues(self, sentences, max_length):
        return [self.complete_dialogue(sentence, max_length) for sentence in sentences]

    def complete_dialogue(self, sentence, max_length):
        encoding = self.tokenizer.encode(sentence)
        ids_tensor = torch.tensor(encoding.ids).to(self.device)


        completed_dialogue_tokens = self.language_model.complete_dialogue(ids_tensor, max_length)
        new_sentence = str(self.tokenizer.decode(
            completed_dialogue_tokens,
            skip_special_tokens=False
        )).replace(" #", "").replace("#", "")

        return new_sentence

    def next_utterance(self, sentence, sep_token):
        encoding = self.tokenizer.encode(sentence)
        ids_tensor = torch.tensor(encoding.ids).to(self.device)

        next_utterance_tokens = self.language_model.next_utterance(ids_tensor, sep_token)
        new_sentence = str(self.tokenizer.decode(
            next_utterance_tokens, 
            skip_special_tokens=False
        )).replace(" #", "").replace("#", "")

#        original_length = len(sentence)
#        new_sentence = self.complete_sentence(sentence, max_length=100)
#        print("[DEBUG]\n", sentence, '\n', new_sentence)
        sep = new_sentence.find('[SEP]')
#       print("DEBUG: ", sep)
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
