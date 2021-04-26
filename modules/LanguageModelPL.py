'''
Pytorch lightning version of the language model.
Used for convenience and reproducability.
'''
import pytorch_lightning as pl
import torch

from utils import to_packed_sequence


class LMPL(pl.LightningModule):

    def __init__(self, language_model, tokenizer, loss_module, hparams=None, ):
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.loss_module = loss_module
        self.log_list = [
            "loss"
        ]
        self.hparams = hparams

    def complete_sentences(self, sentences, max_length):
        return [self.complete_sentence(sentence, max_length) for sentence in sentences]

    def complete_sentence(self, sentence, max_length):
        encoding = self.tokenizer.encode(sentence)


        ids_tensor = torch.tensor(encoding.ids).to(self.device)

        completed_sentence_tokens = self.language_model.complete_sentence(ids_tensor, max_length)
        new_sentence = sentence + str(
            self.tokenizer.decode(completed_sentence_tokens, skip_special_tokens=False)).replace(" #", "").replace("#",
                                                                                                                   "")

        return new_sentence

    def batch_to_out(self, batch):
        '''
        Should return a dictionary with at least the loss inside
        :param batch:
        :param batch_idx:
        :return: dict with {"loss": loss} and other values once finds relevant
        '''

        encodings = self.tokenizer.encode_batch(batch)

        ids_tensor = torch.tensor([encoding.ids for encoding in encodings]).to(self.device)

        input_tensor = to_packed_sequence(ids_tensor[:, :-1]).to(self.device)

        target_tensor = to_packed_sequence(ids_tensor[:, 1:], target=1).to(self.device)
        predictions = self.language_model.forward(input_tensor)
        loss = self.loss_module(predictions.data, target_tensor.data, )

        return {"loss": loss, 'predictions': predictions}

    def training_step(self, batch, batch_idx):
        result = self.batch_to_out(batch)
        self.log_results(result)

        return result["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        result = self.batch_to_out(batch)
        self.log_results(result, prepend="val ")

    def configure_optimizers(
            self,
    ):
        parameters = list(self.language_model.parameters())

        optimizer = torch.optim.Adam(
            parameters,
            lr=self.hparams['learning_rate'])
        return optimizer

    def log_results(self, result, prepend=""):
        for k in self.log_list:
            self.log(prepend + k, result[k], on_step=True, on_epoch=True)
