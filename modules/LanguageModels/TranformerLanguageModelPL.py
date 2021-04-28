'''
Pytorch lightning version of the language model.
'''
import pytorch_lightning as pl
import torch
from transformers import get_cosine_schedule_with_warmup

from modules.LanguageModels.BaseLanguageModelPL import BaseLanguageModelPL


class TransformerLMPL(BaseLanguageModelPL):

    def batch_to_out(self, batch):
        '''
        Should return a dictionary with at least the loss inside
        :param batch:
        :param batch_idx:
        :return: dict with {"loss": loss} and other values once finds relevant
        '''

        # encodings = self.tokenizer.encode_batch(batch)
        batch = batch[0].permute(1, 0).to(self.device)

        input_tensor = batch[:-1, :]

        target_tensor = batch[1:, :]

        predictions = self.language_model.forward(input_tensor)

        loss = self.loss_module(predictions.reshape(-1, self.tokenizer.get_vocab_size()), target_tensor.flatten(), )

        perplexity = torch.exp(loss)

        return {"loss": loss, 'predictions': predictions, "perplexity": perplexity}

    def configure_optimizers(
            self,
    ):
        parameters = list(self.language_model.parameters())

        optimizer = torch.optim.Adam(
            parameters,
            lr=self.hparams['learning_rate'])

        scheduler = {
            "scheduler": get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams["num_warmup_steps"],
                                                         num_training_steps=self.hparams["num_training_steps"]),
            'interval': 'step'}
        return [optimizer], [scheduler]
