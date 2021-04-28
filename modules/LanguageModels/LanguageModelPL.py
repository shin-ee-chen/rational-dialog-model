'''
Pytorch lightning version of the language model.
'''
import pytorch_lightning as pl
import torch

from modules.LanguageModels.BaseLanguageModelPL import BaseLanguageModelPL
from utils import to_packed_sequence


class LMPL(BaseLanguageModelPL):

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
        perplexity = torch.exp(loss)

        return {"loss": loss, 'predictions': predictions, "perplexity": perplexity}
