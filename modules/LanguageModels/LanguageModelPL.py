'''
Pytorch lightning version of the language model.
'''
import math

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

        batch = batch[0].permute(1, 0).to(self.device)

        input_tensor = batch[:-1, :]

        target_tensor = batch[1:, :]


        predictions = self.language_model.forward(input_tensor)

        loss = self.loss_module(predictions.reshape(-1, self.tokenizer.get_vocab_size()), target_tensor.flatten(), )

        perplexity = 0 #math.exp(loss) #torch.exp(loss)

        return {"loss": loss, 'predictions': predictions, "perplexity": perplexity}



class RobertaMLPL(BaseLanguageModelPL):

    def batch_to_out(self, batch):
        '''
        Should return a dictionary with at least the loss inside
        :param batch:
        :param batch_idx:
        :return: dict with {"loss": loss} and other values once finds relevant
        '''

        batch = batch[0].permute(1, 0).to(self.device)
        input_tensor = batch[:-1, :]
        target_tensor = batch[1:, :]
        predictions = self.language_model.forward(input_tensor)
        loss = self.loss_module(predictions.reshape(-1, self.tokenizer.vocab_size), target_tensor.flatten(), )

        perplexity = 0 #math.exp(loss) #torch.exp(loss)

        return {"loss": loss, 'predictions': predictions, "perplexity": perplexity}

    def complete_dialogue(self, sentence, max_length):
        encoding = self.tokenizer(sentence)
        ids_tensor = torch.tensor(encoding.input_ids[1:-1]).to(self.device)

        completed_sentence_tokens = self.language_model.complete_dialogue(ids_tensor, max_length)
        new_sentence = self.tokenizer.decode(completed_sentence_tokens, skip_special_tokens=False)
        return new_sentence