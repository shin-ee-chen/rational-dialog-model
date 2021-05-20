'''
Pytorch lightning version of the language model.
'''

import torch

from modules.pytorch_lightning.LightningBaseLanguageModel import LightningBaseLanguageModel
from utils.utils import calc_acc
from tokenizers import Tokenizer
from utils.token_utils import get_token_id

class LightningLanguageModel(LightningBaseLanguageModel):

    def batch_to_out(self, batch):
        """
        Should return a dictionary with at least the loss inside
        :param batch:
        :param batch_idx:
        :return: dict with {"loss": loss} and other values once finds relevant
        """

        batch = batch[0].permute(1, 0).to(self.device)
        input_tensor = batch[:-1, :]
        target_tensor = batch[1:, :]
        predictions = self.language_model.forward(input_tensor)

        if type(self.tokenizer) == Tokenizer:
            loss = self.loss_module(predictions.reshape(-1, self.tokenizer.get_vocab_size()), target_tensor.flatten(), )
            acc = calc_acc(
                predictions.reshape(-1, self.tokenizer.get_vocab_size()), 
                target_tensor.flatten(), 
                exclude=get_token_id(self.tokenizer, "pad_token")
            )
        else:
            loss = self.loss_module(predictions.reshape(-1, len(self.tokenizer)), target_tensor.flatten(), )
            acc = calc_acc(
                predictions.reshape(-1, len(self.tokenizer)), 
                target_tensor.flatten(), 
                exclude=get_token_id(self.tokenizer, "pad_token")
            )

        ### TODO need to check if this is calculated correctly.
        perplexity = torch.exp(loss)  # math.exp(loss) #torch.exp(loss)

        return {"loss": loss, 'predictions': predictions, "perplexity": perplexity, "acc": acc}


class RobertaMLPL(LightningBaseLanguageModel):

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

        perplexity = torch.exp(loss) # math.exp(loss) #torch.exp(loss)
        acc = calc_acc(predictions.reshape(-1, self.tokenizer.vocab_size), target_tensor)

        return {"loss": loss, 'predictions': predictions, "perplexity": perplexity, "acc": acc}

    def complete_dialogue(self, context, max_length):
        encoding = self.tokenizer(context)
        ids_tensor = torch.tensor(encoding.input_ids[1:-1]).to(self.device)

        completed_sentence_tokens = self.language_model.complete_dialogue(ids_tensor, max_length)
        new_sentence = self.tokenizer.decode(completed_sentence_tokens, skip_special_tokens=False)
        return new_sentence
