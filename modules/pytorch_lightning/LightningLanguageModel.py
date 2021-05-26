'''
Pytorch lightning version of the language model.
'''

import torch

from modules.pytorch_lightning.LightningBaseLanguageModel import LightningBaseLanguageModel
from utils.utils import calc_acc, calc_perplexity
from utils.token_utils import get_token_id, get_vocab_size, get_weights
import torch.nn.functional as F


class LightningLanguageModel(LightningBaseLanguageModel):

    def batch_to_out(self, batch):
        """
        Should return a dictionary with at least the loss inside
        :param batch:
        :param batch_idx:
        :return: dict with {"loss": loss} and other values once finds relevant
        """

        input_tensor = batch[0].permute(1, 0)
        target_tensor = batch[1].permute(1, 0)

        cat_tensor = torch.cat([input_tensor, target_tensor])

        input_tensor = cat_tensor[:-1, :]
        target_tensor = cat_tensor[1:, :]

        predictions = self.language_model.forward(input_tensor)

        vocab_size = get_vocab_size(self.tokenizer)
        weights = get_weights(self.tokenizer).to(input_tensor.device)
        loss = F.cross_entropy(predictions.reshape(-1, vocab_size), target_tensor.flatten(), weight=weights)

        acc = calc_acc(
            predictions.reshape(-1, vocab_size),
            target_tensor.flatten(),
            exclude=get_token_id(self.tokenizer, "pad_token")
        )

        perplexity = calc_perplexity(predictions, target_tensor, self.tokenizer)  # math.exp(loss) #torch.exp(loss)

        return {"loss": loss, 'predictions': predictions, "perplexity": perplexity, "acc": acc}
