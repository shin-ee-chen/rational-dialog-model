import glob
import os
import time

# From: https://github.com/bastings/interpretable_predictions/blob/master/latent_rationale/beer/models/rl.py
# With a small modification to introduce the mean.
import torch
from tokenizers import Tokenizer 


def fussed_lasso(tokens, t, reduce=True, pad_id=0):
    zdiff = t[:, 1:] - t[:, :-1]
    zdiff = zdiff.abs()  # [B]
    if reduce:
        zdiff = zdiff.sum()
        non_paddings = (tokens != pad_id).float().sum()
    else:
        zdiff = zdiff.sum(dim=-1)
        non_paddings = (tokens != pad_id).float().sum(dim=-1)
    return zdiff/non_paddings


def calculate_mask_percentage(tokens, mask, reduce=False, pad_id=None):
    if reduce:
        mask_sum = mask.sum()
        non_paddings = (tokens != pad_id).float().sum()
    else:
        mask_sum = mask.sum(dim=-1)
        non_paddings = (tokens != pad_id).float().sum(dim=-1)

    mean = mask_sum / non_paddings
    return mean


def calc_acc(predictions, targets, exclude=None):

    indices = torch.argmax(predictions, dim=-1)

    if exclude != None:
        to_use = targets != exclude
        total_to_use = to_use.float().sum()
        correct = (indices == targets).float() * to_use
        return correct.sum()/total_to_use
    else:
        correct = (indices == targets).float()
        return torch.mean(correct)

def get_latest_model_name(path):
    list_of_files = glob.glob(path + '*')
    latest_file = max(list_of_files, key=os.path.getctime)
    return (latest_file)


def generate_model_name(path):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    return (path + 'model_' + timestr)


def decode(tokenizer, tokens):
    '''
    Decodes with the help of a tokenizer the given list of tokens
    '''
    if type(tokens) == torch.Tensor:
        tokens = list(tokens.detach().cpu().numpy())
    return tokenizer.decode(tokens, skip_special_tokens=False).replace(" #",
                                                                       "").replace(
        "#", "")


def encode(tokenizer, context):
    """"
    Encodes a context
    """
    return torch.tensor(tokenizer.encode(context).ids).long()


def batch_decode(tokenizer, batch_of_tokens):
    if type(batch_of_tokens) == torch.Tensor:
        batch_of_tokens = list(batch_of_tokens.detach().cpu().numpy())

    return [decode(tokenizer, tokens) for tokens in batch_of_tokens]


def get_pad_id(tokenizer):
    if type(tokenizer) == Tokenizer: #nltk tokenizer
        print(tokenizer.token_to_id(tokenizer.padding["pad_token"]))
        return tokenizer.token_to_id(tokenizer.padding["pad_token"])
    else: #huggingface
        print("pad token id: %d" % (tokenizer.pad_token_id))
        return tokenizer.pad_token_id

