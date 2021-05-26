import glob
import os
import time
import yaml
import json

# From: https://github.com/bastings/interpretable_predictions/blob/master/latent_rationale/beer/models/rl.py
# With a small modification to introduce the mean.
import torch
from tokenizers import Tokenizer 
import torch.nn.functional as F

from utils.token_utils import get_weights, get_token_id


def fussed_lasso(tokens, mask, reduce=True, pad_id=None):
    assert pad_id != None, "Please provide pad_token_id to use for padding"
    non_paddings = (tokens != pad_id).float()

    mask = non_paddings * mask
    non_paddings = non_paddings.sum(dim=-1)


    zdiff = mask[:, 1:] - mask[:, :-1]
    zdiff = zdiff.abs().sum(dim=-1)  # [B]

    if reduce:
        return (zdiff / non_paddings).mean()
    else:
        return zdiff / non_paddings


def calculate_mask_percentage(tokens, mask, reduce=False, pad_id=None):
    non_paddings = (tokens != pad_id).float()
    mask = mask * non_paddings
    mask_sum = mask.sum(dim=-1)
    non_paddings = non_paddings.sum(dim=-1)
    mean = mask_sum / non_paddings
    if reduce:
        return mean.mean()
    else:
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



def calc_perplexity(predictions, targets,  tokenizer, batch_first=False):

    weight= get_weights(tokenizer).to(targets.device)
    loss = F.cross_entropy(predictions.reshape(-1, predictions.shape[2]), targets.flatten(), weight=weight)


    return torch.exp(loss)


def calc_cross_entropy_batch_wise(predictions, targets, tokenizer, batch_first=False):
    if not batch_first:
        predictions = predictions.permute(1, 0, 2)
        targets = targets.permute(1,0)
    weight = get_weights(tokenizer).to(targets.device)
    loss = F.cross_entropy(predictions.reshape(-1, predictions.shape[2]), targets.flatten(),
                    reduce=False, weight=weight)

    #Make sure batch first.
    loss = loss.reshape(predictions.shape[0], -1)
    exclude = get_token_id(tokenizer, "pad_token")
    to_use = (targets != exclude).float()
    total_to_use = to_use.sum(dim=-1)
    return loss.sum(dim=-1)/total_to_use


def calc_policy_loss(rewards, policy):

    ## We want to multiply rewards with policy based on the batch

    rewards = rewards.repeat(policy.shape[0], 1)
    policy = policy.squeeze(dim=-1)
    ## a safety check. 
    assert policy.shape == rewards.shape, "policy and rewards should be of same size"
    return -torch.mean((rewards.detach() * torch.log(policy)).mean(dim=0))

def log_config(trainer, config_ref):
    with open(config_ref, 'r') as f:
        config = yaml.load(f)
    trainer.logger.experiment.add_text("config", str(json.dumps(config)))
