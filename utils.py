import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np



def to_packed_sequence(tensor, target=0):
    np_tensor = tensor.detach().cpu().numpy()
    indices = np_tensor == 0

    lengths = [np.where(ind == 0)[0][-1] + 1 + target for ind in indices ]

    return pack_padded_sequence(tensor, lengths, enforce_sorted=False, batch_first=True)


def collate_fn(dialogues):
    return ['[START] ' + '[SEP]'.join(dialogue["dialog"]) for dialogue in dialogues]


def get_next_input_ids(padded_input_tensor, start_index=0, end_index=-1):
    #
    next_input_ids = padded_input_tensor[0][:, start_index: end_index]
    # length_mask = padded_input_tensor[1] < end_index + padded_input_tensor[1] < end_index
    #
    # lengths = length_mask * padded_input_tensor[1] + end_index * ~length_mask
    lengths = torch.clamp(padded_input_tensor[1], start_index, end_index)
    next_input_ids = pack_padded_sequence(next_input_ids, lengths=lengths, batch_first=True, enforce_sorted=False)
    return next_input_ids


def get_packed_mean(t):
    padded_sequence = pad_packed_sequence(t)
    lenghts = padded_sequence[1].to(t.data.device)
    total_length = torch.sum(lenghts)

    mean = torch.sum(padded_sequence[0]) / total_length

    return mean


