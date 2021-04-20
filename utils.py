from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np



def to_packed_sequence(tensor, target=0):
    np_tensor = tensor.detach().cpu().numpy()
    indices = np_tensor == 0

    lengths = [np.where(ind == 0)[0][-1] + 1 + target for ind in indices ]

    return pack_padded_sequence(tensor, lengths, enforce_sorted=False, batch_first=True)


def collate_fn(dialogues):
    return ['[START] ' + '[SEP]'.join(dialogue["dialog"]) for dialogue in dialogues]

