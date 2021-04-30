'''
Contains the Kumarasqamy layer and a rational extractor (which is currently used for the highest number model)
'''
import torch
from torch import nn
from torch.nn import LSTM
import pytorch_lightning as pl
from torch.nn.utils.rnn import PackedSequence

from modules.packed import PackedGumbellLayer
from utils import get_next_input_ids


class PackedRationalExtractor(nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.lstm = LSTM(n_features, hidden_size=int(n_features / 2), bidirectional=True, num_layers=2)

        self.to_bin_layer = PackedGumbellLayer(n_features)

    def forward(self, embedding):
        '''
        Creates a rational for the given embedding.
        end_index is upto which index we need to get the rational (for the rest of the index we do not create a rational
        '''

        lstm_out, (hidden, cell) = self.lstm(embedding)
        h = self.to_bin_layer(lstm_out)

        # h_repeated = h.data.unsqueeze(-1).repeat(1, 1, self.n_features)
        # print(embedding.data.shape)
        masked_embedding = torch.einsum('i, jk -> ik', h.data, embedding.data)

        masked_embedding = PackedSequence(masked_embedding, embedding.batch_sizes,
                                          sorted_indices=embedding.sorted_indices,
                                          unsorted_indices=embedding.unsorted_indices)

        return {"masked_embedding": masked_embedding, "h": h}


class RationalExtractor(nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.lstm = LSTM(n_features, hidden_size=int(n_features / 2), bidirectional=True, num_layers=2)

        self.to_binary_logits = nn.Linear(n_features, 2)

        self.gumbel_softmax = nn.functional.gumbel_softmax

        self.hard= True

    def forward(self, embedding):
        '''
        Creates a rational for the given embedding.
        end_index is upto which index we need to get the rational (for the rest of the index we do not create a rational
        '''

        lstm_out, (hidden, cell) = self.lstm(embedding)
        binary_logits = self.to_binary_logits(lstm_out)
        if self.training:
            probs = self.gumbel_softmax(binary_logits, hard=True, dim=-1)
        else:
            print("eval")
            probs = self.gumbel_softmax(binary_logits, hard=False, dim=-1)
        h = (probs[:, :, 1] + 1 - probs[:, :, 0]) / 2
        h_repeated = h.unsqueeze(-1).repeat(1, 1, embedding.shape[-1])

        masked_embedding = h_repeated * embedding

        return {"masked_embedding": masked_embedding, "h": h}

