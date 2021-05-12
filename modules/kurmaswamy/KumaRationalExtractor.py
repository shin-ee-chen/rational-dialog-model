import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import LSTM

from modules.kurmaswamy.kuma_gate import KumaGate


class KumaRationalExtractor(nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.lstm = LSTM(n_features, hidden_size=int(n_features / 2), bidirectional=True, num_layers=2)

        self.kuma_gate = KumaGate(n_features, dist_type="hardkuma")

    def forward(self, embedding):
        '''
        Creates a rational for the given embedding.
        end_index is upto which index we need to get the rational (for the rest of the index we do not create a rational
        '''

        lstm_out, (hidden, cell) = self.lstm(embedding)

        kuma_dist = self.kuma_gate(lstm_out)
        if self.training:
            h = kuma_dist.sample()
        else:
            # TODO: this does not work properly I think.
            h = kuma_dist.sample()
            h = torch.round(h)

        h_repeated = h.repeat(1, 1, embedding.shape[-1])

        masked_embedding = h_repeated * embedding
        h = h.squeeze(-1)

        return {"masked_embedding": masked_embedding, "h": h}
