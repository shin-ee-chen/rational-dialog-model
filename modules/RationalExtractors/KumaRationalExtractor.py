import torch
from torch import nn, dist
import torch.nn.functional as F
from torch.nn import LSTM

from modules.kurmaswamy.kuma_gate import KumaGate


class KumaRationalExtractor(nn.Module):

    def __init__(self, embedding_input_size):
        super().__init__()
        self.n_features = embedding_input_size
        self.lstm = LSTM(embedding_input_size, hidden_size=int(embedding_input_size / 2), bidirectional=True, num_layers=2)

        self.kuma_gate = KumaGate(embedding_input_size, dist_type="hardkuma")


    def forward(self, embedding, hard=False):
        '''
        Creates a rational for the given embedding.
        end_index is upto which index we need to get the rational (for the rest of the index we do not create a rational
        '''

        lstm_out, (hidden, cell) = self.lstm(embedding)
        kuma_dist = self.kuma_gate(lstm_out)
        if self.training and hard == False:
            mask = kuma_dist.sample()
        else:
            p0 = kuma_dist.pdf(lstm_out.new_zeros(()))
            p1 = kuma_dist.pdf(lstm_out.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            zero_one = torch.where(
                p0 > p1, lstm_out.new_zeros([1]), lstm_out.new_ones([1]))
            mask = torch.where((pc > p0) & (pc > p1),
                               kuma_dist.mean(), zero_one).bool()  # [B, M]


        h_repeated = mask.repeat(1, 1, embedding.shape[-1])

        masked_embedding = h_repeated * embedding
        mask = mask.squeeze(-1)

        return {"masked_embedding": masked_embedding, "mask": mask}

    def save(self, location):

        torch.save({
            'model_state_dict': self.state_dict(),
            'kwargs': {
                'embedding_input_size': self.n_features,
            }
        }, location)

    @classmethod
    def load(self, location):

        info = torch.load(location)
        model = KumaRationalExtractor(**info['kwargs'])
        model.load_state_dict(info['model_state_dict'])
        model.train()
        return model