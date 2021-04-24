'''
Contains the Kumarasqamy layer and a rational extractor (which is currently used for the highest number model)
'''
import torch
from torch import nn
from torch.nn import Embedding, LSTM
import pytorch_lightning as pl


class RationalExtractor(nn.Module):

    def __init__(self, embedding_input=11, embedding_size=32, out_size=5):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = Embedding(embedding_input, embedding_size)

        self.kur_lstm = LSTM(embedding_size, hidden_size=int(embedding_size / 2), bidirectional=True, num_layers=2)
        self.prediction_LSTM = LSTM(embedding_size, hidden_size=embedding_size)

        self.kur_layer = KumaraswamyLayer(embedding_size)

        self.output_layer = nn.Linear(embedding_size, out_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        embedding = self.embedding(x)

        lstm_out, (hidden, cell) = self.kur_lstm(embedding)
        h = self.kur_layer(lstm_out)
        h_repeated = h.unsqueeze(-1).repeat(1, 1, self.embedding_size)

        embedding = h_repeated * embedding

        lstm_out, (hidden, cell) = self.prediction_LSTM(embedding)
        out = self.output_layer(hidden)
        return {"logits": out.squeeze(0), "h": h}

    def sample(self, probabilities):
        pass


class KumaraswamyLayer(nn.Module):

    def __init__(self, in_features, l=-0.1, r=1.1):
        super().__init__()
        self.in_features = in_features

        self.layer_a = nn.Linear(in_features, 1)
        self.layer_b = nn.Linear(in_features, 1)

        self.l = l
        self.r = r
        self.softplus = nn.Softplus()

    def forward(self, x):

        a = self.softplus(self.layer_a(x)).squeeze(-1)

        b = self.softplus(self.layer_b(x)).squeeze(-1)
        a = a.clamp(1e-6, 100.)  # extreme values could result in NaNs
        b = b.clamp(1e-6, 100.)  # extreme values could result in NaNs

        h = sample_hardkurma(x, a, b, self.l, self.r)

        return h


def invert_k(u, a, b):
    return torch.pow((1 - torch.pow((1 - u), 1/a)), 1/b)


def sample_hardkurma(x, a, b, l, r):
    probs = torch.rand(x.shape[:2]).to(x.device)

    k = invert_k(probs, a, b)

    t = l + (r - l) * k

    maxed = t < 0

    t_maxed = (t * ~maxed) + torch.zeros(t.shape).to(x.device) * maxed
    mined = t_maxed > 1

    h = (t_maxed * ~mined) + torch.ones(t_maxed.shape).to(x.device) * mined

    return h



class RationalExtractorGumbell(nn.Module):

    def __init__(self, embedding_input=11, embedding_size=32, out_size=5):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = Embedding(embedding_input, embedding_size)

        self.rational_lstm = LSTM(embedding_size, hidden_size=int(embedding_size / 2), bidirectional=True, num_layers=2)



        self.prediction_LSTM = LSTM(embedding_size, hidden_size=embedding_size)



        self.gumbel_select_layer = GumbelSelectLayer(embedding_size)

        self.output_layer = nn.Linear(embedding_size, out_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        embedding = self.embedding(x)

        lstm_out, (hidden, cell) = self.rational_lstm(embedding)
        h = self.gumbel_select_layer.forward(lstm_out)
        h_repeated = h.unsqueeze(-1).repeat(1, 1, self.embedding_size)

        embedding = h_repeated * embedding

        lstm_out, (hidden, cell) = self.prediction_LSTM(embedding)
        out = self.output_layer(hidden)
        return {"logits": out.squeeze(0), "h": h}

    def sample(self, probabilities):
        pass


class GumbelSelectLayer(nn.Module):

    def __init__(self, in_features,):
        super().__init__()
        self.in_features = in_features

        self.to_binary_logits = nn.Linear(in_features, 2)
        self.gumbel_softmax = nn.functional.gumbel_softmax

    def forward(self, x):

        logits = self.to_binary_logits(x)
        print("use gumbel")
        probs = self.gumbel_softmax(logits)

        h = probs[:,:, -1]

        return h
