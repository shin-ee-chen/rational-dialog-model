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

        # Layers for Kumaraswamy
        self.kur_lstm = LSTM(embedding_size, hidden_size=int(embedding_size / 2), bidirectional=True, num_layers=2)
        self.kur_layer = KumaraswamyLayer(embedding_size)

        # Layers for prediction
        self.prediction_LSTM = LSTM(embedding_size, hidden_size=embedding_size)
        self.output_layer = nn.Linear(embedding_size, out_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        e = self.embedding(x)

        # Calculate mask
        h, (h_n, c_n) = self.kur_lstm(e)
        z = self.kur_layer(h)

        # Apply mask to embeddings
        z_repeated = z.unsqueeze(-1).repeat(1, 1, self.embedding_size)
        e_masked = z_repeated * e

        # Calculate output on masked embeddings
        h, (h_n, c_n) = self.prediction_LSTM(e_masked)
        out = self.output_layer(h_n)

        return {"logits": out.squeeze(0), "z": z}

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
        a = a.clamp(1e-3, 1000.)  # extreme values could result in NaNs
        b = b.clamp(1e-3, 1000.)  # extreme values could result in NaNs
        h = sample_hardkurma(x, a, b, self.l, self.r)

        return h


def inverse_kuma(u, a, b):
    return torch.pow((1 - torch.pow((1 - u), 1/a)), 1/b)


def sample_hardkurma(x, a, b, l, r):
    probs = torch.rand(x.shape[:2]).to(x.device)

    k = inverse_kuma(probs, a, b)
    t = l + (r - l) * k
    h = t.clamp(0.0, 1.0)

    return h



class RationalExtractorGumbell(nn.Module):

    def __init__(self, embedding_input=11, embedding_size=32, out_size=5):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = Embedding(embedding_input, embedding_size)

        # Layers for selection of input embeddings
        self.rational_lstm = LSTM(embedding_size, hidden_size=int(embedding_size / 2), bidirectional=True, num_layers=2)
        self.gumbel_select_layer = GumbelSelectLayer(embedding_size)

        # Layers for prediction
        self.prediction_LSTM = LSTM(embedding_size, hidden_size=embedding_size)
        self.output_layer = nn.Linear(embedding_size, out_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        e = self.embedding(x)

        # Calculate selector
        h, (h_n, c_n) = self.rational_lstm(e)
        z = self.gumbel_select_layer.forward(h)

        # Apply selector to embeddings
        z_repeated = h.unsqueeze(-1).repeat(1, 1, self.embedding_size)
        e_masked = z_repeated * e

        # Calculate output using masked embeddings
        h, (h_n, c_n) = self.prediction_LSTM(e_masked)
        out = self.output_layer(h_n)
        return {"logits": out.squeeze(0), "z": z}

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
#        print("use gumbel")
        probs = self.gumbel_softmax(logits)
        h = probs[:,:, -1]

        return h
