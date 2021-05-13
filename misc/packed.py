'''
Contains code for packed layers:
(Not used at the moment because of the code cluttering and overhead of transforming from packedSequence to Tensors and back again.
'''
from torch import nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F

from HighestNumber.RationalExtractor import sample_hardkurma


class PackedEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, x):
        data = x.data
        embedding = self.embedding(data)

        return PackedSequence(embedding, x.batch_sizes, sorted_indices=x.sorted_indices,
                              unsorted_indices=x.unsorted_indices)

    def forward_unpacked(self, x):
        return self.embedding(x)


class PackedReLU(nn.Module):

    def forward(self, x):
        out = F.relu(x.data)
        return PackedSequence(out, x.batch_sizes, sorted_indices=x.sorted_indices,
                              unsorted_indices=x.unsorted_indices)

    def forward_unpacked(self, x):
        return F.relu(x)


class PackedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.classification_layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        data = x.data
        result = self.classification_layer(data.reshape(-1, self.in_features))
        return PackedSequence(result, x.batch_sizes, sorted_indices=x.sorted_indices,
                              unsorted_indices=x.unsorted_indices)

    def forward_unpacked(self, x):
        return self.classification_layer(x)


class PackedGumbellLayer(nn.Module):

    def __init__(self, in_features, ):
        super().__init__()
        super().__init__()
        self.in_features = in_features

        self.to_binary_logits = nn.Linear(in_features, 2)
        self.gumbel_softmax = nn.functional.gumbel_softmax

    def forward(self, x):
        data = x.data

        data = data.reshape(-1, self.in_features)

        logits = self.to_binary_logits(data)

        probs = self.gumbel_softmax(logits, hard=True)

        h = (probs[:, 1] + 1 - probs[:, 0]) / 2

        return PackedSequence(h, x.batch_sizes, sorted_indices=x.sorted_indices,
                              unsorted_indices=x.unsorted_indices)


class PackedKumaraswamyLayer(nn.Module):

    def __init__(self, in_features, l=-0.1, r=1.1):
        super().__init__()
        self.in_features = in_features

        self.layer_a = nn.Linear(in_features, 1)
        self.layer_b = nn.Linear(in_features, 1)

        self.l = l
        self.r = r
        self.softplus = nn.Softplus()

    def forward(self, x):
        data = x.data

        data = data.reshape(-1, self.in_features)

        a = self.softplus(self.layer_a(data))

        b = self.softplus(self.layer_b(data))

        a = a.clamp(1e-5, 10.)  # extreme values could result in NaNs
        b = b.clamp(1e-5, 10.)  # extreme values could result in NaNs

        h = sample_hardkurma(data.unsqueeze(1), a, b, self.l, self.r)

        return PackedSequence(h, x.batch_sizes, sorted_indices=x.sorted_indices,
                              unsorted_indices=x.unsorted_indices)
