'''
Contains the language model.
We make use of packing the input as this speeds up the code quite significantly.
Implements "packed" versions of the embedding, relu and linear layer.
Important note: Currently batch first is used.
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import numpy as np

from modules.packed import PackedEmbedding, PackedReLU, PackedLinear


class PackedLSTMLM(nn.Module):
    ''''
    A simple lstm language model
    '''

    def __init__(self, num_embeddings=1024, num_layers=2, embedding_dim=256, hidden_state_size=256):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_state_size = hidden_state_size

        self.embedding = PackedEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_state_size, num_layers=num_layers,
                            batch_first=True, dropout=0.5)

        self.relu = PackedReLU()
        self.classification_layer = PackedLinear(hidden_state_size, num_embeddings)

    def to_embedding(self, x):
        if type(x) == PackedSequence:
            return self.embedding(x)
        else:
            return self.embedding.forward_unpacked(x)

    def forward(self, x):
        embedding = self.to_embedding(x)

        return self.forward_embedding(embedding)

    def forward_embedding(self, embedding):
        '''
        Function that forward the embedding through the rest of the model.
        (Used in tendem with RE)
        '''
        out, hidden = self.lstm(embedding)
        out = self.relu.forward(out)
        classification = self.classification_layer(out, )

        return classification

    def complete_sentence(self, sentence, max_length):
        embedding = self.embedding.embedding(sentence)

        embedding = embedding.unsqueeze(dim=0)
        tokens = []

        # Initialization
        out, hidden = self.lstm(embedding)
        out = F.relu(out[0, -1])  # Get the latest token

        logits = self.classification_layer.classification_layer(out)
        next_token = self.get_next_from_logits(logits)

        tokens.append(next_token)
        next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)

        next_embedding = self.embedding.embedding(next_token_tensor)

        for i in range(max_length):
            next_embedding = next_embedding.reshape(1, 1, -1)

            out, hidden = self.lstm(next_embedding, hidden)
            # Greedy sampling

            out = F.relu(out)
            logits = self.classification_layer.classification_layer(out)
            next_token = self.get_next_from_logits(logits)

            tokens.append(next_token)
            next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
            next_embedding = self.embedding.embedding(next_token_tensor)
        return tokens

    def get_next_from_logits(self, logits, top=10):
        logits = logits.flatten().detach().cpu().numpy()
        top_indices = logits.argsort()[::-1][:top]
        top_logits = logits[top_indices]
        p = np.exp(top_logits) / sum(np.exp(top_logits))

        index = np.random.choice(top_indices, p=p)

        return index

    def save(self, location):
        torch.save({
            'model_state_dict': self.state_dict(),
            'kwargs': {
                'num_embeddings': self.num_embeddings,
                'num_layers': self.num_layers,
                'embedding_dim': self.embedding_dim,
                'hidden_state_size': self.hidden_state_size,
            }

        }, location)

    @classmethod
    def load(self, location):
        info = torch.load(location)
        model = PackedLSTMLM(**info['kwargs'])

        model.load_state_dict(info['model_state_dict'])
        model.train()
        return model


