import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import PackedSequence
import numpy as np


class LSTMLM(nn.Module):
    ''''
    A simple lstm language model
    '''

    def __init__(self, num_embeddings, num_layers=2, embedding_dim=128, hidden_state_size=128, ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_state_size = hidden_state_size

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_state_size, num_layers=num_layers,
                            )
        self.relu = F.relu
        self.classification_layer = nn.Linear(hidden_state_size, num_embeddings)

    def to_embedding(self, x):
        return self.embedding(x)

    def forward_embedding(self, embedding):
        out, hidden = self.lstm(embedding)

        out = self.relu(out)
        classification = self.classification_layer(out, )

        return classification

    def forward(self, x):
        if type(x) == PackedSequence:
            return self.packed_forward(x)
        embedding = self.to_embedding(x)

        out, hidden = self.lstm(embedding)

        out = self.relu(out)
        classification = self.classification_layer(out, )

        return classification

    def packed_forward(self, x):
        data = x.data

        embedding = self.embedding(data)

        embedding = PackedSequence(embedding, x.batch_sizes, sorted_indices=x.sorted_indices,
                                   unsorted_indices=x.unsorted_indices)
        out, hidden = self.lstm(embedding)
        out = F.relu(out.data)
        result = self.classification_layer(out.reshape(-1, self.hidden_state_size))
        return PackedSequence(result, x.batch_sizes, sorted_indices=x.sorted_indices,
                              unsorted_indices=x.unsorted_indices)

    def generate_next_tokens_from_embedding(self, embedding, n_tokens=10):
        tokens = []
        ## Initialize:
        out, hidden = self.lstm(embedding)

        out = F.relu(out[-1, 0])  # Get the latest token

        logits = self.classification_layer(out)
        next_token = self.get_next_from_logits(logits)

        tokens.append(next_token)
        next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
        next_embedding = self.embedding(next_token_tensor)
        for i in range(n_tokens - 1):
            next_embedding = next_embedding.reshape(1, 1, -1)

            out, hidden = self.lstm(next_embedding, hidden)

            out = F.relu(out)
            logits = self.classification_layer(out)
            next_token = self.get_next_from_logits(logits)

            tokens.append(next_token)
            next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
            next_embedding = self.embedding(next_token_tensor)
        return tokens

    def complete_sentence(self, sentence_ids, max_length=100):
        tokens = []
        ## Initialize:
        sentence_ids = sentence_ids.unsqueeze(1)
        embedding = self.embedding(sentence_ids)
        out, hidden = self.lstm(embedding)
        out = F.relu(out[-1, 0])  # Get the latest token

        logits = self.classification_layer(out)
        next_token = self.get_next_from_logits(logits)

        tokens.append(next_token)
        next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
        next_embedding = self.embedding(next_token_tensor)
        for i in range(max_length):
            next_embedding = next_embedding.reshape(1, 1, -1)

            out, hidden = self.lstm(next_embedding, hidden)

            out = F.relu(out)
            logits = self.classification_layer(out)
            next_token = self.get_next_from_logits(logits)

            tokens.append(next_token)
            next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
            next_embedding = self.embedding(next_token_tensor)
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
        model = LSTMLM(**info['kwargs'])
        model.load_state_dict(info['model_state_dict'])
        model.train()
        return model
