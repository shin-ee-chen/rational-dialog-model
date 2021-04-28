import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import PackedSequence
import numpy as np


class BaseLanguageModel(nn.Module):
    ''''
    A simple lstm language model
    '''

    def to_embedding(self, x):
        return self.embedding(x)

    def forward_embedding(self, embedding):
        raise NotImplementedError("forward embedding not implemented")

    def forward(self, x):
        raise NotImplementedError("forward not implemented")

    def generate_next_tokens_from_embedding(self, embedding, n_tokens=10):
        raise NotImplementedError()

    def generate_next_token(self, tokens):
        raise NotImplementedError()

    def complete_sentence(self, sentence_ids, max_length=100):
        tokens = sentence_ids.clone()
        while len(tokens) < max_length:
            next_token = self.generate_next_token(tokens)

            tokens = torch.cat([tokens, next_token], dim=-1)
        return tokens.detach().cpu().numpy()


    def get_next_from_logits(self, logits, top=10):
        logits = logits.flatten().detach().cpu().numpy()
        top_indices = logits.argsort()[::-1][:top]
        top_logits = logits[top_indices]
        p = np.exp(top_logits) / sum(np.exp(top_logits))

        index = np.random.choice(top_indices, p=p)

        return index

    def save(self, location):
        raise NotImplementedError()

    @classmethod
    def load(self, location):
        raise NotImplementedError()
