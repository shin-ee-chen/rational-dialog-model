import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential

from torch.nn.utils.rnn import PackedSequence
import numpy as np

from modules.LanguageModels.BaseLanguageModel import BaseLanguageModel


class PretrainedWrapper(BaseLanguageModel):
    '''
    A wrapper for pretrained hugginface models.
    '''

    def __init__(self, hf_lm):
        super().__init__()
        self.lm = hf_lm
        self.embedding = self.lm.get_input_embeddings()

        self.layers = self.lm.get_output_embeddings()

    def to_embedding(self, x):
        return self.embedding(x)

    def forward_embedding(self, embedding):
        return self.lm(inputs_embeds=embedding).logits

    def forward(self, x):
        return self.lm.forward(x).logits

    def generate_next_tokens_from_embedding(self, embedding, n_tokens=10):
        tokens = []
        ## Initialize:
        logits = self.forward_embedding(embedding)
        logits = logits[-1]
        next_token = self.get_next_token_from_logits(logits)

        tokens.append(next_token)
        next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
        next_embedding = self.embedding(next_token_tensor)
        for i in range(n_tokens - 1):
            next_embedding = next_embedding.reshape(1, 1, -1)

            logits = self.forward_embedding(next_embedding)

            next_token = self.get_next_token_from_logits(logits)

            tokens.append(next_token)
            next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
            next_embedding = self.embedding(next_token_tensor)
        return tokens

    # def generate_next_token(self, tokens):
    #     tokens = tokens.reshape(1, -1 )
    #     logits = self.forward(tokens)[: , -1]
    #
    #     next_token = self.get_next_from_logits(logits)
    #
    #     next_token = torch.tensor([next_token]).to(tokens.device)
    #
    #     return next_token