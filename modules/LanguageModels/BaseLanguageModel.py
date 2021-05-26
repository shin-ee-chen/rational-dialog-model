import torch
from torch import nn

import numpy as np


class BaseLanguageModel(nn.Module):
    ''''
    The base language model.
    '''

    def to_embedding(self, x):
        return self.embedding(x)

    def forward_embedding(self, embedding):
        raise NotImplementedError("forward embedding not implemented")

    def forward(self, x):
        raise NotImplementedError("forward not implemented")

    def generate_next_tokens_from_embedding(self, embedding, n_tokens=10):
        raise NotImplementedError()

    def generate_next_token(self, tokens_ids):
        '''
        Given tokens we generate the next token
        '''
        if len(tokens_ids.shape) == 1:
            tokens_ids = tokens_ids.view(-1, 1)
        elif len(tokens_ids.shape) > 2:
            raise ValueError("tokens ids should have 1 or 2 dim but found: " + str(len(tokens_ids.shape)))
        logits = self.forward(tokens_ids)[-1, 0, :]
        next_token = self.get_next_token_from_logits(logits)
        return torch.tensor([next_token]).to(logits.device)
        # return next_token

    def complete_dialogue(self, context_tokens_ids, max_length=100):
        '''
        Complete the dialogue given the context
        '''
        tokens = context_tokens_ids.clone()
        while len(tokens) < max_length:
            next_token = self.generate_next_token(tokens)
            tokens = torch.cat([tokens, next_token], dim=-1)
        return tokens.long()

    def generate_next_tokens(self, context_tokens_ids, n_tokens=10):
        """
        Generate the next tokens
        """
        tokens = context_tokens_ids.clone()
        next_tokens = torch.tensor([]).to(context_tokens_ids.device)
        while len(next_tokens) < n_tokens:
            next_token = self.generate_next_token(tokens)
            next_token = next_token.reshape(-1, 1)
            next_tokens = torch.cat([next_tokens, next_token])
            tokens = torch.cat([tokens, next_token])
        return next_tokens.long()

    def next_utterance(self, context_tokens_ids, sep_token, max_length=100):
        """
        Generate the next utterance given the context.
        """
        assert(type(sep_token) == int)
        tokens = context_tokens_ids.clone()
        utterances = torch.tensor([], dtype=torch.int).to(context_tokens_ids.device)
        next_token = self.generate_next_token(tokens)
        # print("next token: ", next_token)
        while len(utterances) < max_length and next_token != sep_token:
            tokens = torch.cat([tokens, next_token], dim=-1)
            utterances = torch.cat([utterances, next_token], dim=-1)
            next_token = self.generate_next_token(tokens)
            # print(next_token, end=' ')
        #append sep token in the end
        utterances = torch.cat([utterances, torch.tensor([sep_token]).to(utterances.device)], dim=-1)
        return utterances.long()

    def get_next_token_from_logits(self, logits, top=10):
        logits_np = logits.flatten().detach().cpu().numpy()
        top_indices = logits_np.argsort()[::-1][:top]
        top_logits = logits_np[top_indices]
        p = np.exp(top_logits) / sum(np.exp(top_logits))

        index = np.random.choice(top_indices, p=p)
        # return torch.tensor([index]).to(logits.device)
        return index

    def save(self, location):
        raise NotImplementedError()

    @classmethod
    def load(self, location):
        raise NotImplementedError()
