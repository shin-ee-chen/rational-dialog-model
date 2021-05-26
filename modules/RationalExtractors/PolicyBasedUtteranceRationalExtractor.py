import torch
from torch import nn
from torch.nn import Embedding, LSTM
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import itertools
flatten = itertools.chain.from_iterable

class PolicyBasedUtteranceRationalExtractor(nn.Module):

    def __init__(self, embedding_input_size, embedding_size=32, mask_token=0, sep_token=None):
        assert sep_token != None, "sep_token_id must be provided"

        print(embedding_input_size)
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = Embedding(embedding_input_size, embedding_size)

        # Layers for prediction
        self.prediction_LSTM = LSTM(embedding_size, hidden_size=int(embedding_size / 2), bidirectional=True)
        self.output_layer = nn.Linear(embedding_size, 2)
        self.mask_token = mask_token
        self.sep_token = sep_token

    def forward(self, x, greedy=False):
        '''
            Given a batch of sequences it outputs information about the policy, the actual chosen policy,
            the mask that was used and the masked x.
            x has dimensions L x B, where L is length of sequences, B is batch size
        '''

        u, l = self.get_utterance_representations(x, device=x.device)

        # Calculate mask for utterances
        h, (h_n, c_n) = self.prediction_LSTM(u)
        logits = self.output_layer(h)
        policy = F.softmax(logits, dim=-1)
        policy_reshaped = policy.view(-1, 2)
        if greedy:
            u_mask = torch.argmax(policy_reshaped, dim=-1).bool()
        else:
            u_mask = torch.multinomial(policy_reshaped, 1).bool()
        u_mask = u_mask.reshape(-1, x.shape[1])

        # apply mask to input
        mask = self.get_token_mask(u_mask, l, device=x.device)
        masked_input = torch.mul(x, mask) + ~mask * self.mask_token

        # Selects the probabilities of the actual chosen policy.
        mask_to_gather = u_mask.reshape(-1, x.shape[1], 1).long()
        chosen_policy = torch.gather(policy, -1, mask_to_gather)

        return {"policy_logits": logits, "policy": policy, "chosen_policy": chosen_policy, "mask": u_mask,
                "masked_input": masked_input}


    def get_utterance_representations(self, x, device="cpu"):
        # Split batch in utterances. Utterances are separated by sep_token
        batch = torch.transpose(x, 0, 1).tolist()
        utterances_batch = [
            self.split_on_sep(context)
            for context in batch
        ]
        utterance_lengths = [
            [len(utterance) for utterance in context]
            for context in utterances_batch
        ]

        # Calculate the utterance representation by taking average of the token embeddings
        utterance_reps_batch = [[self.utterance_rep(utterance, device=x.device)
                for utterance in context] 
                for context in utterances_batch]

        # Add zero embeddings where necessary as padding
        max_utterances = max([len(context) for context in utterances_batch])
        padded_reps_batch = torch.stack([
            torch.stack(context + [torch.zeros(self.embedding_size).to(device)] * (max_utterances - len(context)))
            for context in utterance_reps_batch
        ]).transpose(1,0)

        return (padded_reps_batch.to(x[0].device), utterance_lengths)

    def utterance_rep(self, utterance, device="cpu"):
        '''
        Calculates the representation of an utterance, as the mean of the embeddings of the tokens
        '''
        u_embed = torch.stack([self.embedding(torch.tensor(token_id).long().to(device)) for token_id in utterance])
        u_rep = torch.mean(u_embed, dim=0)
        return u_rep


    def get_token_mask(self, mask_batch, length_batch, device="cpu"):
        '''
        mask_batch: list of True, False; True means whole utterance is masked
        length_batch: list of lenghts of the utterances in context
        Returns tensor with mask_tokens for complete input
        '''

        token_masks_lst = [
            list(flatten([[mask_batch[u, b]] * len_u for (u, len_u) in enumerate(length_batch[b])]))
            for b in range(len(length_batch))
        ]
        token_masks = torch.tensor(token_masks_lst).transpose(1,0)

        return token_masks.to(device)


    def split_on_sep(self, l):

        size = len(l)
        idx_list = [idx + 1 for idx, val in enumerate(l) if val == self.sep_token]
        if len(idx_list) == 0:
            return([l])
        assert len(idx_list) > 0, l
        
        res = [
            l[i: j] 
            for i, j in zip(
                [0] + idx_list, 
                idx_list + ([size] if idx_list[-1] != size else [])
            )
        ]
        return res
