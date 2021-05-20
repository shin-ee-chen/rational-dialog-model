import torch
from torch import nn
from torch.nn import Embedding, LSTM
import torch.nn.functional as F


class PolicyBasedRationalExtractor(nn.Module):

    def __init__(self, embedding_input_size, embedding_size=32, mask_token=0):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = Embedding(embedding_input_size, embedding_size)

        # Layers for prediction
        self.prediction_LSTM = LSTM(embedding_size, hidden_size=int(embedding_size / 2), bidirectional=True)
        self.output_layer = nn.Linear(embedding_size, 2)
        self.mask_token = mask_token

    def forward(self, x, greedy=False):
        '''
            Given a batch of sequences it outputs information about the policy, the actual chosen policy,
            the mask that was used and the masked x.
        '''
        e = self.embedding(x)

        # Calculate mask
        h, (h_n, c_n) = self.prediction_LSTM(e)

        logits = self.output_layer(h)

        policy = F.softmax(logits, dim=-1)

        policy_reshaped = policy.view(-1, 2)

        if greedy:
            mask = torch.argmax(policy_reshaped, dim=-1).bool()
        else:
            mask = torch.multinomial(policy_reshaped, 1).bool()

        mask = mask.reshape(x.shape[0], -1)
        
        masked_input = torch.mul(x, mask) + ~mask * self.mask_token

        # Selects the probabilities of the actual chosen policy.
        mask_to_gather = mask.reshape(x.shape[0], -1, 1).long()
        chosen_policy = torch.gather(policy, -1, mask_to_gather)

        return {"policy_logits": logits, "policy": policy, "chosen_policy": chosen_policy, "mask": mask,
                "masked_input": masked_input}
