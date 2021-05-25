'''
Contains the Kumarasqamy layer and a rational extractor (which is currently used for the highest number model)
'''
import torch
from torch import nn
from torch.nn import LSTM

class RationalExtractor(nn.Module):
    '''
    Rational extractor using the gumbal softmax trick.
    '''

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.lstm = LSTM(n_features, hidden_size=int(n_features / 2), bidirectional=True, num_layers=2)

        self.to_binary_logits = nn.Linear(n_features, 2)

        self.gumbel_softmax = nn.functional.gumbel_softmax

        self.hard = True

    def forward(self, embedding):
        '''
        Creates a rational for the given embedding.
        end_index is upto which index we need to get the rational (for the rest of the index we do not create a rational
        '''

        lstm_out, (hidden, cell) = self.lstm(embedding)
        binary_logits = self.to_binary_logits(lstm_out)
        if self.training:
            probs = self.gumbel_softmax(binary_logits, hard=True, dim=-1) ###TODO: Maybe set hard to false?
        else:
            probs = self.gumbel_softmax(binary_logits, hard=True, dim=-1)
        h = (probs[:, :, 1] + 1 - probs[:, :, 0]) / 2
        h_repeated = h.unsqueeze(-1).repeat(1, 1, embedding.shape[-1])

        masked_embedding = h_repeated * embedding

        return {"masked_embedding": masked_embedding, "h": h}

    def save(self, location):
    
        torch.save({
            'model_state_dict': self.state_dict(),
            'kwargs': {
                'embedding_input_size': self.n_features,
                # "embedding_size": self.embedding_size,
                # 'mask_token': self.mask_token,

            }
        }, location)

    @classmethod
    def load(self, location):
        
        info = torch.load(location)
        model = RationalExtractor(**info['kwargs'])
        model.load_state_dict(info['model_state_dict'])
        model.train()
        return model