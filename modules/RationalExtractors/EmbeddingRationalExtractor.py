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

    def __init__(self, embedding_input_size):
        super().__init__()
        self.n_features = embedding_input_size
        self.lstm = LSTM(embedding_input_size, hidden_size=int(embedding_input_size / 2), bidirectional=True, num_layers=2)

        self.to_binary_logits = nn.Linear(embedding_input_size, 2)

        self.gumbel_softmax = nn.functional.gumbel_softmax


    def forward(self, embedding, hard=False):
        '''
        Creates a rational for the given embedding.
        end_index is upto which index we need to get the rational (for the rest of the index we do not create a rational
        '''
        lstm_out, (hidden, cell) = self.lstm(embedding)
        binary_logits = self.to_binary_logits(lstm_out)
        if self.training and hard == False:
            probs = self.gumbel_softmax(binary_logits, hard=True, dim=-1) ###TODO: Maybe set hard to false?
            mask = (probs[:, :, 1] + 1 - probs[:, :, 0]) / 2
        else:
            probs = self.gumbel_softmax(binary_logits, hard=True, dim=-1)
            mask = ((probs[:, :, 1] + 1 - probs[:, :, 0]) / 2).bool()
        h_repeated = mask.unsqueeze(-1).repeat(1, 1, embedding.shape[-1])

        masked_embedding = h_repeated * embedding

        return {"masked_embedding": masked_embedding, "mask": mask}

    def save(self, location):
    
        torch.save({
            'model_state_dict': self.state_dict(),
            'kwargs': {
                'embedding_input_size': self.n_features,
            }
        }, location)

    @classmethod
    def load(self, location):
        
        info = torch.load(location)
        model = RationalExtractor(**info['kwargs'])
        model.load_state_dict(info['model_state_dict'])
        model.train()
        return model