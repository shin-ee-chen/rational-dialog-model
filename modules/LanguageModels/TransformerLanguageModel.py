import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from torch.nn.utils.rnn import PackedSequence
import numpy as np

from modules.LanguageModels.BaseLanguageModel import BaseLanguageModel


class TransformerLM(BaseLanguageModel):
    ''''
    A simple lstm language model
    '''

    def __init__(self, num_embeddings, embedding_dim=128, num_head=2, num_hid=2, num_layers=2, dropout=0.5):
        super(TransformerLM, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_head, num_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_head = num_head
        self.num_hid = num_hid
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_embeddings = num_embeddings
        self.decoder = nn.Linear(embedding_dim, num_embeddings)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def to_embedding(self, x):
        return self.embedding(x)

    def forward(self, x):
        src_mask = self.generate_square_subsequent_mask(x.size(0)).to(x.device)
        return self.forward_with_mask(x, src_mask)

    def forward_with_mask(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


    def generate_next_token(self, sentence_ids):
        sentence_ids = sentence_ids.unsqueeze(1)
        embedding = self.embedding(sentence_ids)
        src = embedding * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        logits = self.decoder(output)[-1, 0, :]

        next_token = self.get_next_from_logits(logits)

        next_token = torch.tensor([next_token]).to(sentence_ids.device)

        return next_token


    def save(self, location):
        torch.save({
            'model_state_dict': self.state_dict(),
            'kwargs': {
                'num_embeddings': self.num_embeddings,
                'num_layers': self.num_layers,
                'embedding_dim': self.embedding_dim,
                'num_hid': self.num_hid,
                'num_head': self.num_head,
                'dropout': self.dropout,
            }

        }, location)

    @classmethod
    def load(self, location):
        info = torch.load(location)
        model = TransformerLM(**info['kwargs'])
        model.load_state_dict(info['model_state_dict'])
        model.train()
        return model


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
