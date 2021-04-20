'''
Pytorch lightning version of the rationalized language model.
Used for convenience and reproducability.
'''
import pytorch_lightning as pl
import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from daily_dialog.RationalExtractor import sample_hardkurma
from daily_dialog.language_model import PackedEmbedding, PackedReLU, PackedLinear
from utils import to_packed_sequence


class RationalLMPL(pl.LightningModule):

    def __init__(self, language_model, tokenizer, loss_module, hparams=None, ):
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.loss_module = loss_module
        self.log_list = [
            "loss"
        ]
        self.hparams = hparams

    def complete_sentences(self, sentences, max_length):
        return [self.complete_sentence(sentence, max_length) for sentence in sentences]

    def complete_sentence(self, sentence, max_length):
        encoding = self.tokenizer.encode(sentence)
        ids_tensor = torch.tensor(encoding.ids).to(self.device)

        completed_sentence_tokens = self.language_model.complete_sentence(ids_tensor, max_length)
        new_sentence = sentence + str(
            self.tokenizer.decode(completed_sentence_tokens, skip_special_tokens=False)).replace(" #", "").replace("#",
                                                                                                                   "")

        return new_sentence

    def batch_to_out(self, batch):
        '''
        Should return a dictionary with at least the loss inside
        :param batch:
        :param batch_idx:
        :return: dict with {"loss": loss} and other values once finds relevant
        '''

        encodings = self.tokenizer.encode_batch(batch)

        ids_tensor = torch.tensor([encoding.ids for encoding in encodings]).to(self.device)

        input_tensor = to_packed_sequence(ids_tensor[:, :-1]).to(self.device)

        target_tensor = to_packed_sequence(ids_tensor[:, 1:], target=1).to(self.device)
        out = self.language_model.forward(input_tensor)
        predictions = out["logits"]
        h_loss = 0
        if "h" in out.keys():
            h = out["h"].data
            h_loss += 0.0001 * torch.mean(h)


        loss = self.loss_module(predictions.data, target_tensor.data, ) + h_loss

        return {"loss": loss, 'predictions': predictions}

    def training_step(self, batch, batch_idx):
        result = self.batch_to_out(batch)
        self.log_results(result)

        return result["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        result = self.batch_to_out(batch)

        self.log_results(result, prepend="val ")

    def configure_optimizers(
            self,
    ):
        parameters = list(self.language_model.parameters())

        optimizer = torch.optim.Adam(
            parameters,
            lr=self.hparams['learning_rate'])
        return optimizer

    def log_results(self, result, prepend=""):
        for k in self.log_list:
            self.log(prepend + k, result[k], on_step=True, on_epoch=True)


class PackedRationalLSTMLM(nn.Module):
    ''''
    A simple lstm language model
    '''

    def __init__(self, num_embeddings, num_layers=2, embedding_dim=256, hidden_state_size=256):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_state_size = hidden_state_size

        self.embedding = PackedEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.kur_lstm = nn.LSTM(embedding_dim, hidden_size=int(embedding_dim / 2), bidirectional=True, num_layers=2, batch_first=True)
        self.kur_layer = PackedKumaraswamyLayer(embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_state_size, num_layers=num_layers,
                            batch_first=True, dropout=0.5)
        self.relu = PackedReLU()
        self.classification_layer = PackedLinear(hidden_state_size, num_embeddings)

    def to_embedding(self, x):
        return self.embedding(x)

    def forward(self, x):

        embedded = self.to_embedding(x)

        ##Apply Kumaraswamy.
        out_kur, hidden_kur = self.kur_lstm(embedded)
        h = self.kur_layer(out_kur)

        ##Apply kur on packed sequence.


        new_embedding = PackedSequence( h.data * embedded.data, x.batch_sizes, sorted_indices=x.sorted_indices,
                       unsorted_indices=x.unsorted_indices)


        out, hidden = self.lstm(new_embedding)

        out = self.relu.forward(out)
        logits = self.classification_layer(out, )

        return {"logits": logits, "h": h}

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


class PackedKumaraswamyLayer(nn.Module):

    def __init__(self, in_features, l=-0.1, r=1.1):
        super().__init__()
        self.in_features = in_features

        self.layer_a = nn.Linear(in_features, 1)
        self.layer_b = nn.Linear(in_features, 1)

        self.l = l
        self.r = r
        self.softplus = nn.Softplus()

    def forward(self, x):
        data = x.data

        data = data.reshape(-1, self.in_features)

        a = self.softplus(self.layer_a(data))

        b = self.softplus(self.layer_b(data))

        a = a.clamp(1e-5, 10.)  # extreme values could result in NaNs
        b = b.clamp(1e-5, 10.)  # extreme values could result in NaNs

        h = sample_hardkurma(data.unsqueeze(1), a, b, self.l, self.r)

        return PackedSequence(h, x.batch_sizes, sorted_indices=x.sorted_indices,
                              unsorted_indices=x.unsorted_indices)
