'''
Pytorch lightning version of the rationalized language model.
Used for convenience and reproducability.
'''
import pytorch_lightning as pl
import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from old.language_model import PackedEmbedding, PackedReLU, PackedLinear
from modules.packed import PackedGumbellLayer
from utils import to_packed_sequence, get_next_input_ids, get_packed_mean


class RationalLMPL(pl.LightningModule):

    def __init__(self, language_model, rational_extractor, tokenizer, loss_module, hparams=None, ):
        super().__init__()
        self.language_model = language_model
        self.rational_extractor = rational_extractor
        self.tokenizer = tokenizer
        self.loss_module = loss_module
        self.log_list = [
            "loss"
        ]
        self.hparams = hparams

        self.count = 0

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

    def batch_to_out(self, batch, h_weight=1e-4):
        '''
        Should return a dictionary with at least the loss inside
        :param batch:
        :return: dict with {"loss": loss} and other values once finds relevant
        '''
        encodings = self.tokenizer.encode_batch(batch)
        loss = 0

        n_rational = 500

        # Construct the input tensors
        ids_tensor = torch.tensor([encoding.ids for encoding in encodings]).to(self.device)
        input_tensor = to_packed_sequence(ids_tensor).to(self.device)

        padded_input_tensor = pad_packed_sequence(input_tensor, batch_first=True)

        n_steps = int(ids_tensor.shape[1] / n_rational) + 1
        losses = []
        for i in range(n_steps):
            start_index = i * n_rational
            end_index = (i + 1) * n_rational

            target_ids = ids_tensor[:, start_index + 1: end_index + 1].to(self.device)

            h = 0
            h_loss = 0

            if i != 0:
                # Calculate the rational:
                rational_input_ids = get_next_input_ids(padded_input_tensor, end_index=i * n_rational)
                non_rational_input_ids = get_next_input_ids(padded_input_tensor, end_index=(i + 1) * n_rational)
                rationalized_embedding = self.language_model.to_embedding(rational_input_ids)
                non_rationalized_embedding = self.language_model.to_embedding(non_rational_input_ids)

                rational = self.rational_extractor.forward(rationalized_embedding)

                embedding = rational['masked_embedding']

                # Next we need to combine the two embeddings.
                rationalized_padded_embedding = pad_packed_sequence(embedding, batch_first=True, total_length=end_index)
                non_rationalized_padded_embedding = pad_packed_sequence(non_rationalized_embedding, batch_first=True,
                                                                        total_length=end_index)

                is_rationalized_mask = torch.zeros(non_rationalized_padded_embedding[0].shape).to(self.device)
                is_rationalized_mask[:, :start_index, :] += 1
                # Make it a boolean
                is_rationalized_mask = is_rationalized_mask == 1
                embedding = rationalized_padded_embedding[0] * is_rationalized_mask + non_rationalized_padded_embedding[
                    0] * ~is_rationalized_mask

                embedding = pack_padded_sequence(embedding, non_rationalized_padded_embedding[1], batch_first=True,
                                                 enforce_sorted=False)

                h = rational["h"]

                mean = get_packed_mean(h)
                print(mean)
                ## Problems with calculating the mean for a packed sequence.
                h_loss = h_weight * mean
            else:
                next_input_ids = get_next_input_ids(padded_input_tensor, end_index=(i + 1) * n_rational)
                embedding = self.language_model.to_embedding(next_input_ids)

            predictions = self.language_model.forward_embedding(embedding)

            padded_predictions = pad_packed_sequence(predictions, batch_first=True)
            upto_prediction = min(start_index + target_ids.shape[-1], end_index)
            predictions_to_score = padded_predictions[0][:, start_index: upto_prediction]

            losses.append(self.loss_module(predictions_to_score.reshape(-1, self.tokenizer.get_vocab_size()),
                                           target_ids.flatten()) + h_loss)

        losses = torch.stack(losses)

        loss = torch.mean(losses)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self.batch_to_out(batch, 0.0001)
        self.log_results(result)

        return result["loss"]

    def on_epoch_end(self) -> None:
        print(self.count)
        self.count += 1

    def get_next_h_weight(self):

        return 1 / 40 * self.count

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        result = self.batch_to_out(batch)

        self.log_results(result, prepend="val ")

    def configure_optimizers(
            self,
    ):
        parameters = list(self.language_model.parameters()) + list(self.rational_extractor.parameters())

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

    def __init__(self, num_embeddings, num_layers=1, embedding_dim=256, hidden_state_size=256):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_state_size = hidden_state_size

        self.embedding = PackedEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.kur_lstm = nn.LSTM(embedding_dim, hidden_size=int(embedding_dim / 2), bidirectional=True, num_layers=1,
                                batch_first=True)
        self.kur_layer = PackedGumbellLayer(embedding_dim)

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

        new_embedding = PackedSequence(h.data.view(-1, 1) * embedded.data, x.batch_sizes,
                                       sorted_indices=x.sorted_indices,
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

