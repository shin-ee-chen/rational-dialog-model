import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.utils.rnn import PackedSequence
import numpy as np

from PredictionDataset import postprocess_dataloader_out


class LSTMLM(nn.Module):
    ''''
    A simple lstm language model
    '''

    def __init__(self, num_embeddings, num_layers=2, embedding_dim=128, hidden_state_size=128, batch_first=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_state_size = hidden_state_size

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_state_size, num_layers=num_layers,
                            batch_first=batch_first)
        self.relu = F.relu
        self.classification_layer = nn.Linear(hidden_state_size, num_embeddings)

    def to_embedding(self, x):
        return self.embedding(x)

    def forward_embedding(self, embedding):
        out, hidden = self.lstm(embedding)

        out = self.relu(out)
        classification = self.classification_layer(out, )

        return classification

    def forward(self, x):
        if type(x) == PackedSequence:
            return self.packed_forward(x)
        embedding = self.to_embedding(x)

        out, hidden = self.lstm(embedding)

        out = self.relu(out)
        classification = self.classification_layer(out, )

        return classification

    def packed_forward(self, x):
        data = x.data

        embedding = self.embedding(data)

        embedding = PackedSequence(embedding, x.batch_sizes, sorted_indices=x.sorted_indices,
                                   unsorted_indices=x.unsorted_indices)
        out, hidden = self.lstm(embedding)
        out = F.relu(out.data)
        result = self.classification_layer(out.reshape(-1, self.hidden_state_size))
        return PackedSequence(result, x.batch_sizes, sorted_indices=x.sorted_indices,
                              unsorted_indices=x.unsorted_indices)



    def generate_next_tokens_from_embedding(self, embedding, n_tokens=10):
        tokens = []
        ## Initialize:
        out, hidden = self.lstm(embedding)

        out = F.relu(out[-1, 0])  # Get the latest token

        logits = self.classification_layer(out)
        next_token = self.get_next_from_logits(logits)

        tokens.append(next_token)
        next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
        next_embedding = self.embedding(next_token_tensor)
        for i in range(n_tokens - 1):
            next_embedding = next_embedding.reshape(1, 1, -1)

            out, hidden = self.lstm(next_embedding, hidden)

            out = F.relu(out)
            logits = self.classification_layer(out)
            next_token = self.get_next_from_logits(logits)

            tokens.append(next_token)
            next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
            next_embedding = self.embedding(next_token_tensor)
        return tokens

    def complete_sentence(self, sentence_ids, max_length=100):
        tokens = []
        ## Initialize:
        sentence_ids = sentence_ids.unsqueeze(1)
        embedding = self.embedding(sentence_ids)
        out, hidden = self.lstm(embedding)
        out = F.relu(out[-1, 0])  # Get the latest token

        logits = self.classification_layer(out)
        next_token = self.get_next_from_logits(logits)

        tokens.append(next_token)
        next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
        next_embedding = self.embedding(next_token_tensor)
        for i in range(max_length):
            next_embedding = next_embedding.reshape(1, 1, -1)

            out, hidden = self.lstm(next_embedding, hidden)

            out = F.relu(out)
            logits = self.classification_layer(out)
            next_token = self.get_next_from_logits(logits)

            tokens.append(next_token)
            next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
            next_embedding = self.embedding(next_token_tensor)
        return tokens

    def get_next_from_logits(self, logits, top=10):
        logits = logits.flatten().detach().cpu().numpy()
        top_indices = logits.argsort()[::-1][:top]
        top_logits = logits[top_indices]
        p = np.exp(top_logits) / sum(np.exp(top_logits))

        index = np.random.choice(top_indices, p=p)

        return index

    def save(self, location):
        torch.save({
            'model_state_dict': self.state_dict(),
            'kwargs': {
                'num_embeddings': self.num_embeddings,
                'num_layers': self.num_layers,
                'embedding_dim': self.embedding_dim,
                'hidden_state_size': self.hidden_state_size,
            }

        }, location)

    @classmethod
    def load(self, location):
        info = torch.load(location)
        model = LSTMLM(**info['kwargs'])
        model.load_state_dict(info['model_state_dict'])
        model.train()
        return model


class PredictionLMPL(pl.LightningModule):

    def __init__(self, lstm, rational_extractor, tokenizer, loss_module, hparams=None, ):
        super().__init__()
        self.hparams = hparams
        self.lstm = lstm
        self.rational_extractor = rational_extractor
        self.loss_module = loss_module
        self.tokenizer = tokenizer

        self.log_list = [
            "loss", "acc", "h_loss", "h_mean"
        ]

    def forward(self, x, targets):

        rational = self.get_rational(x)
        target_embedding = self.lstm.to_embedding(targets)

        masked_embedding = rational['masked_embedding']

        ## Concatenate the two together and put through the lstm
        lstm_in = torch.cat([masked_embedding, target_embedding])

        prediction = self.lstm.forward_embedding(lstm_in)

        return {"logits": prediction, **rational}

    def get_rational(self, x):
        rational_embedding = self.lstm.to_embedding(x)
        rational = self.rational_extractor.forward(rational_embedding)
        return rational

    def training_step(self, batch, batch_idx):
        # Change to long and
        rational_in, targets = postprocess_dataloader_out(batch)

        out = self.forward(rational_in, targets)

        targets = targets.long()

        n_targets = targets.shape[0]

        predictions = out["logits"][-(n_targets + 1):-1]
        h_loss = 0
        h_mean = 0
        if "h" in out.keys():
            h = out["h"].permute(1, 0)
            h_mean = torch.mean(h)

            h_loss = h_mean

        loss = self.loss_module(predictions.view(-1, self.tokenizer.get_vocab_size()), targets.flatten()) + h_loss

        acc = self.calc_acc(predictions, targets)

        self.log_results({"loss": loss, "acc": acc, "h_loss": h_loss, "h_mean": h_mean})

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        rational_in, targets = postprocess_dataloader_out(batch)

        out = self.forward(rational_in, targets)

        targets = targets.long()

        n_targets = targets.shape[0]

        predictions = out["logits"][-(n_targets + 1):-1]
        h_loss = 0
        h_mean = 0
        if "h" in out.keys():
            h = out["h"].permute(1, 0)
            h_mean = torch.mean(h)

            h_loss = 0.01 * h_mean

        loss = self.loss_module(predictions.view(-1, self.tokenizer.get_vocab_size()), targets.flatten()) + h_loss

        acc = self.calc_acc(predictions, targets)

        self.log_results({"loss": loss, "acc": acc, "h_loss": h_loss, "h_mean": h_mean}, prepend="val_")

        return loss

    def complete_sentences(self, sentences, total_length):
        return [self.complete_sentence(sentence, total_length=total_length) for sentence in sentences]



    def complete_sentence(self, sentence, n_rational=10, total_length=100, with_rational=True):

        encoding = self.tokenizer.encode(sentence)
        all_tokens = encoding.ids

        ids_tensor = torch.tensor(all_tokens).to(self.device)
        ids_tensor = ids_tensor.unsqueeze(1)
        rationals = []
        sentences = []
        rationalized_input = []
        while(len(ids_tensor)) < total_length:

            # Extract rationals if needed.
            if with_rational and len(ids_tensor) > n_rational:
                rational = self.get_rational(ids_tensor)

                rational_input =(ids_tensor * rational["h"]).int().flatten().detach().cpu().numpy()
                rational_input =self.tokenizer.decode(rational_input, skip_special_tokens=False).replace(" #", "").replace("#", "")
                rationalized_input.append(rational_input)

                rationals.append(rational["h"].flatten())
                embedding = rational["masked_embedding"]
            else:
                rational_input = self.tokenizer.decode(ids_tensor.flatten().flatten().detach().cpu().numpy(), skip_special_tokens=False).replace(" #",
                                                                                                          "").replace(
                    "#", "")
                rationalized_input.append(rational_input)
                rationals.append(torch.tensor([]))
                embedding = self.lstm.to_embedding(ids_tensor)

            next_ids = self.lstm.generate_next_tokens_from_embedding(embedding, n_tokens=n_rational)


            all_tokens += next_ids
            sentences.append(self.tokenizer.decode(next_ids, skip_special_tokens=False).replace(" #", "").replace("#",  ""))
            ids_tensor = torch.tensor(all_tokens).to(self.device)

            ids_tensor = ids_tensor.unsqueeze(1)

        sentence = self.tokenizer.decode(all_tokens, skip_special_tokens=False).replace(" #", "").replace("#",  "")


        return {"complete_sentence": sentence, "rationals": rationals, "rationalized_input":rationalized_input, "response": sentences}


    def configure_optimizers(
            self,
    ):
        parameters = list(self.lstm.parameters()) + list(self.rational_extractor.parameters())

        optimizer = torch.optim.Adam(
            parameters,
            lr=self.hparams['learning_rate'])
        return optimizer

    def log_results(self, result, prepend=""):

        for k in self.log_list:
            self.log(prepend + k, result[k], on_step=True, on_epoch=True)

    def calc_acc(self, predictions, targets):
        indices = torch.argmax(predictions, dim=-1)

        correct = indices == targets
        return torch.mean(correct.float())
