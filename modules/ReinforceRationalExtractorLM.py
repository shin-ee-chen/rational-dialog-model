import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import Embedding, LSTM
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AdamW

from daily_dialog.NextNPredictionDataset import postprocess_dataloader_out
from utils import fussed_lasso


class ReinforceRationalExtractorLM(nn.Module):

    def __init__(self, embedding_input=11, embedding_size=32, mask_token=0):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = Embedding(embedding_input, embedding_size)

        # Layers for prediction
        self.prediction_LSTM = LSTM(embedding_size, hidden_size=embedding_size)
        self.output_layer = nn.Linear(embedding_size, 2)
        self.mask_token = mask_token

    def forward(self, x, greedy=False):
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

        mask_to_gather = mask.reshape(x.shape[0], -1, 1).long()

        chosen_policy = torch.gather(policy, -1, mask_to_gather)

        return {"mask_logits": logits, "policy": policy, "chosen_policy": chosen_policy, "mask": mask,
                "masked_input": masked_input}


class RELMPL(pl.LightningModule):
    '''
    PL wrapper for training a language model together with a rational extractor.
    '''

    def __init__(self, language_model, rational_extractor, tokenizer, loss_module, hparams=None,
                 sparsity_weight=1,
                 fussed_lasso_weight=1, ):
        super().__init__()
        self.hparams = hparams
        self.language_model = language_model
        self.rational_extractor = rational_extractor
        self.loss_module = loss_module
        self.tokenizer = tokenizer

        self.sparsity_weight = sparsity_weight
        self.fussed_lasso_weight = fussed_lasso_weight

        self.log_list = [
            "loss", "acc", "h_loss", "h_mean", "fussed_lasso"
        ]
        self.teacher_forcing = hparams["teacher_forcing"]
        self.freeze_language_model = hparams["freeze_language_ml"]

    def forward(self, x, targets, ):

        rational = self.rational_extractor(x)

        masked_input = rational["masked_input"]

        ## Concatenate the two together and put through the lstm
        lstm_in = torch.cat([masked_input, targets])
        prediction = self.language_model(lstm_in)
        return {"logits": prediction, **rational}

    def batch_out(self, batch):
        rational_in, targets = postprocess_dataloader_out(batch)

        out = self.forward(rational_in, targets)

        targets = targets.long()

        n_targets = targets.shape[0]

        if self.teacher_forcing:

            predictions = out["logits"][-(n_targets + 1):-1]
        else:
            predictions = out["logits"]
        h_loss = 0
        h_mean = 0
        fussed_lasso_loss = 0
        if "mask" in out.keys():
            h = out["mask"].permute(1, 0).float()
            h_mean = torch.mean(h, dim=-1)
            fussed_lasso_loss = fussed_lasso(h, reduce=False)
            h_loss = self.sparsity_weight * h_mean + self.fussed_lasso_weight * fussed_lasso_loss

        if type(self.tokenizer) == Tokenizer:
            rewards = self.loss_module(predictions.view(-1, self.tokenizer.get_vocab_size()),
                                       targets.flatten()) + h_loss


        else:
            rewards = self.loss_module(predictions.view(-1, self.tokenizer.vocab_size), targets.flatten(),
                                       reduce=False) + h_loss

        loss = torch.mean(rewards.detach() * torch.log(out["chosen_policy"]))
        # Get the policy loss.
        if not self.freeze_language_model:
            loss += torch.mean(rewards)

        acc = self.calc_acc(predictions, targets)
        return {"loss": loss, "acc": acc, "h_loss": h_loss.mean(), "h_mean": h_mean.mean(), "fussed_lasso": fussed_lasso_loss.mean()}

    def training_step(self, batch, batch_idx):

        batch_out = self.batch_out(batch)

        self.log_results(batch_out)

        return batch_out["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_out = self.batch_out(batch)

        self.log_results(batch_out, prepend="val_")

        return batch_out["loss"]

    def complete_dialogues(self, sentences, total_length):
        return [self.complete_dialogue(sentence, total_length=total_length) for sentence in sentences]

    def complete_dialogue(self, sentence, n_rational=10, total_length=100, with_rational=True):

        if type(self.tokenizer) == Tokenizer:
            encoding = self.tokenizer.encode(sentence).ids
        else:
            encoding = self.tokenizer.encode(sentence)

        all_tokens = encoding

        ids_tensor = torch.tensor(all_tokens).to(self.device)
        ids_tensor = ids_tensor.unsqueeze(1)
        rationals = []
        sentences = []
        rationalized_input = []
        while (len(ids_tensor)) < total_length:

            # Extract rationals if needed.
            if with_rational and len(ids_tensor) > n_rational:
                rational = self.get_rational(ids_tensor)

                rational_input = (ids_tensor * rational["h"]).int().flatten().detach().cpu().numpy()
                rational_input = self.tokenizer.decode(rational_input, skip_special_tokens=False).replace(" #",
                                                                                                          "").replace(
                    "#", "")
                rationalized_input.append(rational_input)

                rationals.append(rational["h"].flatten())
                embedding = rational["masked_embedding"]
            else:
                rational_input = self.tokenizer.decode(ids_tensor.flatten().flatten().detach().cpu().numpy(),
                                                       skip_special_tokens=False).replace(" #",
                                                                                          "").replace(
                    "#", "")
                rationalized_input.append(rational_input)
                rationals.append(torch.tensor([]))
                embedding = self.language_model.to_embedding(ids_tensor)

            next_ids = self.language_model.generate_next_tokens_from_embedding(embedding, n_tokens=n_rational)

            all_tokens += next_ids
            sentences.append(
                self.tokenizer.decode(next_ids, skip_special_tokens=False).replace(" #", "").replace("#", ""))
            ids_tensor = torch.tensor(all_tokens).to(self.device)

            ids_tensor = ids_tensor.unsqueeze(1)

        sentence = self.tokenizer.decode(all_tokens, skip_special_tokens=False).replace(" #", "").replace("#", "")

        return {"complete_sentence": sentence, "rationals": rationals, "rationalized_input": rationalized_input,
                "response": sentences}

    def configure_optimizers(
            self,
    ):
        if not self.freeze_language_model:
            parameters = list(self.language_model.parameters()) + list(self.rational_extractor.parameters())
        else:
            parameters = list(self.rational_extractor.parameters())

        optimizer = AdamW(
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
