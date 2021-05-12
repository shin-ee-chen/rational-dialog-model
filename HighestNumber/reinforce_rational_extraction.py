import torch
from torch import nn
from torch.nn import Embedding, LSTM
import torch.nn.functional as F
import pytorch_lightning as pl


class ReinforceRationalExtractor(nn.Module):

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
            print("greedy")
            mask = torch.argmax(policy_reshaped, dim=-1).bool()
        else:
            mask = torch.multinomial(policy_reshaped, 1).bool()

        mask = mask.reshape(x.shape[0], -1)

        masked_input = torch.mul(x, mask) + ~mask * self.mask_token

        mask_to_gather = mask.reshape(x.shape[0], -1, 1).long()

        chosen_policy = torch.gather(policy, -1, mask_to_gather)

        return {"mask_logits": logits, "policy": policy, "chosen_policy": chosen_policy, "mask": mask,
                "masked_input": masked_input}


class HighestNumberLSTM(nn.Module):

    def __init__(self, embedding_input=11, embedding_size=32, out_size=5):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = Embedding(embedding_input, embedding_size)

        # Layers for prediction
        self.prediction_LSTM = LSTM(embedding_size, hidden_size=embedding_size)
        self.output_layer = nn.Linear(embedding_size, out_size)

    def forward(self, x):
        e = self.embedding(x)

        # Calculate output on masked embeddings
        h, (h_n, c_n) = self.prediction_LSTM(e)
        logits = self.output_layer(h_n)

        return {"logits": logits.squeeze(0)}


class ReinforceModelPL(pl.LightningModule):

    def __init__(self, lstm, rational_extractor, loss_module, hparams=None, pretrain=True):
        super().__init__()
        self.hparams = hparams
        self.lstm = lstm
        self.loss_module = loss_module
        self.rational_extractor = rational_extractor
        self.pretrain = pretrain
        self.log_list = [
            "loss", "acc", "h_loss",
        ]

    def forward(self, x, greedy=False):

        if self.pretrain:
            return self.lstm.forward(x)
        else:
            rational = self.rational_extractor.forward(x, greedy=greedy)
            masked_x = rational["masked_input"]

            return {**self.lstm.forward(masked_x), **rational}

    def to_batch_out(self, batch, greedy=False):
        # Change to long and
        x = batch[0].long().permute(1, 0, )
        targets = batch[1].long()
        out = self.forward(x, greedy=greedy)

        if self.pretrain:
            predictions = out["logits"]

            h_loss = 0


            loss = self.loss_module(predictions, targets) + h_loss
            acc = self.calc_acc(predictions, targets)



            return {"loss": loss, "acc": acc, "h_loss": h_loss}
        else:
            predictions = out["logits"]

            h_loss = 0
            h_loss_step = 0
            if "mask" in out.keys():
                h = out["mask"].permute(1, 0).float()

                h_loss = torch.mean(h, dim=-1)

                h_loss_step = torch.mean(h_loss)

            score = self.loss_module(predictions, targets, ) + h_loss

            loss =  torch.mean(score.detach() * torch.log(out['chosen_policy']))
            acc = self.calc_acc(predictions, targets)
            return {"loss": loss, "acc": acc, "h_loss": h_loss_step, "mask": out["mask"]}

    def training_step(self, batch, batch_idx):
        results = self.to_batch_out(batch)

        self.log_results(results)

        return results["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        results = self.to_batch_out(batch, greedy=True)
        if "mask" in results.keys():
            print(results["mask"].permute(1, 0).float())

        self.log_results(results, prepend="val")

        return results["loss"]

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
