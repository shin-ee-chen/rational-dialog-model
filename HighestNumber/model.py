import torch
import pytorch_lightning as pl


class LstmPL(pl.LightningModule):

    def __init__(self, lstm, loss_module, hparams=None, ):
        super().__init__()
        self.hparams = hparams
        self.lstm = lstm
        self.loss_module = loss_module

        self.log_list = [
            "loss", "acc", "h_loss",
        ]

    def forward(self, x):
        return self.lstm.forward(x)

    def training_step(self, batch, batch_idx):
        # Change to long and
        x = batch[0].long().permute(1, 0, )
        targets = batch[1].long()
        out = self.forward(x)

        predictions = out["logits"]

        h_loss = 0
        if "h" in out.keys():
            h = out["h"].permute(1, 0)

            h_loss += 0.1 * torch.mean(h)

        loss = self.loss_module(predictions, targets) + h_loss

        acc = self.calc_acc(predictions, targets)

        self.log_results({"loss": loss, "acc": acc, "h_loss": h_loss})

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Change to long and
        x = batch[0].long().permute(1, 0, )
        targets = batch[1].long()
        out = self.forward(x)

        predictions = out["logits"]

        h_loss = 0
        if "h" in out.keys():
            h = out["h"].permute(1, 0)
            print(h)
            h_loss += 0.1 * torch.mean(h)
        loss = self.loss_module(predictions, targets) + h_loss
        acc = self.calc_acc(predictions, targets)

        self.log_results({"loss": loss, "acc": acc, "h_loss": h_loss}, prepend="val")

        return loss

    def configure_optimizers(
            self,
    ):
        parameters = list(self.lstm.parameters())

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
