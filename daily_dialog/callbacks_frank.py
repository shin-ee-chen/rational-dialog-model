import pytorch_lightning as pl


class PerturbationCallback(pl.Callback):
    '''
    Callback to evaluate results on perturbated dataset
    '''

    def __init__(self, dataloader, every_n_epochs=1):
        '''
        Dataloader: the data with the perturbated dataset
        '''
        super().__init__()
        self.dataloader = dataloader
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module, outputs):

        pl_module.eval()
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:

            print('Evaluate on perturbated dataset')
            total_perplexity = 0

            for (contexts, targets) in self.dataloader:
                contexts = contexts.to(pl_module.device)
                targets = targets.to(pl_module.device)
                n_targets = targets.shape[0]

                # Get the rational and the masked input
                rational = pl_module.get_rational(contexts)
                masked_input = rational["masked_input"]

                # Get logits and calculate perplexity
                logits = pl_module.forward_masked_input(masked_input, targets)[-(n_targets + 1):-1]
                perplexity = pl_module.get_perplexity(logits, targets)
                total_perplexity += perplexity

            mean_perplexity = total_perplexity/len(self.dataloader)

            trainer.logger.log_metrics({"perplexity perturbated": mean_perplexity}, step=(trainer.current_epoch + 1))

        pl_module.train()

