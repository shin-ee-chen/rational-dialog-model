import pytorch_lightning as pl
import torch


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

class RationaleAnalysisCallback(pl.Callback):
    '''
    Callback to analyse the rationales
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

            n = 0
            abs_averages = 0.0
            rel_averages = 0.0
            for batch in self.dataloader:

                # This assumes a batch consists of a list of (context, response) pairs
                context = torch.tensor([c for (c, _) in batch])
                context = context.to(pl_module.device)
#                print("Context: ", context)

                # Get the mask
                rational = pl_module.get_rational(context)
                mask = rational["mask"]
#                print("Mask: ", mask, mask.size(), len(mask[0]))

                num_positions = len(mask[0])
                positions_reversed = torch.tensor(list(range(num_positions, 0, -1)))
#                print(positions_reversed)

                mask_positions = torch.mul(mask, positions_reversed).float()
#                print("Positions: ", mask_positions)
                average_absolute = mask_positions.sum(dim=1)/mask.sum(dim=1)
                average_relative = average_absolute / num_positions
                # print("Average absolute: ", average_absolute)
                # print("Average relative: ", average_relative)
                abs_averages += torch.mean(average_absolute)
                rel_averages += torch.mean(average_relative)
                n += len(context)

            trainer.logger.log_metrics({
                "Absolute mask position": abs_averages / n, 
                "Relative mask position": rel_averages / n,
            }, step=(trainer.current_epoch + 1))

        pl_module.train()

