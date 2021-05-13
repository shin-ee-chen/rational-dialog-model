import os

import pytorch_lightning as pl
import numpy as np

from daily_dialog.NextNPredictionDataset import postprocess_dataloader_out


class ChangeInPerplexityCallback(pl.Callback):
    '''
    Callback that Lets the model finish the given dialogue.
    Print out the results
    '''

    def __init__(self, dataloader, n_extra_masks=1, every_n_epochs=1, mask=4, weight=None):
        '''
        Dataloader: the data which we check the change in perplexity
        n_extra_mask: how many tokens we mask extra.
        '''
        super().__init__()
        self.dataloader = dataloader
        self.n_extra_mask = n_extra_masks
        self.every_n_epochs = every_n_epochs
        self.mask = mask
        self.weight = weight

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        pl_module.eval()
        print('change in perplexity')
        total_diff_perplexity = 0
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:

            for batch in self.dataloader:
                context, targets = postprocess_dataloader_out(batch)

                context = context.to(pl_module.device)
                targets = targets.to(pl_module.device)
                n_targets = targets.shape[0]
                # Get the rational
                rational = pl_module.get_rational(context)
                # Mask n extra tokens (in each rational)

                masked_input = rational["masked_input"]



                logits = pl_module.forward_masked_input(masked_input, targets)[-(n_targets + 1):-1]
                extra_masked = self.mask_extra(masked_input, self.n_extra_mask)
                logits_altered = pl_module.forward_masked_input(extra_masked, targets)[-(n_targets + 1):-1]
                # Get the perplexity


                perplexity_non_altered = pl_module.get_perplexity(logits, targets, weights=self.weight)
                perplexity_altered = pl_module.get_perplexity(logits_altered, targets, weights=self.weight)

                diff = (perplexity_non_altered - perplexity_altered).mean()
                total_diff_perplexity += diff

            mean_diff = total_diff_perplexity/len(self.dataloader)

            trainer.logger.log_metrics({"mean_diff_perplexity": mean_diff}, step=(trainer.current_epoch + 1))

        pl_module.train()

    def mask_extra(self, masked_input, n_extra_mask):
        # First find all the locations in which there is no mask
        result = masked_input

        n_batches = masked_input.shape[1]
        non_mask_indices = (masked_input != self.mask).nonzero().cpu().numpy()

        # Next we extract for each batch extra tokens to mask
        for batch_number in range(n_batches):

            # Filter the possible locations
            # Get sequence index of the current batch where we do not have a mask
            mask = non_mask_indices[:, 1] == batch_number #THe mask
            possible_locations = non_mask_indices[mask][:, 0] #The actual indices

            # Pick a random_ones (make sure we do not pick more than there are available)
            n_mask = min(len(possible_locations), n_extra_mask)
            locations = np.random.choice(possible_locations, n_mask)

            # Mask those locations
            for location in locations:
                result[location, batch_number] = self.mask

        return result
