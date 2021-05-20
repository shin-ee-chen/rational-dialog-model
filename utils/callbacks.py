import os
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from misc.old.NextNPredictionDataset import postprocess_dataloader_out


class FinishDialogueCallback(pl.Callback):
    '''
    Callback that Lets the model finish the given dialogue.
    Print out the results
    '''

    def __init__(self, sentences, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.sentences = sentences
        self.reaction_length = 100


    def on_train_epoch_end(self, trainer, pl_module: LightningModule, outputs: Any):
        pl_module.eval()
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:

            completed_sentences = pl_module.complete_dialogues(self.sentences, self.reaction_length)
            for i, s in enumerate(completed_sentences):
                try:
                    print("\n----- ", i, '\n', s)
                except:
                    #UnicodeEncodeError: 'latin-1' codec can't encode character '\u2019' in position 21: ordinal not in range(256)
                    print("[can't generate response!]")
        pl_module.train()


class ReshuffleDatasetCallback(pl.Callback):
    '''
    A callback that can reshuffle certain datasets that allow it.
    '''

    def __init__(self, dataset, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.dataset = dataset

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        """

        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.dataset.reshuffle_dataset()


class FinishDialogueRationalizedCallback(pl.Callback):
    '''
    Callback that Lets the model finish the given dialogue it extracts a rational at the same time.
    The resulted dialogue get's put into a text file in the log directory
    '''

    def __init__(self, start_of_dialogues, every_n_epochs=1, with_rational=True, greedy_policy=False):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.start_of_dialogues = start_of_dialogues
        self.reaction_length = 100
        self.with_rational = with_rational
        self.greedy_policy = greedy_policy

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        """
        This function is called after every epoch.
        """

        log_dir = trainer.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        pl_module.eval()
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
                completed_sentences = pl_module.complete_dialogues(self.start_of_dialogues, self.reaction_length,
                                                                   with_rational=self.with_rational,
                                                                   greedy_rationals=self.greedy_policy)
                for i, (completed_sentence, start_dialogue) in enumerate(zip(completed_sentences, self.start_of_dialogues)):
                    title = '_'.join(start_dialogue[7:].replace("?", "").split())

                    text_file = os.path.join(log_dir, "{}_{}_{}_{}.txt".format(title, trainer.current_epoch,
                                                                            "with_rational" if self.with_rational else "no_rational",
                                                                            "greedy" if self.greedy_policy else "probs"))
                    ### We write it to a text file:
                    print(text_file)
                    with open(text_file, 'w+', encoding='utf-8') as f:
                        f.write('Rationalized input ------> Response\n')
                        for rationalized_input, response in zip(completed_sentence["rationalized_input"],
                                                                completed_sentence["response"]):
                            f.write(rationalized_input + ' ------> ' + response + '\n')
                        f.write('\n\nCompleted dialogue:\n')    
                        f.write(completed_sentence["completed_dialogue"])
        pl_module.train()


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
                context, targets =

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
