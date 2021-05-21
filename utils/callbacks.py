import os
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
import numpy as np


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
            for index, sentence in enumerate(completed_sentences):
                try:
                    print("\n----- ", index, '\n', sentence)
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
                completed_sentences = pl_module.complete_dialogues(
                    self.start_of_dialogues, 
                    self.reaction_length,
                    with_rational=self.with_rational,
                    greedy_rationals=self.greedy_policy
                )
                for i, (completed_sentence, start_dialogue) in enumerate(zip(completed_sentences, self.start_of_dialogues)):
                    title = '_'.join(start_dialogue[7:].replace("?", "").split())

                    text_file = os.path.join(
                        log_dir, 
                        "{}_{}_{}_{}.txt".format(
                            title, 
                            trainer.current_epoch,
                            "with_rational" if self.with_rational else "no_rational",
                            "greedy" if self.greedy_policy else "probs"
                    ))

                    ### We write it to a text file:
                    print(text_file)
                    with open(text_file, 'w+', encoding='utf-8') as f:
                        f.write('Rationalized input ------> Response\n')
                        for rationalized_input, response in zip(completed_sentence["rationalized_input"], completed_sentence["response"]):
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

            for (contexts, targets) in self.dataloader:

                contexts = contexts.to(pl_module.device)
                targets = targets.to(pl_module.device)

                # Make batch second
                contexts = contexts.permute(1, 0, )
                targets = targets.permute(1, 0, )
                n_targets = targets.shape[0]

                # Get the rational
                rational = pl_module.get_rational(contexts)

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
            for (contexts, _) in self.dataloader:

                # This assumes a batch consists of a list of (context, response) pairs
                contexts = contexts.to(pl_module.device)
#                print("Context: ", context)

                # Get the mask
                rational = pl_module.get_rational(contexts)
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
                abs_averages += torch.sum(average_absolute)
                rel_averages += torch.sum(average_relative)
                n += len(contexts)

            trainer.logger.log_metrics({
                "Absolute mask position": abs_averages / n, 
                "Relative mask position": rel_averages / n,
            }, step=(trainer.current_epoch + 1))

        pl_module.train()

