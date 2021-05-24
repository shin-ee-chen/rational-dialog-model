import os
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning import LightningModule


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

    def on_train_epoch_end(self, trainer, pl_module, outputs):
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
                        f.write('\n'.join([utterance.strip() for utterance in completed_sentence["completed_dialogue"].split('[SEP]')]))
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
        total_diff_perplexity = 0
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:

            mean_diff = calc_change_in_perplexity(pl_module, self.dataloader)

            trainer.logger.log_metrics({"mean_diff_perplexity": mean_diff}, step=(trainer.current_epoch + 1))

        pl_module.train()

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

            analysis = rational_analysis(pl_module, self.dataloader)

            trainer.logger.log_metrics({
                "Absolute mask position": analysis["abs_average"],
                "Relative mask position": analysis["rel_average"],
            }, step=(trainer.current_epoch + 1))

        pl_module.train()

