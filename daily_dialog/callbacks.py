import os

import pytorch_lightning as pl


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

    def on_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:

            completed_sentences = pl_module.complete_sentences(self.sentences, self.reaction_length)
            for i, s in enumerate(completed_sentences):
                print("----- ",i, '\n', s)
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

    def on_epoch_end(self, trainer, pl_module):
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
                for i, (completed_sentence, start_dialogue) in enumerate(
                        zip(completed_sentences, self.start_of_dialogues)):
                    title = '_'.join(start_dialogue[7:].replace("?", "").split())

                    text_file = '.\\' + log_dir + "\{}_{}_{}_{}.txt".format(title, trainer.current_epoch,
                                                                            "with_rational" if self.with_rational else "no_rational",
                                                                            "greedy" if self.greedy_policy else "probs")
                    ### We write it to a text file:
                    print(text_file)
                    with open(text_file, 'w+', encoding='utf-8') as f:
                        for rationalized_input, response in zip(completed_sentence["rationalized_input"],
                                                                completed_sentence["response"]):
                            f.write(rationalized_input + '->' + response + '\n')
                        f.write(completed_sentence["completed_dialogue"])

        pl_module.train()
