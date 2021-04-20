import pytorch_lightning as pl



class FinishSentenceCallback(pl.Callback):
    '''
    Callback that Lets the model finish the given sentences.
    '''
    def __init__(self, sentences, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.sentences = sentences
        self.reaction_length = 100

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        """
        pl_module.eval()
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
                completed_sentences = pl_module.complete_sentences(self.sentences, self.reaction_length)
                print(completed_sentences)
        pl_module.train()


