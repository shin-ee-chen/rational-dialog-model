import os

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


class FinishSentenceRationalizedCallback(pl.Callback):
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

        log_dir = trainer.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        pl_module.eval()
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
                completed_sentences = pl_module.complete_sentences(self.sentences, self.reaction_length)
                for i, (completed_sentence, sentence) in enumerate(zip(completed_sentences, self.sentences)):
                    title = '_'.join(sentence[7:].replace("?", "").split())

                    text_file = '.\\' + log_dir  + "\{}_{}.txt".format(title, trainer.current_epoch)
                    ### We write it to a text file:
                    print(text_file)
                    with open(text_file, 'w+', encoding='utf-8') as f:
                        for rationalized_input, response in zip(completed_sentence["rationalized_input"], completed_sentence["response"]):
                            f.write(rationalized_input + '->' + response + '\n')
                        f.write(completed_sentence["complete_sentence"])

        pl_module.train()