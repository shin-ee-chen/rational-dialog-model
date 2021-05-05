'''
Code to create a tokenizer for the daily dialog.
'''
from tokenizers.pre_tokenizers import Whitespace
import datasets
from tokenizers import Tokenizer, normalizers

from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.trainers import WordPieceTrainer

special_tokens = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "mask_token": "[MASK]",
                  "rmask_token": "[RMASK]", "start_token": "[START]"}


def get_daily_dialog_tokenizer(tokenizer_location=None):
    '''
    Get the daily diolog tokenizer. Trains a new one if no location is provided
    :param tokenizer_location: Json containing information about the tokenizer.
    :return:
    '''
    if tokenizer_location:
        tokenizer = Tokenizer.from_file(tokenizer_location, )
        tokenizer.enable_padding()
        return tokenizer
    else:

        dataset_train = datasets.load_dataset("daily_dialog", split="train", )

        utterances = ['[SEP]'.join(dialogue["dialog"]) for dialogue in dataset_train]

        # Write every dialogue to file

        location = './daily_dialog/'

        trainer = WordPieceTrainer(
            vocab_size=2096, special_tokens=["[PAD]", "[UNK]", "[SEP]", "[MASK]", "[RMASK]"]  # RMASK = rational mask
        )

        custom_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]", ))
        custom_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        custom_tokenizer.pre_tokenizer = Whitespace()

        custom_tokenizer.train_from_iterator(utterances, trainer, )
        custom_tokenizer.save(location + "tokenizer.json")
        custom_tokenizer.enable_padding()
        return custom_tokenizer
