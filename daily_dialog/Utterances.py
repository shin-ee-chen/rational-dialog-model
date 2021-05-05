import datasets
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import itertools
import random


class Utterances(Dataset):

    def __init__(self, tokenizer, size=None, subsets="start", split="train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert (subsets == "start") or (subsets == "full"), "subsets should be 'start' or 'full'"
        self.original_dataset = datasets.load_dataset("daily_dialog", split=split, )
        self.tokenizer = tokenizer
        self.size = size
        self.subsets = subsets
        self.dataset = self.process_dataset(subsets)

    def process_dataset(self, subsets):

        n = self.size if self.size else len(self.original_dataset)

        # Split all the dialogues in subdialogues
        dialogue_samples = itertools.chain.from_iterable([
            self.subdialogues(dialogue["dialog"], subsets) 
            for i, dialogue in enumerate(self.original_dataset) if i < n  
        ])

        # Tokenize the samples
        tokenized_samples = [
            (self.tokenizer.encode(context).ids, self.tokenizer.encode(response).ids)
            for (context, response) in dialogue_samples
        ]

        # Now shuffle samples and sort on length (to prevent amount of padding in batches)
        random.shuffle(tokenized_samples)
        sorted_samples = sorted(tokenized_samples, key=lambda x: len(x[0]))
        return sorted_samples

    def subdialogues(self, utterances, subsets):
        '''
        Create subsets of the dialogue.
        Input is a list of utterances.
        Ouput is a list with sub-dialogues with (context, reponse) pairs.
        If subsets == 'full', then return ALL subsets of utterances
        If subsets == 'start', then all subsets start from the first utterances
        '''
        num = len(utterances)
        if num < 2:
            return None
        startrange = 1 if subsets == "start" else num - 1
        subsets = [
            ('[SEP]'.join(utterances[start:end]), utterances[end])
            for start in range(0, startrange, 1)
            for end in range(start + 1, num, 1)
        ]
        return subsets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def reshuffle_dataset(self):
        '''
        Reshuffles the dataset
        '''
        self.dataset = self.process_dataset(self.subsets)

    def collate_fn(items):

        inputs = pad_sequence(
            [torch.tensor(sample[0]) for sample in items],
            batch_first = True, 
            padding_value = torch.tensor(2) # Padding code
        ) 
        targets = pad_sequence(
            [torch.tensor(sample[1]) for sample in items], 
            batch_first = True, 
            padding_value = torch.tensor(2) # Padding code
        )

        return inputs, targets

