from torch.utils.data import Dataset
import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
import itertools
import random


class UtterancesDataset(Dataset):

    def __init__(self, tokenizer, size=None, subsets="start", split="train"):
        assert subsets in ["start", "end", "single", "full"], "subsets should be 'start', 'end', 'single, or 'full'"
        self.original_dataset = datasets.load_dataset("daily_dialog", split=split, )
        self.tokenizer = tokenizer
        self.size = size
        self.subsets = subsets
        self.dataset = self.process_dataset(subsets)

    def process_dataset(self, subsets):
        '''
        Returns tokenized subsets of the dataset.
        Output is a list with (context, response) pairs, sorted on combined length
        '''

        n = self.size if self.size else len(self.original_dataset)

        # Split all the dialogues in subdialogues
        dialogue_samples = list(itertools.chain.from_iterable([
            self.subdialogues(dialogue["dialog"], subsets)
            for i, dialogue in enumerate(self.original_dataset) if i < n
        ]))

        # Tokenize the samples
        tokenized_samples = [
            (self.tokenizer.encode(context).ids, self.tokenizer.encode(response).ids)
            for (context, response) in dialogue_samples
        ]

        # Now shuffle samples and sort on length (to prevent amount of padding in batches)
        random.shuffle(tokenized_samples)
        sorted_samples = sorted(tokenized_samples, key=lambda x: len(x[0]) * 10 + len(x[1]) + random.randint(0, 10))
        return sorted_samples

    def subdialogues(self, utterances, subsets):
        '''
        Create subsets of the dialogue.
        Input is a list of utterances.
        Output is a list with sub-dialogues with (context, reponse) pairs.
        If subsets == 'full', then return ALL subsets of utterances
        If subsets == 'start', then all subsets start from the first utterances
        If subsets == 'end', the all subsets END at utterance length-1
        If subsets == 'single', context is just a single utterance
        '''
        l = len(utterances)
        if l < 2:
            return None
        if subsets == "single":
            results = [
                (utterances[start] + '[SEP]', utterances[start + 1] + '[SEP]')
                for start in range(0, l - 1)
            ]
        elif subsets == "start":
            results = [
                ('[SEP]'.join(utterances[0:end]) + '[SEP]', utterances[end] + '[SEP]')
                for end in range(1, l)
            ]
        elif subsets == "end":
            results = [
                ('[SEP]'.join(utterances[start:l - 1]) + '[SEP]', utterances[l - 1] + '[SEP]')
                for start in range(0, l - 1)
            ]
        elif subsets == "full":
            results = [
                ('[SEP]'.join(utterances[start:end]) + '[SEP]', utterances[end] + '[SEP]')
                for start in range(0, l - 1)
                for end in range(start + 1, l)
            ]
        else:
            results = None
        # print(subsets, '\n', utterances)
        # for i, s in enumerate(results):
        #     print(i, s)
        return results

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def reshuffle_dataset(self):
        '''
        Reshuffles the dataset
        '''
        self.dataset = sorted(self.dataset, key=lambda x: len(x[0]) * 10 + len(x[1]) + random.randint(0, 10))

    @staticmethod
    def collate_fn(items):

        inputs = torch.fliplr(pad_sequence(
            [torch.tensor(list(reversed(sample[0]))) for sample in items],
            batch_first=True,
            padding_value=torch.tensor(0)  # Padding code
        ))
        targets = pad_sequence(
            [torch.tensor(sample[1]) for sample in items],
            batch_first=True,
            padding_value=torch.tensor(0)  # Padding code
        )
        return inputs, targets
