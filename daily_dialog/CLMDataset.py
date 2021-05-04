'''
The dataset of causal language modelling.
Batch same sizes together such that we use minimal padding.
'''
from collections import Counter

import datasets
import torch

from torch.utils.data import Dataset, DataLoader
import itertools
import numpy as np

flatten = itertools.chain.from_iterable


class CLMDataset(Dataset):

    def __init__(self, tokenizer, size=None, transform=None, split="train", batch_size=16):
        self.original_dataset = datasets.load_dataset("daily_dialog", split=split, )
        self.tokenizer = tokenizer
        # Next we process the dataset to split it up properly.
        self.size = size
        self.batch_size = batch_size
        self.dataset = self.process_dataset()

    def process_dataset(self):
        '''
        Batch same sizes together. To get minimal overhead when applying padding.
        '''
        sorted_results = self.get_sorted_samples()

        #### Naive way: pick
        final_samples = []

        current_sample_len = 0
        current_batch = []
        for r in sorted_results:

            if len(current_batch) == 0:
                current_batch.append(r)
            # Keep adding until the batch is full
            elif len(current_batch) >= self.batch_size:
                final_samples.append(current_batch)
                current_batch = [r]
            # We simply add to the batch
            else:
                current_batch.append(r)

        if self.size:
            final_samples = final_samples[:self.size]

        ### Lastly we must make sure the padding is there. Hence we decode and than batch encode to automatically pad the samples.
        result = []
        for final_sample in final_samples:
            temp = [' '.join(s) for s in final_sample]
            result.append(torch.stack([torch.tensor(enc.ids) for enc in self.tokenizer.encode_batch(temp)]))

        return result

    def get_sorted_samples(self):
        '''
        Sorts the samples in lists of approximatly the same size.
        '''
        dialogues = ['[START] ' + '[SEP]'.join(dialogue["dialog"]) for dialogue in self.original_dataset]
        tokenized_dataset = [self.tokenizer.encode(sample).tokens for sample in dialogues]

        sorted_results = sorted(tokenized_dataset, key=lambda x: len(x))

        # Next we create lists of utterances that are approximatly the same size. We shuffle the utterances within such a list.
        # This has a result that when we batch them we end up with batches that have utterances of approximatly the same length.
        results = []
        current = []
        current_len = len(sorted_results[0])
        for r in sorted_results:
            if abs(current_len - len(r)) <= 15:
                current.append(r)
            else:
                # shuffle the set itself.
                np.random.shuffle(current)
                results += current
                current = [r]
                current_len = len(r)
        return results

    def reshuffle_dataset(self):
        '''
        Reshuffles the dataset
        '''
        self.dataset = self.process_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
