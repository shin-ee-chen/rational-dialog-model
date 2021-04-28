'''
The dataset of causal language modelling.
Batch same sizes together such that we use minimal padding.
'''
from collections import Counter

import datasets
import torch

from torch.utils.data import Dataset, DataLoader
import itertools

flatten = itertools.chain.from_iterable


class CLMDataset(Dataset):

    def __init__(self, tokenizer, size=None, transform=None, split="train"):
        self.original_dataset = datasets.load_dataset("daily_dialog", split=split, )
        self.tokenizer = tokenizer
        # Next we process the dataset to split it up properly.
        self.size = size
        self.dataset = self.process_dataset()

    def process_dataset(self):
        '''
        Batch same sizes together. To get minimal overhead when applying padding.
        '''
        ### First we tokenize each example
        dialogues = ['[START] ' + '[SEP]'.join(dialogue["dialog"]) for dialogue in self.original_dataset]
        tokenized_dataset = [self.tokenizer.encode(sample).tokens for sample in dialogues]

        sorted_results = sorted(tokenized_dataset, key=lambda x: len(x))


        #### Naive way: pick
        final_samples = []
        batch_size = 16

        current_sample_len = 0
        current_batch = []
        for r in sorted_results:

            ## If there is no
            if len(current_batch) == 0:
                current_batch.append(r)
            # If either we have not the same length or the batch is full. Start the new ba
            elif len(current_batch) >= batch_size:
                final_samples.append(current_batch)
                current_batch = [r]
            # We simply add to the batch
            else:
                current_batch.append(r)

        if self.size:
            final_samples = final_samples[:self.size]

        ### Lastly we must make sure the padding is there. Hence we decode and then do batch encode
        result = []
        for final_sample in final_samples:
            temp = [' '.join(s) for s in final_sample]
            result.append(torch.stack([torch.tensor(enc.ids) for enc in self.tokenizer.encode_batch(temp)]))

        return result

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
