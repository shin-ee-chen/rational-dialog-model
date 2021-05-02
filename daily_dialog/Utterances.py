import datasets
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import itertools


class Utterances(Dataset):

    def __init__(self, tokenizer, size=None, transform=None, subsets="start", split="train", batch_size=64):
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
        self.batch_size = batch_size
        self.dataset = self.process_dataset(subsets)

    def process_dataset(self, subsets):

        # Split all the dialogues in subdialogues
        dialogue_samples = itertools.chain.from_iterable([
            self.subdialogues(dialogue["dialog"], subsets) 
            for dialogue in self.original_dataset
        ])
        print("Processed dataset, length = ", len(self.original_dataset))

        # Tokenize the samples
        tokenized_samples = [
            (self.tokenizer.encode(context).ids, self.tokenizer.encode(response).ids)
            for (context, response) in dialogue_samples
        ]
        print("Tokenized samples, length = ", len(tokenized_samples))
        print("Example: ", tokenized_samples[0])
        print("Decoded context: ", self.tokenizer.decode(tokenized_samples[0][0]))
        print("Decoded reponse: ", self.tokenizer.decode(tokenized_samples[0][1]))

        # Now sort the samples on length
        sorted_samples = sorted(tokenized_samples, key=lambda x: len(x[0]))

        # Put samples in batches
        final_samples = []
        current_batch = []
        for r in sorted_samples:

            # If batch is full. Start the new batch
            if len(current_batch) >= self.batch_size:

                # First add padding if not all samples have same length
                padded_batch = self.add_padding(current_batch)
                final_samples.append(padded_batch)

                # Start new batch
                current_batch = [r]

            # We simply add to the batch
            else:
                current_batch.append(r)
        
        # Add last (possibly shorter) batch to final
        if len(current_batch) > 0:
            padded_batch = self.add_padding(current_batch)
            final_samples.append(padded_batch)

        if self.size:
            final_samples = final_samples[:self.size]
        return final_samples

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

    def add_padding(self, batch):
        ''' Add padding if not all samples have same length
        '''
        longest = len(batch[-1][0])       # Length of context in last sample in batch
        dif = longest - len(batch[0][0])  # Difference in length between last and first sample
        if dif > 0:
 #           print("Padding needed: ", dif)
            padded = [(
                list(np.pad(b[0], (0, longest - len(b[0])), constant_values = torch.tensor(2))), # Padding code
                b[1])
                for b in batch
            ]
        else:
            padded = batch
        return padded

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def postprocess_dataloader_out(x):
    in_tensors = [torch.cat(b[0]) for b in x]

    # Make batch second
    rational_in = torch.stack(in_tensors).permute(1, 0, )
    target_tensor = [torch.cat(b[1]) for b in x]

    # Sometimes we need to pad the target sequences
    target = pad_sequence(target_tensor)

    return rational_in, target
