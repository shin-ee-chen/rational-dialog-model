import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import itertools
flatten = itertools.chain.from_iterable


class PredictionDataset(Dataset):

    def __init__(self, tokenizer, size=None,transform=None, split="train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.original_dataset = datasets.load_dataset("", split=split, )
        self.tokenizer = tokenizer
        # Next we process the dataset to split it up properly.
        self.size = size
        self.dataset = self.process_dataset()

    def process_dataset(self):
        ### First we tokenize each example
        dialogues = ['[START] ' + '[SEP]'.join(dialogue["dialog"]) for dialogue in self.original_dataset]
        tokenized_dataset = [self.tokenizer.encode(sample).ids for sample in dialogues]

        prediction_size = 10
        print(len(tokenized_dataset))
        ### Next we split it up
        result = []
        for sample in tokenized_dataset:
            splitted_sample = [(sample[:i], sample[i:i + prediction_size]) for i in
                               range(prediction_size, len(sample), prediction_size)]

            result += splitted_sample

        sorted_results = sorted(result, key=lambda x: len(x[0]))

        ### Next we want to split it up in even sample_sizes:
        len_results = [len(r[0]) for r in sorted_results]

        #### Naive way: pick
        final_samples = []
        batch_size = 64

        current_sample_len = 0
        current_batch = []
        for r in sorted_results:
            current_length = len(r[0])
            ## If there is no
            if len(current_batch) == 0:
                current_sample_len = current_length
                current_batch.append(r)
            # If either we have not the same lenght or the batch is full. Start the new ba
            elif len(r[0]) != current_sample_len or len(current_batch) >= batch_size:
                final_samples.append(current_batch)
                current_batch = [r]
                current_sample_len = current_length
            # We simply add to the batch
            else:
                current_batch.append(r)

        ### TO speed up we only use sampls with size 64

        final_samples = [sample for sample in final_samples if len(sample) == batch_size]
        if self.size:
            final_samples = final_samples[:self.size]
        return final_samples

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
