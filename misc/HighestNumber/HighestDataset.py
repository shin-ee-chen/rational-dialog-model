from torch.utils.data import Dataset
import numpy as np


class HighestDataset(Dataset):

    def __init__(self, length=5, n_numbers=10, n_examples=int(10e4)):
        self.length = length
        self.n_numbers = n_numbers
        self.n_examples = n_examples
        self.dataset = self.init_dataset()

    def init_dataset(self):
        dataset = []

        for i in range(self.n_examples):
            choices = np.random.choice(self.n_numbers, self.length, replace=False)
            max_index = np.argmax(choices)
            dataset.append((choices, max_index))
        return dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.n_examples