from torch.utils.data import Dataset
import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
import itertools
import random


class UtterancesDataset(Dataset):

    def __init__(self, tokenizer, size=None, subsets="start", perturbation=None, split="train", ):
        assert subsets in ["start", "end", "single", "full"], "subsets should be 'start', 'end', 'single, or 'full'"
        assert perturbation in [None, "utterance_dialogue", "words_dialogue", "words_utterance"]
        self.original_dataset = datasets.load_dataset("daily_dialog", split=split, )
        self.dialogues = [item["dialog"] for item in self.original_dataset]
        self.size = len(self.dialogues)
        if (size != None) and (size > 0) and (size < self.size):
            self.dialogues = random.sample(self.dialogues, size)
            self.size = size
        self.tokenizer = tokenizer
        self.subsets = subsets
        self.dataset = self.process_dataset(subsets, perturbation)


    def process_dataset(self, subsets, perturbation):
        '''
        Returns tokenized subsets of the dataset.
        Output is a list with (context, response) pairs, sorted on combined length
        '''

        # Split all the dialogues in subdialogues --> ([utterances in context], reponse)
        dialogue_samples = list(itertools.chain.from_iterable([
            self.subdialogues(dialogue, subsets)
            for dialogue in self.dialogues
        ]))

        # Perturn the context, based on specified perturbation option
        perturbed_samples = self.perturb_dataset(dialogue_samples, perturbation)

        # Tokenize the samples
        tokenized_samples = [
            (self.tokenizer.encode((' [SEP] ').join(context) + ' [SEP] ').ids, self.tokenizer.encode(response + ' [SEP] ').ids)
            for (context, response) in perturbed_samples
        ]

        # Now shuffle samples and sort on length (to prevent amount of padding in batches)
        random.shuffle(tokenized_samples)
        sorted_samples = sorted(tokenized_samples, key=lambda x: len(x[0]) +  len(x[1]) + random.randint(0, 10))
        return sorted_samples

    def perturb_dataset(self, samples, perturbation):
        # print("DEBUG: Perturbation = ", perturbation)

        if perturbation == None:

            # No perturbation, so just return the dialogues in normal order
            result = samples

        elif perturbation == "utterance_dialogue":

            # Shuffle order of utterances within dialogue
            result = [(random.sample(context, len(context)), response) for (context, response) in samples]

        elif perturbation == "words_utterance":

            # Shuffle order of words within utterance, but keep order of utterances
            result = [
                ([self.shuffled_utterance(u) for u in context], response) 
                for (context, response) in samples
            ]

        elif perturbation == "words_dialogue":

            # Shuffle order of words accross the whole dialogue
            result = [
                ([self.shuffled_utterance(" ".join(context))], response) 
                for (context, response) in samples
            ]

        else:
            raise AssertionError("Unkown perturbation: " + perturbation)

        # print("DEBUG: result of perturbations")
        # print(result)
        return result


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
        # print("DEBUG subdialogues ", subsets, '\n', utterances)
        l = len(utterances)
        if l < 2:
            return None
        if subsets == "single":
            results = [
                ([utterances[start]], utterances[start + 1])
                for start in range(0, l - 1)
            ]
        elif subsets == "start":
            results = [
                (utterances[0:end], utterances[end])
                for end in range(1, l)
            ]
        elif subsets == "end":
            results = [
                (utterances[start:l - 1], utterances[l - 1])
                for start in range(0, l - 1)
            ]
        elif subsets == "full":
            results = [
                (utterances[start:end], utterances[end])
                for start in range(0, l - 1)
                for end in range(start + 1, l)
            ]
        else:
            results = None

        # for i, s in enumerate(results):
        #     print(i, s)
        return results

    def shuffled_utterance(self, utterance):
        words = utterance.split()
        random.shuffle(words)
        return(' '.join(words))

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
    def get_collate_fn(padding_value=2):
        def collate_fn(items):
            inputs = torch.fliplr(pad_sequence(
                [torch.tensor(list(reversed(sample[0]))) for sample in items],
                batch_first=True,
                padding_value=padding_value  # Padding code
            ))
            targets = pad_sequence(
                [torch.tensor(sample[1]) for sample in items],
                batch_first=True,
                padding_value=padding_value  # Padding code
            )
            # print("Result of collate")
            # print("Inputs: ", inputs)
            # print("Targets: ", targets)
            return inputs, targets
        return collate_fn
