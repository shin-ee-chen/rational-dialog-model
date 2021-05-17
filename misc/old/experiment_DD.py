'''
Experiment with Daily Dialog dataset
'''
from torch.utils.data import DataLoader
from daily_dialog.UtterancesDataset import UtterancesDataset
from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer
import datasets


# original_dataset = datasets.load_dataset("daily_dialog", split="train", )
# example = original_dataset[0]["dialog"]
# print("----------\nExample in original dataset\nLength = ", len(example), "\n", example)


batch_size = 8
my_tokenizer = get_daily_dialog_tokenizer(tokenizer_location='./daily_dialog/tokenizer.json', )

# tokenized_example = my_tokenizer.encode('[SEP]'.join(example)).ids
# print("tokenized: ", tokenized_example)
# decoded_example = my_tokenizer.decode(tokenized_example)
# print("decoded:   ", decoded_example)

dataset_train = UtterancesDataset(my_tokenizer, split="train", subsets="start", perturbation="words_dialogue", size=1)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=UtterancesDataset.collate_fn)

def verify_batches(dl):
    bnum = 0
    for batch in dl:
        print("Batch #",bnum, "len: ", len(batch[0]))
        (contexts, responses) = batch
        print("---------- ",bnum)
        for i in range(len(contexts)):
            print(i, contexts[i], responses[i])
            print("Context: ", my_tokenizer.decode(contexts[i].tolist()))
            print("Response:", my_tokenizer.decode(responses[i].tolist()))
        bnum += 1

verify_batches(dataloader_train)