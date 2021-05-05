'''
Experiment with Daily Dialog dataset
'''
from torch.utils.data import DataLoader
from daily_dialog.Utterances import Utterances
from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer
import datasets


original_dataset = datasets.load_dataset("daily_dialog", split="train", )
example = original_dataset[0]["dialog"]
print("----------\nExample in original dataset\nLength = ", len(example), "\n", example)

def subs(utterances):
    num = len(utterances)
    if num < 2:
        return None
    subsets = [
        ('[SEP]'.join(utterances[start:end]), utterances[end])
        for start in range(0, num-1)
        for end in range(start+1, num)
    ]
    return subsets

# for i, u in enumerate(subs(example)):
#     print(i, u)


batch_size = 256
my_tokenizer = get_daily_dialog_tokenizer(tokenizer_location='./daily_dialog/tokenizer.json', )

# tokenized_example = my_tokenizer.encode('[SEP]'.join(example)).ids
# print("tokenized: ", tokenized_example)
# decoded_example = my_tokenizer.decode(tokenized_example)
# print("decoded:   ", decoded_example)

dataset_train = Utterances(my_tokenizer, split="train", batch_size=batch_size, subsets="full")
dataloader_train = DataLoader(dataset_train, batch_size=1, )

def verify_batches(dl):
    bnum = 0
    for batch in dl:
        print("Batch #",bnum, "len: ", len(batch), "min: ", len(batch[0][0]), "max: ", len(batch[-1][0]))
        (context, response) = batch[0]
        print("---------- ",bnum)
        print("\n", context, '\n', response)
        print("Context: ", my_tokenizer.decode(context))
        print("Response:", my_tokenizer.decode(response))
        bnum += 1

#verify_batches(dataloader_train)