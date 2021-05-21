from tokenizers import Tokenizer



special_tokens = {
    "unk_token": "[UNK]", 
    "sep_token": "[SEP]", 
    "pad_token": "[PAD]", 
    "mask_token": "[MASK]",
    "start_token": "[START]",
    "rmask_token": "[RMASK]", 
}

def get_token(tokenizer, token_type):

    if type(tokenizer) == Tokenizer:
        token_str = special_tokens[token_type]
    else:
        token_str = {
            "unk_token": tokenizer.unk_token, 
            "sep_token": tokenizer.sep_token, 
            "pad_token": tokenizer.pad_token, 
            "mask_token": tokenizer.mask_token,
            "start_token": tokenizer.bos_token,
        }[token_type]

    return token_str

def get_token_id(tokenizer, token_type):

    if type(tokenizer) == Tokenizer:
        token_id = tokenizer.token_to_id(special_tokens[token_type])
    else:
        token_id = {
            "unk_token": tokenizer.unk_token_id, 
            "sep_token": tokenizer.sep_token_id, 
            "pad_token": tokenizer.pad_token_id, 
            "mask_token": tokenizer.mask_token_id,
            "start_token": tokenizer.bos_token_id,
        }[token_type]

    return token_id


def get_vocab_size(tokenizer):
    if type(tokenizer) == Tokenizer:
        return tokenizer.get_vocab_size()
    else:
        return len(tokenizer)