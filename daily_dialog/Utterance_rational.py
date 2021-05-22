import torch
from utils.token_utils import get_token_id

def get_utterance_representations(context, tokenizer, embed_fn):

    # Split context in utterances. Utterances are separated by sep_token
    sep_id = get_token_id(tokenizer, "sep_token")
    utterances = context.split(sep_id)
    utterance_lengths = [len(u) + 1 for u in utterances]

    # Calculate the utterance representation by taking average of the token embeddings
    pad_id = get_token_id(tokenizer, "pad_token")
    utterance_reps = [
        torch.mean([embed_fn(token_id) for token_id in utterance if token_id != pad_id])
        for utterance in utterances
    ]

    return (utterance_reps, utterance_lengths)

def get_masked_context(context, utterance_mask, utterance_lengths, tokenizer):
    '''
    utterance_mask: list of True, False; True means whole utterance is masked
    utterance_lengths: list of lenghts of the utterances in context
    Returns the context, with mask_tokens for complete utterances if they are masked
    '''

    assert sum(utterance_lengths) = len(context), "length of context should be equal to sum of length of utterances"
    
    mask_id = get_token_id(tokenizer, "mask_token")
    mask = [[masked] * length for (masked, length) in zip(utterance_mask, utterance_lengths)]
    masked_context = [not(m) * c + m * mask_id for (c, m) in zip(context, mask)]
    return masked_context

