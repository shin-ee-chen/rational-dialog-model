import math
from modules.LanguageModels.BaseLanguageModel import BaseLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class DailoGPTLanguageModel(BaseLanguageModel):
    ''''
    Load a pretrained DialoGPT language model
    '''

    def __init__(self, pretrained_model='microsoft/DialoGPT-small'):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def forward(self, tokenized_input_ids):
        return self.pretrained_model.generate(tokenized_input_ids, max_length=1000, pad_token_id=self.pretrained_tokenizer.eos_token_id)

    def save(self, location):
        print("save")
        # torch.save({
        #     'model_state_dict': self.state_dict(),
        #     'kwargs': {
        #         'num_embeddings': self.num_embeddings,
        #         'num_layers': self.num_layers,
        #         'embedding_dim': self.embedding_dim,
        #         'hidden_state_size': self.hidden_state_size,
        #     }
        # }, location)