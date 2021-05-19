import math
from modules.LanguageModels.BaseLanguageModel import BaseLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class PretrainedLanguageModel(BaseLanguageModel):
    ''''
    Load a pretrained DialoGPT language model
    '''

    def __init__(self, pretrained_model='microsoft/DialoGPT-small'):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(pretrained_model)

        self.embedding = self.lm.get_input_embeddings()
        self.layers = self.lm.get_output_embeddings()

    def forward(self, tokenized_input_ids):
        # return self.model.generate(tokenized_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        return self.lm(tokenized_input_ids).logits

    def save(self, location):
        self.lm.save_pretrained(location)
        print("Model saved")

    def to_embedding(self, x):
        return self.embedding(x)

    def forward_embedding(self, embedding):
        return self.lm(inputs_embeds=embedding).logits

    def generate_next_tokens_from_embedding(self, embedding, n_tokens=10):
        pass
        # tokens = []
        # ## Initialize:
        # logits = self.forward_embedding(embedding)
        # logits = logits[-1]
        # next_token = self.get_next_token_from_logits(logits)

        # tokens.append(next_token)
        # next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
        # next_embedding = self.embedding(next_token_tensor)
        # for i in range(n_tokens - 1):
        #     next_embedding = next_embedding.reshape(1, 1, -1)

        #     logits = self.forward_embedding(next_embedding)

        #     next_token = self.get_next_token_from_logits(logits)

        #     tokens.append(next_token)
        #     next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
        #     next_embedding = self.embedding(next_token_tensor)
        # return tokens
    

    @classmethod
    def load(self, location):
        self.lm = AutoModelForCausalLM.from_pretrained(location)
        self.embedding = self.lm.get_input_embeddings()
        self.layers = self.lm.get_output_embeddings()
        return self.lm