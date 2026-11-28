from math import sqrt
from torch import nn
from transformers import AutoConfig, AutoTokenizer

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained(model_ckpt)

text = "time flies like an arrow"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
print(f"Input encodings are: {inputs.input_ids}")

decoded = tokenizer.decode(inputs.input_ids[0])
print(f"Decoded tokens: {decoded}")

print(f"the vocabulary size is: {config.vocab_size}")
print(f"the hidden size is: {config.hidden_size}")

token_emb = nn.Embedding(config.vocab_size, config.hidden_size).to(device)
inputs_embeds = token_emb(inputs.input_ids)
print(f'the input embedding size is: {inputs_embeds.size()}')

# Lets postpone adding positional embeddings to this input_embeds
query = key = value = inputs_embeds
dim_k = key.size(-1)
scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
scores.size()
print('this is just for debugging')















