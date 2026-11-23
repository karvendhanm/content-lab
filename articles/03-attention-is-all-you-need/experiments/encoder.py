from transformers import AutoConfig, AutoTokenizer

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model_ckpt = "bert-base-uncased"
tokenzier = AutoTokenizer.from_pretrained(model_ckpt, device=device)
config = AutoConfig.from_pretrained(model_ckpt, device=device)

text = "time flies like an arrow"
inputs = tokenzier(text, return_tensors="pt", add_special_tokens=False)
print(f"Input encodings are: {inputs.input_ids}")

decoded = tokenzier.decode(inputs.input_ids[0])
print(f"Decoded tokens: {decoded}")

print(f"the vocabulary size is: {config.vocab_size}")
print(f"the hidden size is: {config.hidden_size}")



