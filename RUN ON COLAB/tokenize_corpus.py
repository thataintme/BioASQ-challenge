# This code is confusing because of Huggingface's definition of "encode"
# "Encode" in huggingface is the job of the tokenizer which splits a long sentence into tokens and then into numbers
#   representing those tokens. This is just a lookup from a fixed vocabulary. There is no meaning attached to these tokens.
#   when two tokens have close integer values, that does not imply they are close in meaning.


# But here we are getting an "Encoder model" from MedCPT, whose task is to turn the above taken IDs into meanings.
# The "Encoder" here is the "Encoder" half of the transformer.
# It takes the integer IDs, looks up a learned vector for each, then passes them through layers of attention so every token’s
# vector gets updated based on the whole context.

# For ease this brain-fry I renamed the tokenizer output to "tokenized" instead of the conventional "enc"


import torch
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")

batch = [[d["title"], d["abstract"]] for d in docs[:32]]   # title + abstract pairs
with torch.no_grad():
    tokenized = tok(batch, truncation=True, padding=True, max_length=512, return_tensors="pt")
    vecs = model(**tokenized).last_hidden_state[:, 0, :]   # one 768-dim vector per abstract