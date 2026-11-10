# dense_search.py
import faiss, numpy as np, torch
from transformers import AutoTokenizer, AutoModel

from corpus import PMIDS as pmids

vecs = torch.load("corpus_embeddings.pt").float().cpu().numpy()   # (n_docs, 768)
# The corpus_embeddings holds only numbers, with row 0, row 1, and so on, and no record of which paper each row belongs to. FAISS returns row numbers, so you need that list to turn row number to pmid.

assert len(pmids) == len(vecs)

index = faiss.IndexFlatIP(vecs.shape[1])   # exact inner-product search, fine at this size
index.add(vecs)

q_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
q_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").eval()

def search(question, k=10):
    with torch.no_grad():
        enc = q_tok([question], truncation=True, padding=True, max_length=64, return_tensors="pt")
        q = q_model(**enc).last_hidden_state[:, 0, :].numpy()
    scores, idx = index.search(q, k)
    return [(int(pmids[i]), float(s)) for i, s in zip(idx[0], scores[0])]