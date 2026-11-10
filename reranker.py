# rerank() is also imported live by generate_response.py / app.py for single-query reranking
# (fine on CPU for one question at a time). The __main__ block below batch-reranks a whole
# candidates.jsonl file and is what you'd still run on Colab/GPU for a full eval pass.

import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from corpus import CORPUS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder").to(DEVICE).eval()


def rerank(question, candidate_pmids, k=10, chunk=25, max_length=256):
    """max_length=256 (not the model's full 512) trades ~1% relative recall@10 for ~2x
    speed: truncating further, to 128, erases the reranker's gain entirely and goes
    net-negative vs. not reranking at all. See search_evals/cross-encoder-reranked_ml*.csv."""
    docs = CORPUS.loc[[str(p) for p in candidate_pmids]]
    texts = [f"{t} {a}" for t, a in zip(docs["title"], docs["abstract"])]
    scores = []
    with torch.no_grad():
        for i in range(0, len(texts), chunk):
            pairs = [[question, t] for t in texts[i:i + chunk]]
            enc = tok(pairs, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(DEVICE)
            scores.extend(model(**enc).logits.squeeze(dim=1).tolist())
    ranked = sorted(zip(candidate_pmids, scores), key=lambda x: -x[1])
    return [pmid for pmid, _ in ranked[:k]]


if __name__ == "__main__":
    with open("candidates.jsonl") as f, open("reranked.jsonl", "w") as out:
        for n, line in enumerate(f, 1):
            item = json.loads(line)
            top = rerank(item["question"], item["candidates"])
            out.write(json.dumps({"qid": item["qid"], "top": top}) + "\n")
            if n % 100 == 0:
                print(n)
