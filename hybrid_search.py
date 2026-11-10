# hybrid_search.py
import bm25_search, dense_search

def search(question, k=10, pool=100, c=60):
    scores = {}
    for results in (bm25_search.search(question, k=pool), dense_search.search(question, k=pool)):
        for rank, (pmid, _) in enumerate(results, start=1):
            scores[pmid] = scores.get(pmid, 0) + 1 / (c + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:k]