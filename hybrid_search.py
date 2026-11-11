# hybrid_search.py
import bm25_search, dense_search

def fuse(bm25_results, dense_results, c=60):
    """Reciprocal rank fusion of two ranked (pmid, score) lists."""
    scores = {}
    for results in (bm25_results, dense_results):
        for rank, (pmid, _) in enumerate(results, start=1):
            scores[pmid] = scores.get(pmid, 0) + 1 / (c + rank)
    return sorted(scores.items(), key=lambda x: -x[1])

def search(question, k=10, pool=100, c=60):
    return fuse(bm25_search.search(question, k=pool), dense_search.search(question, k=pool), c=c)[:k]