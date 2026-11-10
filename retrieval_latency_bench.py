"""Retrieval-only latency benchmark: BM25 / dense / RRF fusion / cross-encoder rerank.

No Gemini call here on purpose - generation latency is bound by the Gemini free tier's
rate limits, not by this hardware, so it doesn't belong in a "how fast is my hardware"
story. TODO: benchmark generation latency separately (see latency_bench.py) once free-tier
throttling isn't the dominant cost, or paid billing is set up.

Times each stage separately, per query:
    bm25_ms       sparse retrieval
    dense_ms      dense retrieval (MedCPT query encoder + FAISS)
    fusion_ms     reciprocal rank fusion of the two (expected to be ~0, included to confirm)
    retrieve_ms   bm25_ms + dense_ms + fusion_ms
    rerank_ms     MedCPT cross-encoder -> top 10
    e2e_ms        retrieve_ms + rerank_ms

Run:
    python retrieval_latency_bench.py Dataset/BioASQ-training14b/training14b.json --n 20
"""

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

import bm25_search
import dense_search
import reranker

POOL = 100   # matches generate_response.POOL / hybrid_search's default pool
C = 60       # matches hybrid_search.py's RRF constant
TOP_K = 10


def fuse(bm25_results, dense_results, c=C):
    scores = {}
    for results in (bm25_results, dense_results):
        for rank, (pmid, _) in enumerate(results, start=1):
            scores[pmid] = scores.get(pmid, 0) + 1 / (c + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def run_one(question):
    t0 = time.perf_counter()
    bm25_results = bm25_search.search(question, k=POOL)
    bm25_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dense_results = dense_search.search(question, k=POOL)
    dense_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    fused = fuse(bm25_results, dense_results)
    fusion_ms = (time.perf_counter() - t0) * 1000
    candidates = [pmid for pmid, _ in fused[:POOL]]

    t0 = time.perf_counter()
    rerank(question, candidates)
    rerank_ms = (time.perf_counter() - t0) * 1000

    retrieve_ms = bm25_ms + dense_ms + fusion_ms
    return dict(bm25_ms=bm25_ms, dense_ms=dense_ms, fusion_ms=fusion_ms,
                retrieve_ms=retrieve_ms, rerank_ms=rerank_ms, e2e_ms=retrieve_ms + rerank_ms)


def rerank(question, candidates):
    return reranker.rerank(question, candidates, k=TOP_K)


def pct(series, p):
    s = sorted(series)
    if not s:
        return float("nan")
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="retrieval_latency_runs.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qs = json.loads(Path(args.questions).read_text())
    qs = qs["questions"] if isinstance(qs, dict) else qs
    random.Random(args.seed).shuffle(qs)
    qs = qs[: args.n]

    run_one(qs[0]["body"])  # warm-up, dropped (first-call overhead, not representative)

    rows = []
    for i, q in enumerate(qs, 1):
        rows.append(run_one(q["body"]))
        print(f"{i}/{len(qs)}")
        if i % 5 == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)  # checkpoint

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\nqueries={len(df)}")
    print(f"{'stage':<12}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}")
    for col in ["bm25_ms", "dense_ms", "fusion_ms", "retrieve_ms", "rerank_ms", "e2e_ms"]:
        print(f"{col:<12}{pct(df[col], 50):>10.0f}{pct(df[col], 95):>10.0f}{df[col].mean():>10.0f}")


if __name__ == "__main__":
    main()
