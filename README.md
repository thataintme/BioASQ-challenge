# BioASQ Biomedical Literature RAG

Retrieval pipeline for the [BioASQ](http://bioasq.org/) challenge (task 14b) over ~58k PubMed
abstracts: sparse (BM25) + dense (MedCPT) retrieval fused with reciprocal rank fusion, reranked
with a MedCPT cross-encoder, with an optional Gemini generation step on top that answers from
the retrieved abstracts and cites them by PMID.

## Results (5729 BioASQ training questions, recall@10 / MRR@10)

| Method                            | recall@10 | MRR@10 |
|------------------------------------|-----------|--------|
| BM25 (sparse)                      | 0.655     | 0.859  |
| Dense (MedCPT)                     | 0.607     | 0.790  |
| Hybrid (BM25 + dense, RRF)          | 0.683     | 0.869  |
| + cross-encoder rerank, max_length=512 | 0.699 | 0.873  |
| + cross-encoder rerank, max_length=256 (**default**) | 0.691 | 0.870 |
| 🔴 + cross-encoder rerank, max_length=128 (regresses below hybrid) | 0.655 | 0.860 |

Reranking's real effect is on recall@10 (pulling more relevant docs into the top 10). The
hybrid stage alone already nearly saturates MRR@10 (whether the *single best* result lands
first), so there's little room left for reranking to move that number.

`max_length` is how many tokens of each title+abstract the cross-encoder sees per candidate;
512 is the model's full length. Truncating to 256 halves the input per candidate at a cost of
~1% relative recall@10. Truncating further, to 128, erases the reranker's entire gain and makes
it net-negative versus not reranking at all. Full per-question numbers are in `search_evals/`.

## Latency (n=100)

Collected with the retrieval system deployed on Hugging Face Spaces on ZeroGPU (a slice of an
NVIDIA RTX PRO 6000 Blackwell GPU), cross-encoder `max_length=256`, 100 BioASQ training
questions sampled with a fixed seed, one warm-up query dropped. Every stage of every query comes
from the same run (`hf_space/latency_bench.py`).

| stage                              | p50 ms | p95 ms | mean ms |
|-------------------------------------|-------:|-------:|--------:|
| BM25                                |    1.8 |    5.0 |     2.5 |
| Dense (MedCPT query encoder + FAISS) |    9.2 |   16.1 |    18.1 |
| RRF fusion                          |    0.1 |    0.2 |     0.1 |
| retrieve (BM25 + dense + fusion)    |   11.6 |   20.9 |    20.8 |
| cross-encoder rerank (100 -> 10)    |    261 |    273 |     263 |
| time to first token                 |  1,424 |  3,545 |   1,810 |
| generation (full response)          |  1,957 |  4,335 |   2,657 |
| end-to-end (retrieve + rerank + generation) | 2,260 | 4,692 | 2,941 |

- **Generation dominates end-to-end time** (~90% of the mean); reranking is ~9% and retrieval
  under 1%.
- **Means above medians come from a few outliers.** Three queries ran dense encoding and
  reranking at ~290 ms and ~350 ms instead of ~9 ms and ~260 ms (still on the GPU), and one
  generation call took 27.7 s. The p95 numbers are unaffected by the single slow generation.
- **Generation used `gemini-3.5-flash-lite`**, not `generate_response.MODEL`
  (`gemini-3.6-flash`): 3.6-flash's free tier is capped at 20 requests/day (confirmed), too few
  for n=100 in one run. 0 retries and 0 rate-limit errors across all 100 queries. Average
  ~3,570 input / ~140 output tokens per query; free-tier cost is $0, paid-tier equivalent
  ~$0.0014/query (flash-lite pricing as of 2026-11-09, third-party sourced, unverified).
- Stage times are measured on the serving machine, so they exclude network time to the client.

## Pipeline

```
corpus.py                               single shared load of corpus.jsonl (bm25_search, dense_search,
                                         reranker, and generate_response all import this instead of
                                         each loading their own copy)

corpus_builder.py                      -> corpus.jsonl            (fetch abstracts from PubMed)
notebooks/..._corpus_encoding.ipynb    -> corpus_embeddings.pt    (MedCPT article encoder, run on GPU)

bm25_search.py      search(question, k)  sparse retrieval
dense_search.py     search(question, k)  dense retrieval (MedCPT query encoder + FAISS)
hybrid_search.py    search(question, k)  reciprocal rank fusion of the two above
reranker.py         rerank(question, candidate_pmids, k)  MedCPT cross-encoder

bi-encoder_retrievals_k100.py          -> retrievals/hybrid_search_retrievals_k100.csv
reranker.py / notebooks/..._cross_encoder_reranking.ipynb
                                        -> retrievals/reranked_top10_ml{512,256,128}.csv

search_evals.py        -> search_evals/summary.csv, per_question_all.csv   (bm25 / dense / hybrid @10)
cross-encoder_eval.py  -> search_evals/cross-encoder-reranked_ml{512,256,128}.csv  (reranked @10, per max_length)

generate_response.py  answer(question, qtype, docs=...)  Gemini generation with citations, given docs
app.py                 Streamlit UI: search -> show reranked results -> optional "generate answer"
```

`generate_response.answer()` can retrieve on its own (hybrid search + rerank by default) or take
an already-retrieved `docs` list, so the answer is grounded in exactly the documents shown.
`qtype` is the BioASQ question type (`yesno`, `factoid`, `list`, `summary`), which sets the
shape of the short `exact_answer`; `auto` lets the model infer it from the question when the
type isn't known.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY
```

The BioASQ training data isn't redistributed here — register and download it from
http://participants-area.bioasq.org/ and place it at:

```
Dataset/BioASQ-training14b/training14b.json
```

Then rebuild the corpus and embeddings (both gitignored, ~90MB + ~170MB):

```bash
python corpus_builder.py                                   # -> corpus.jsonl
# run notebooks/BioASQ_challenge_corpus_encoding.ipynb on a GPU (e.g. Colab) -> corpus_embeddings.pt
```

## Running

```bash
streamlit run app.py
```

Or use the pipeline directly:

```python
import hybrid_search, reranker
candidates = [pmid for pmid, _ in hybrid_search.search("your question", k=100)]
top10 = reranker.rerank("your question", candidates, k=10)
```

To reproduce the retrieval evals:

```bash
python search_evals.py          # bm25 / dense / hybrid @10
python bi-encoder_retrievals_k100.py
# rerank retrievals/hybrid_search_retrievals_k100.csv -> retrievals/reranked_top10_ml{512,256,128}.csv
#   (notebooks/..._cross_encoder_reranking.ipynb on a GPU)
python cross-encoder_eval.py --input retrievals/reranked_top10_ml256.csv \
    --output search_evals/cross-encoder-reranked_ml256.csv --label "Cross-Encoder ml=256"
```

## future_work/

LLM-as-judge faithfulness checking and the manual human-evaluation harness. Parked indefinitely —
human evaluation of generated answers was taking too long to be practical, so this was never
finished or wired into the main pipeline. Kept for reference; expect stale paths if you try to run it.
