# BioASQ Biomedical Literature RAG

Retrieval pipeline for the [BioASQ](http://bioasq.org/) challenge (task 14b): sparse (BM25) +
dense (MedCPT) retrieval fused with reciprocal rank fusion, reranked with a MedCPT cross-encoder,
with an optional Gemini-based answer generation step on top.

## Results (5729 BioASQ training questions, recall@10 / MRR@10)

| Method                            | recall@10 | MRR@10 |
|------------------------------------|-----------|--------|
| BM25 (sparse)                      | 0.655     | 0.859  |
| Dense (MedCPT)                     | 0.607     | 0.790  |
| Hybrid (BM25 + dense, RRF)          | 0.683     | 0.869  |
| + cross-encoder rerank, max_length=512 | 0.699 | 0.873  |
| + cross-encoder rerank, max_length=256 (**deployed default**) | 0.691 | 0.870 |
| 🔴 + cross-encoder rerank, max_length=128 (regresses below hybrid) | 0.655 | 0.860 |

Reranking's real effect is on recall@10 (pulling more relevant docs into the top 10) — the
hybrid stage alone already nearly saturates MRR@10 (whether the *single best* result lands
first), so there's little room left for reranking to move that number. `max_length` is how
many tokens of each title+abstract the cross-encoder sees per candidate; 512 is the model's
full length. Truncating to 256 trades ~1% relative recall@10 for ~2x less CPU time (see
"Deploying on low-power hardware" below) — truncating further, to 128, erases the reranker's
entire gain and makes it net-negative versus not reranking at all. Full per-question numbers
and the three-way comparison are in `search_evals/`.

## Latency (retrieval only, n=20, this laptop: 2019 quad-core CPU, no GPU, max_length=256)

| stage                          | p50 ms | p95 ms | mean ms |
|---------------------------------|-------:|-------:|--------:|
| BM25                            |      3 |     11 |       4 |
| Dense (MedCPT query encoder)    |    154 |    219 |     162 |
| RRF fusion                      |      0 |      0 |       0 |
| retrieve (bm25 + dense + fusion)|    158 |    224 |     166 |
| cross-encoder rerank            | 58,534 | 60,968 |  58,107 |
| end-to-end (retrieve + rerank)  | 58,683 | 61,178 |  58,273 |

Retrieval itself is near-instant — p95 under a quarter of a second. The cross-encoder rerank
step is essentially the entire latency budget (>99%), which is exactly the cost the
`max_length` tradeoff above is managing. See `retrieval_latency_bench.py`.

## Latency with generation (n=100, end to end)

| stage                              | p50 ms | p95 ms | mean ms |
|-------------------------------------|-------:|-------:|--------:|
| retrieve (bm25 + dense + fusion)    |    114 |    157 |     122 |
| cross-encoder rerank                | 53,388 | 56,295 |  53,596 |
| time to first token                 |  3,768 |  9,616 |   4,743 |
| generation (full response)          |  5,200 | 11,445 |   6,404 |
| end-to-end                          | 58,853 | 66,090 |  60,123 |

Measured end to end in one benchmark pass per query (`latency_bench.py`), not by separately
measuring generation and adding it to the retrieval numbers above - that would miss any
interaction between the stages. 0 retries and 0 rate-limit errors across all 100 queries.
Ran against **`gemini-3.5-flash-lite`**, not the `gemini-3.6-flash` production default -
3.6-flash's free tier caps out at a confirmed 20 requests/day (see commit history: hit this
directly while testing), which can't support a statistically credible n in one sitting;
flash-lite's free tier is reported (third-party, unverified) at ~500 requests/day. Average
~3,500 input / ~140 output tokens per query; free tier cost is $0, paid-tier equivalent
~$0.0014/query (~$1.41 per 1k queries, flash-lite pricing as of 2026-11-09, unverified,
third-party sourced). Generation (ttft+gen) is real but small next to reranking - cross-encoder
rerank is still >90% of total latency.

## Pipeline

```
corpus.py                               single shared load of corpus.jsonl (bm25_search, dense_search,
                                         reranker, and generate_response all import this instead of
                                         each loading their own copy - matters on constrained hardware)

corpus_builder.py                      -> corpus.jsonl            (fetch abstracts from PubMed)
notebooks/..._corpus_encoding.ipynb    -> corpus_embeddings.pt    (MedCPT article encoder, run on GPU)

bm25_search.py      search(question, k)  sparse retrieval
dense_search.py     search(question, k)  dense retrieval (MedCPT query encoder + FAISS)
hybrid_search.py     search(question, k)  reciprocal rank fusion of the two above

bi-encoder_retrievals_k100.py          -> retrievals/hybrid_search_retrievals_k100.csv
reranker.py / notebooks/..._cross_encoder_reranking.ipynb
                                        -> retrievals/reranked_top10_ml{512,256,128}.csv

search_evals.py        -> search_evals/summary.csv, per_question_all.csv   (bm25 / dense / hybrid @10)
cross-encoder_eval.py  -> search_evals/cross-encoder-reranked_ml{512,256,128}.csv  (reranked @10, per max_length)

generate_response.py  answer(question, qtype, docs=...)  Gemini generation with citations, given docs
app.py                 Streamlit UI: search -> show reranked results -> optional "generate answer"
latency_bench.py        p50/p95 latency + free-tier cost benchmark, stage by stage
```

`generate_response.answer()` can retrieve on its own (hybrid search + rerank by default) or take
an already-retrieved `docs` list, which is what `app.py` uses so the generated answer is grounded
in exactly what's shown on screen.

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
#   (notebooks/..._cross_encoder_reranking.ipynb on a GPU - a full 5729-question CPU rerank
#   is impractically slow, see "Deploying on low-power hardware" below)
python cross-encoder_eval.py --input retrievals/reranked_top10_ml256.csv \
    --output search_evals/cross-encoder-reranked_ml256.csv --label "Cross-Encoder ml=256"
```

To benchmark retrieval latency (BM25 / dense / RRF fusion / cross-encoder rerank — no Gemini
call, so no rate limits and no throttling to wait through):

```bash
python retrieval_latency_bench.py Dataset/BioASQ-training14b/training14b.json --n 20
```

To benchmark latency with generation included (results above): `latency_bench.py` does the
same thing end to end, adding the Gemini generation step (ttft/gen_ms) and free-tier cost
accounting in the same per-query run. The production default model (`generate_response.MODEL`)
may not have enough free-tier daily quota for a credible sample size — use `--model` (and
`--rpm`/`--rpd`/`--price-in`/`--price-out`/`--price-date`) to point at a different one for the
benchmark run without changing the production default:

```bash
python latency_bench.py Dataset/BioASQ-training14b/training14b.json --n 100 \
    --model gemini-3.5-flash-lite --rpm 10 --rpd 400 --price-in 0.30 --price-out 2.50
```

Verify whatever `FREE_RPM`/`FREE_RPD`/pricing you use against your own AI Studio dashboard and
ai.google.dev/pricing first — Google doesn't publish a reliable per-model table, and the
defaults baked into the script may be stale by the time you read this.

## Deploying on low-power hardware

This runs fine CPU-only on something like an old Surface Pro 3 (8GB RAM) acting as a mini
server for single-user, non-concurrent use — but a few things matter there:

- **Install the CPU-only torch wheel**, not the default one. Plain `pip install torch` often
  pulls in CUDA binaries you don't need and don't want to download on a GPU-less machine:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- **Expect slow, not fast.** Two BERT-base-sized models (MedCPT query encoder + MedCPT
  cross-encoder) running on CPU means cold start (loading both models + building the
  BM25/FAISS indexes) can take 1-3 minutes. Cross-encoder reranking is the dominant cost in
  the whole pipeline: it pads every batch to its *longest* document, and a single long
  abstract can force a full chunk up to the model's max sequence length. Measured on a
  2019 quad-core laptop CPU (no GPU), a chunk of 25 candidates at full length took ~30s;
  that's exactly why the default `max_length` is 256, not 512 (see Results above) — on an
  actual 2014-era dual-core CPU like a Surface Pro 3, expect this to be meaningfully slower
  still.
- **Memory is already deduplicated** — `corpus.py` loads `corpus.jsonl` once and is shared by
  `bm25_search`, `dense_search`, `reranker`, and `generate_response`, instead of each module
  keeping its own copy. Without that, 4 separate copies of the corpus plus both models would be
  tight on 8GB.
- **Watch for thermal throttling** on fanless variants (base i3 Surface Pro 3). Single queries
  are short bursts so this shouldn't matter in normal use, but if it feels like it's slowing down
  the longer it runs, that's likely why.

## future_work/

LLM-as-judge faithfulness checking and the manual human-evaluation harness. Parked indefinitely —
human evaluation of generated answers was taking too long to be practical, so this was never
finished or wired into the main pipeline. Kept for reference; expect stale paths if you try to run it.
