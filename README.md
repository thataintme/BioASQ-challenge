# BioASQ Biomedical Literature RAG

Retrieval pipeline for the [BioASQ](http://bioasq.org/) challenge (task 14b): sparse (BM25) +
dense (MedCPT) retrieval fused with reciprocal rank fusion, reranked with a MedCPT cross-encoder,
with an optional Gemini-based answer generation step on top.

## Results (5729 BioASQ training questions, recall@10 / MRR@10)

| Method                          | recall@10 | MRR@10 |
|----------------------------------|-----------|--------|
| BM25 (sparse)                    | 0.655     | 0.859  |
| Dense (MedCPT)                   | 0.607     | 0.790  |
| Hybrid (BM25 + dense, RRF)        | 0.683     | 0.869  |
| **Hybrid + cross-encoder rerank** | **0.699** | **0.873** |

Full per-question numbers are in `search_evals/`.

## Pipeline

```
corpus_builder.py                      -> corpus.jsonl            (fetch abstracts from PubMed)
notebooks/..._corpus_encoding.ipynb    -> corpus_embeddings.pt    (MedCPT article encoder, run on GPU)

bm25_search.py      search(question, k)  sparse retrieval
dense_search.py     search(question, k)  dense retrieval (MedCPT query encoder + FAISS)
hybrid_search.py     search(question, k)  reciprocal rank fusion of the two above

bi-encoder_retrievals_k100.py          -> retrievals/hybrid_search_retrievals_k100.csv
reranker.py / notebooks/..._cross_encoder_reranking.ipynb
                                        -> retrievals/cross-encoder_reranked_top10.csv

search_evals.py        -> search_evals/summary.csv, per_question_all.csv   (bm25 / dense / hybrid @10)
cross-encoder_eval.py  -> search_evals/cross-encoder-reranked.csv          (final reranked @10)

generate_response.py  answer(question, qtype, docs=...)  Gemini generation with citations, given docs
app.py                 Streamlit UI: search -> show reranked results -> optional "generate answer"
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
# rerank retrievals/hybrid_search_retrievals_k100.csv -> retrievals/cross-encoder_reranked_top10.csv
#   (notebooks/..._cross_encoder_reranking.ipynb, or reranker.py's __main__ on a candidates.jsonl)
python cross-encoder_eval.py
```

## future_work/

LLM-as-judge faithfulness checking and the manual human-evaluation harness. Parked indefinitely —
human evaluation of generated answers was taking too long to be practical, so this was never
finished or wired into the main pipeline. Kept for reference; expect stale paths if you try to run it.
