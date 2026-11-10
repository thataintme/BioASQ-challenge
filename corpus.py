"""Single shared load of corpus.jsonl.

bm25_search, dense_search, reranker, and generate_response all need the corpus;
importing it from here instead of each loading their own copy matters on
memory-constrained hardware (saves ~4x the corpus size in RAM).
"""

import pandas as pd

CORPUS = pd.read_json("./corpus.jsonl", lines=True, dtype={"pmid": str}).set_index("pmid")
PMIDS = CORPUS.index.tolist()  # row order matches corpus_embeddings.pt
TEXTS = (CORPUS["title"] + " " + CORPUS["abstract"]).tolist()
