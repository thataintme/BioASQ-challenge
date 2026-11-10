"""Precompute real retrieval + generation results for a handful of curated example
questions, so the showcase page (index.html) can show instant results for these
without hitting the live backend - only free-text search hits it live.

Run from the repo root:
    python web/precompute_examples.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hybrid_search
import reranker
import generate_response
from corpus import CORPUS

GENERATION_MODEL = "gemini-3.5-flash-lite"  # same model the live server uses publicly

EXAMPLES = [
    {"id": "51588c34d24251bc05000095", "qtype": "yesno",
     "question": "Is TREM2 associated with Alzheimer's disease in humans?"},
    {"id": "552421082c8b63434a000005", "qtype": "factoid",
     "question": "What is the enzymatic activity of the breast cancer associated gene BRCA1?"},
    {"id": "56c1f007ef6e39474100003b", "qtype": "list",
     "question": "Tumor-treating fields are effective for treatment of which cancers?"},
    {"id": "550324d7e9bde6963400002f", "qtype": "summary",
     "question": "What is the association between personality trait of neuroticism and risk for Alzheimer's disease?"},
]

POOL = 100
TOP_K = 10


def pubmed_url(pmid):
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def retrieve(question):
    candidates = [pmid for pmid, _ in hybrid_search.search(question, k=POOL)]
    top = reranker.rerank(question, candidates, k=TOP_K)
    docs = CORPUS.loc[[str(p) for p in top]].reset_index().to_dict("records")
    return [
        {"pmid": d["pmid"], "title": d["title"], "abstract": d["abstract"], "url": pubmed_url(d["pmid"])}
        for d in docs
    ]


def main():
    out = []
    for ex in EXAMPLES:
        print("retrieving:", ex["question"])
        docs = retrieve(ex["question"])
        print("generating:", ex["question"])
        result = generate_response.answer(
            ex["question"], qtype=ex["qtype"],
            docs=[{"pmid": d["pmid"], "title": d["title"], "abstract": d["abstract"]} for d in docs],
            model=GENERATION_MODEL,
        )
        out.append({
            "id": ex["id"],
            "qtype": ex["qtype"],
            "question": ex["question"],
            "docs": docs,
            "exact_answer": result["exact_answer"],
            "ideal_answer": result["ideal_answer"],
        })
        print("  ->", len(docs), "docs,", "answer:", (result["ideal_answer"] or "")[:80])

    out_path = Path(__file__).resolve().parent / "examples.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
