"""Generation step: retrieve abstracts, build the prompt, ask Gemini for a cited answer.

Usage:
    from generate_response import answer
    result = answer("What is the difference between Cushing syndrome and Cushing disease?",
                    qtype="summary")
    print(result["ideal_answer"])
"""

import argparse
import json
import os
import re

import pandas as pd
from dotenv import load_dotenv
from google import genai

import hybrid_search
import reranker

load_dotenv()

MODEL = "gemini-3.6-flash"
K = 10
POOL = 100  # hybrid search pool size handed to the cross-encoder


def _hybrid_rerank_search(question, k=K):
    """Default retriever: hybrid search for a generous candidate pool, then cross-encoder
    rerank down to k. Matches the pipeline bm25/dense/hybrid + reranker.py were evaluated with."""
    candidates = [pmid for pmid, _ in hybrid_search.search(question, k=POOL)]
    return [(pmid, None) for pmid in reranker.rerank(question, candidates, k=k)]


SEARCH_FN = _hybrid_rerank_search

# Loaded once at import, not once per question
CORPUS = pd.read_json("./corpus.jsonl", lines=True, dtype={"pmid": str}).set_index("pmid")
CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ---------------------------------------------------------------- prompts

SYSTEM_STRICT = """You are a biomedical question-answering assistant.
Answer using ONLY the provided PubMed abstracts. Do not use outside knowledge.
After every sentence, cite the PMIDs that support it in square brackets, e.g. [36036941] or [36036941, 18226734].
If the abstracts do not contain the answer, say you do not know and that the answer was not found in the corpus.
Provide your answer as a JSON with two fields: {\"ideal_answer\": \"...\", \"exact_answer\": ...}

Every question has ideal_answer, which is a natural human-readable, conversational answer.\n

For a question that can be answered with "yes" or "no" (e.g., 'Are painkillers safe during pregnancy?'), the exact_answer field MUST contain "yes" or "no" accordingly. Further explanations may be given in the ideal_answer field
If a question expects a list of items (e.g., "what are the cells known to be present in blood?" ,"What are the symptoms of  covid-19?"), exact_answer field should contain a list of items only (in json format).
Answers that need a summary (not yes/no, and not a list) do not require a value for the exact_answer field.
"""

# For the judge-validation experiment only: lets the model answer from memory,
# so its claims are NOT supported by the given abstracts (known negatives).
SYSTEM_UNGROUNDED = """You are a biomedical question-answering assistant.
After every sentence, cite PMIDs from the provided abstracts in square brackets.
Provide your answer as a JSON with two fields: {\"ideal_answer\": \"...\", \"exact_answer\": ...}
Every question has ideal_answer, which is a natural and conversational answer.\n
For a question that can be answered with "yes" or "no" (e.g., 'Are painkillers safe during pregnancy?'), the exact_answer field MUST contain "yes" or "no" accordingly. Further explanations may be given in the ideal_answer field
If a question expects a list of items (such as symptoms, causes, etc.), exact_answer field should contain a list of items only.
Answers that need a summary (not yes/no, and not a list) do not require a value for the exact_answer field.
"""

EXACT_ANSWER_RULES = {
    "yesno": '"yes" or "no"',
    "factoid": "a list of up to 5 candidate entity names, best first",
    "list": "a list of entity names",
    "summary": "null",
}

USER_TEMPLATE = """Question: {question}

Abstracts:
{context}

Return JSON: {{"exact_answer": <{exact_rule}>, "ideal_answer": "<a short paragraph with citations>"}}"""

# ---------------------------------------------------------------- helpers


def retrieve(question, k=K):
    """Top-k docs from the corpus, in search-rank order."""
    pmids = [str(pmid) for pmid, _ in SEARCH_FN(question, k=k)]
    return CORPUS.loc[pmids].reset_index().to_dict("records")


def format_context(docs):
    return "\n\n".join(f"[PMID {d['pmid']}] {d['title']}\n{d['abstract']}" for d in docs)


def parse_json(text):
    """Tolerate a ```json fence around the reply."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(cleaned)


def call_llm(system, prompt, stream_to_console=False, model=MODEL):
    stream = CLIENT.interactions.create(
        model=model,
        system_instruction=system,
        input=prompt,
        stream=True,
    )
    chunks = []
    for event in stream:
        if event.event_type == "step.delta" and event.delta.type == "text":
            chunks.append(event.delta.text)
            if stream_to_console:
                print(event.delta.text, end="", flush=True)
    return "".join(chunks)


# ---------------------------------------------------------------- main entry point


def answer(question, qtype="summary", docs=None, grounded=True, stream_to_console=False, model=MODEL, return_docs=False):
    """Retrieve (unless docs are given), generate, and return everything the evals need.

    docs: pass another question's docs here to build sabotaged answers for judge validation.
    grounded: False uses SYSTEM_UNGROUNDED (judge-validation negatives only).
    """
    if docs is None:
        docs = retrieve(question)
    prompt = USER_TEMPLATE.format(
        question=question,
        context=format_context(docs),
        exact_rule=EXACT_ANSWER_RULES[qtype],
    )
    raw = call_llm(SYSTEM_STRICT if grounded else SYSTEM_UNGROUNDED, prompt, stream_to_console, model)

    try:
        parsed = parse_json(raw)
    except json.JSONDecodeError:
        parsed = {"exact_answer": None, "ideal_answer": None}

    response = {
        "question": question,
        "qtype": qtype,
        "retrieved_pmids": [d["pmid"] for d in docs],
        "exact_answer": parsed.get("exact_answer"),
        "ideal_answer": parsed.get("ideal_answer"),
        "raw": raw,  # keep it: useful when parsing fails
    }

    return (docs, response) if return_docs else response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, help=f"Gemini model to use (default: {MODEL})")
    args = parser.parse_args()

    result = answer(
        "What is the difference between Cushing Syndrome and Cushing Disease?",
        qtype="summary",
        stream_to_console=True,
        model=args.model,
    )
    print("\n\nParsed:", json.dumps({k: result[k] for k in ("exact_answer", "ideal_answer")}, indent=2))
