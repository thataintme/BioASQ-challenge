"""Prompts for the judge LLM

Three judge calls, each returning a JSON:
  1. extract_claims   -> split an answer into atomic claims + their cited PMIDs
  2. check_support    -> is one claim supported by the given abstracts?
  3. grade_summary    -> does the answer cover the BioASQ ideal answer?

Provider-agnostic: send SYSTEM + the filled USER template to your judge model
with temperature 0 and JSON output mode, then json.loads() the reply.
Use a different model from your generator (Gemini) to avoid self-preference bias.
"""

import json
import re

# ---------------------------------------------------------------- 1. claims

EXTRACT_CLAIMS_SYSTEM = """You split biomedical answers into atomic factual claims.
An atomic claim states exactly one fact and can be checked on its own.
Rewrite pronouns so each claim stands alone ("It causes..." -> "Cushing disease causes...").
Ignore hedges, transitions, and statements about the sources themselves.
Return only JSON matching the schema."""

EXTRACT_CLAIMS_USER = """Answer:
{answer}

Return JSON: {{"claims": [{{"claim": "...", "cited_pmids": ["..."]}}]}}
cited_pmids are the PMIDs in square brackets attached to the sentence the claim came from;
use [] if that sentence has no citation."""

EXTRACT_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "cited_pmids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "cited_pmids"],
            },
        }
    },
    "required": ["claims"],
}

# ---------------------------------------------------------------- 2. support

SUPPORT_SYSTEM = """You check whether a claim is supported by the provided PubMed abstracts.
Use ONLY the abstracts. Do not use your own medical knowledge: a claim that is true
but not stated or directly implied by the abstracts is NOT supported.
- "yes": an abstract states it or directly implies it.
- "partial": part of the claim is supported, part is not (e.g. a number or population differs).
- "no": not supported, or contradicted.
Write your reason before the verdict. Return only JSON matching the schema."""

SUPPORT_USER = """Abstracts:
{abstracts}

Claim: {claim}

Return JSON: {{"reason": "...", "evidence_pmid": "<pmid or null>", "supported": "yes|partial|no"}}"""

SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "evidence_pmid": {"type": ["string", "null"]},
        "supported": {"type": "string", "enum": ["yes", "partial", "no"]},
    },
    "required": ["reason", "evidence_pmid", "supported"],
}

# ---------------------------------------------------------------- 3. correctness

SUMMARY_SYSTEM = """You grade a biomedical answer against an expert reference answer.
Judge medical content only, not style or length.
Score 1-5:
5 = all key facts of the reference present, nothing contradicts it
4 = most key facts present, nothing contradicts it
3 = some key facts present, or a minor inaccuracy
2 = few key facts present, or a significant inaccuracy
1 = wrong, or contradicts the reference
Write your reason before the score. Return only JSON matching the schema."""

SUMMARY_USER = """Question: {question}

Reference answer:
{ideal_answer}

Answer to grade:
{answer}

Return JSON: {{"reason": "...", "missing_facts": ["..."], "contradictions": ["..."], "score": 1}}"""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "missing_facts": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["reason", "missing_facts", "contradictions", "score"],
}

# ---------------------------------------------------------------- helpers


def format_abstracts(docs):
    """docs: iterable of dicts with pmid, title, abstract."""
    return "\n\n".join(f"[PMID {d['pmid']}] {d['title']}\n{d['abstract']}" for d in docs)


def cited_pmids(text):
    """Pull [12345678] style citations out of an answer."""
    return re.findall(r"\[(\d+)\]", text)


def parse_json(reply_text):
    """Tolerate a ```json fence around the reply."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply_text.strip())
    return json.loads(cleaned)


def faithfulness(verdicts):
    """verdicts: list of 'yes'/'partial'/'no' from SUPPORT checks against ALL retrieved abstracts.
    partial counts as half."""
    if not verdicts:
        return None
    score = {"yes": 1.0, "partial": 0.5, "no": 0.0}
    return sum(score[v] for v in verdicts) / len(verdicts)


def agreement(human, judge):
    """Percent agreement and Cohen's kappa between your grades and the judge's."""
    from sklearn.metrics import cohen_kappa_score  # pip install scikit-learn

    pct = sum(h == j for h, j in zip(human, judge)) / len(human)
    return pct, cohen_kappa_score(human, judge)


# How to compute citation accuracy:
#   for each claim with cited_pmids, run SUPPORT against ONLY the cited abstracts.
#   citation_precision = supported (yes) / claims-with-citations
#   uncited_rate       = claims with cited_pmids == [] / all claims
