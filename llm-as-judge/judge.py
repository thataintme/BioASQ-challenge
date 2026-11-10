"""Judge LLM: runs the prompts in judge_prompts.py and checks agreement with your hand grades.

Judge model is Claude (a different model family from the Gemini generator,
to avoid self-preference bias).

Setup:
    pip install anthropic scikit-learn
    add ANTHROPIC_API_KEY=... to your .env

Hand-grade CSV (hand_grades.csv), one row per RESPONSE, needs these columns:
    answer         the generated ideal_answer text
    context_pmids  PMIDs that were in that answer's prompt, comma-separated
    human          your grade for the whole response:
                   yes = every claim supported, partial = some unsupported, no = mostly unsupported

Run:
    python judge.py hand_grades.csv
"""



import copy
import json
import sys

import anthropic
import pandas as pd
from dotenv import load_dotenv

from judge_llm_prompts import (
    EXTRACT_SCHEMA, EXTRACT_SYSTEM, EXTRACT_USER,
    SUMMARY_SCHEMA, SUMMARY_SYSTEM, SUMMARY_USER,
    SUPPORT_SCHEMA, SUPPORT_SYSTEM, SUPPORT_USER,
    agreement, format_abstracts,
)

load_dotenv()

JUDGE_MODEL = "claude-haiku-5-5"  # cheap; try "claude-sonnet-5-5" if agreement is low
CLIENT = anthropic.Anthropic()    # reads ANTHROPIC_API_KEY
CORPUS = pd.read_json("./corpus.jsonl", lines=True, dtype={"pmid": str}).set_index("pmid")


def strict(schema):
    """Structured outputs need additionalProperties: false on every object,
    and plain types, so adjust the schemas from judge_prompts.py."""
    s = copy.deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            if node.get("type") == ["string", "null"]:
                node["type"] = "string"  # judge writes "none" instead of null
            if node.get("type") == "integer" and "minimum" in node:
                node["enum"] = list(range(node.pop("minimum"), node.pop("maximum") + 1))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(s)
    return s


def ask_judge(system, user, schema):
    resp = CLIENT.messages.create(
        model=JUDGE_MODEL,
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": strict(schema)}},
    )
    if resp.stop_reason == "refusal":
        return None
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


# ---------------------------------------------------------------- the three checks


def extract_claims(answer_text):
    out = ask_judge(EXTRACT_SYSTEM, EXTRACT_USER.format(answer=answer_text), EXTRACT_SCHEMA)
    return out["claims"] if out else []


def check_support(claim, pmids):
    docs = CORPUS.loc[[str(p) for p in pmids]].reset_index().to_dict("records")
    return ask_judge(
        SUPPORT_SYSTEM,
        SUPPORT_USER.format(abstracts=format_abstracts(docs), claim=claim),
        SUPPORT_SCHEMA,
    )


def grade_summary(question, ideal_answer, answer_text):
    return ask_judge(
        SUMMARY_SYSTEM,
        SUMMARY_USER.format(question=question, ideal_answer=ideal_answer, answer=answer_text),
        SUMMARY_SCHEMA,
    )


# ---------------------------------------------------------------- validation run


def response_verdict(answer_text, pmids):
    """Judge a whole response: extract claims, check each, roll up to yes/partial/no.
    Rule: all claims supported -> yes; half or more unsupported -> no; otherwise partial.
    Change this rule if it doesn't match how you graded."""
    claims = extract_claims(answer_text)
    checks = [check_support(c["claim"], pmids) for c in claims]
    grades = [c["supported"] if c else "no" for c in checks]
    if not grades:
        return "no", 0.0, []
    score = sum({"yes": 1.0, "partial": 0.5, "no": 0.0}[g] for g in grades) / len(grades)
    unsupported = [c["claim"] for c, g in zip(claims, grades) if g != "yes"]
    if score == 1.0:
        verdict = "yes"
    elif grades.count("no") >= len(grades) / 2:
        verdict = "no"
    else:
        verdict = "partial"
    return verdict, score, unsupported


def validate(hand_grades_csv):
    df = pd.read_csv(hand_grades_csv, dtype=str)
    verdicts, scores, flagged = [], [], []
    for row in df.itertuples():
        pmids = [p.strip() for p in row.context_pmids.split(",") if p.strip()]
        verdict, score, unsupported = response_verdict(row.answer, pmids)
        verdicts.append(verdict)
        scores.append(round(score, 2))
        flagged.append(" | ".join(unsupported))
        print(f"{row.Index + 1}/{len(df)}  you: {row.human:<8} judge: {verdict:<8} faithfulness={score:.2f}")

    df["judge"] = verdicts
    df["judge_faithfulness"] = scores
    df["judge_unsupported_claims"] = flagged
    out = hand_grades_csv.replace(".csv", "_with_judge.csv")
    df.to_csv(out, index=False)

    human = df["human"].str.strip().str.lower().tolist()
    pct, kappa = agreement(human, verdicts)
    print(f"\nAgreement: {pct:.0%}   Cohen's kappa: {kappa:.2f}   (n={len(df)})")
    print("Disagreements (read the unsupported claims the judge flagged):")
    mask = df["human"].str.strip().str.lower() != df["judge"]
    print(df[mask][["human", "judge", "judge_unsupported_claims"]].to_string(max_colwidth=100))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    validate(sys.argv[1] if len(sys.argv) > 1 else "hand_grades.csv")
