"""Latency + cost benchmark for the full pipeline on the Gemini free tier.

Times each stage separately, per query:
    retrieve_ms   hybrid search -> candidate PMIDs
    rerank_ms     MedCPT cross-encoder -> top 10
    ttft_ms       request sent -> first answer token (what a streaming user feels)
    gen_ms        request sent -> last token
    e2e_ms        retrieve + rerank + gen (excludes any rate-limit waiting)

Free-tier rules this script follows so the numbers stay honest:
  * Client-side throttle to stay under RPM. Time spent waiting for a slot is
    recorded as `throttle_wait_ms` and is NOT part of any latency number.
  * A 429 (rate limit) or 5xx (transient server error - the free tier gets these
    often under "high demand") is retried with backoff; the query's timing comes
    from the attempt that succeeded, and `retries` tells you it happened.
  * One warm-up query is run and dropped (model load, TLS handshake, CUDA init).
  * Stops cleanly when the daily request cap would be exceeded.

Cost: free tier costs $0. The script also logs real token counts and prices them at
the paid-tier rate you set below, so you can report "equivalent paid cost per query".

Run:
    python latency_bench.py Dataset/BioASQ-training14b/training14b.json --n 100 --out latency_runs.csv
    python latency_bench.py Dataset/BioASQ-training14b/training14b.json --n 100 --cache      # repeat with cache on
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import pandas as pd

import generate_response as generate
from reranker import rerank
import hybrid_search

# ---------------------------------------------------------------- config to fill in


def get_candidates(question, k=100):
    """Matches generate_response.POOL: hybrid search for a generous pool, then rerank down."""
    return [str(pmid) for pmid, _ in hybrid_search.search(question, k=k)]


# Free-tier limits for generate.MODEL (gemini-3.6-flash). FREE_RPD=20 is CONFIRMED - hit it
# directly on 2026-11-09, the API's own 429 body says "limit: 20 ... Free Tier" verbatim.
# FREE_RPM=5 is still third-party-reported (Google's docs don't publish a per-model RPM
# table) - verify against your own AI Studio dashboard if that one matters to your report.
FREE_RPM = 5
FREE_RPD = 20

# Paid-tier prices in USD per 1M tokens for generate.MODEL, third-party-reported as of
# 2026-11-09 (introductory price through 2026-12-31, then $1.50 in / $7.50 out).
# VERIFY at ai.google.dev/pricing on the day you run, and write the date in your report.
PRICE_IN_PER_M = 0.75
PRICE_OUT_PER_M = 3.75
PRICES_CHECKED_ON = "2026-11-09 (unverified, third-party sourced)"

TOP_K = 10
MAX_RETRIES = 5

# ---------------------------------------------------------------- throttle


class Throttle:
    """Sliding 60 s window. Returns how long it made you wait (ms)."""

    def __init__(self, rpm, rpd):
        self.rpm, self.rpd = rpm, rpd
        self.sent = []
        self.total = 0

    def wait(self):
        if self.total >= self.rpd:
            raise RuntimeError("daily request cap reached; resume tomorrow")
        t0 = time.perf_counter()
        while True:
            now = time.time()
            self.sent = [t for t in self.sent if now - t < 60]
            if len(self.sent) < self.rpm:
                break
            time.sleep(60 - (now - self.sent[0]) + 0.05)
        self.sent.append(time.time())
        self.total += 1
        return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- answer cache

CACHE = {}


def cache_key(question, pmids):
    norm = " ".join(question.lower().split())
    return hashlib.sha256((norm + "|" + ",".join(pmids) + "|" + generate.MODEL).encode()).hexdigest()


# ---------------------------------------------------------------- timed generation

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _status_of(e):
    """The genai SDK raises different exception classes for the same kind of failure
    depending on which internal code path handled the request (e.g. a 429 ClientError
    vs. a 503 from a different internal error hierarchy that isn't even an APIError
    subclass) - so check attribute names generically instead of catching one class."""
    for attr in ("code", "status_code", "status"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    return None


def _usage_from(event):
    """Pull token counts off a stream event if the SDK exposes them there."""
    for obj in (getattr(event, "usage", None), getattr(getattr(event, "interaction", None), "usage", None)):
        if obj is None:
            continue
        tin = getattr(obj, "total_input_tokens", None) or getattr(obj, "input_tokens", None)
        tout = getattr(obj, "total_output_tokens", None) or getattr(obj, "output_tokens", None)
        if tin or tout:
            return tin, tout
    return None


def timed_generate(system, prompt):
    t0 = time.perf_counter()
    ttft = None
    chunks, usage = [], None
    stream = generate.CLIENT.interactions.create(
        model=generate.MODEL, system_instruction=system, input=prompt, stream=True
    )
    for event in stream:
        if event.event_type == "step.delta" and event.delta.type == "text":
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            chunks.append(event.delta.text)
        usage = _usage_from(event) or usage
    gen_ms = (time.perf_counter() - t0) * 1000
    text = "".join(chunks)

    if usage is None:  # fall back to counting (free, not timed)
        count = lambda s: generate.CLIENT.models.count_tokens(model=generate.MODEL, contents=s).total_tokens
        usage = (count(system + "\n" + prompt), count(text) if text else 0)
    return text, ttft, gen_ms, usage


def run_one(q, throttle, use_cache):
    question, qtype = q["body"], q["type"]
    row = {"qid": q.get("id"), "qtype": qtype}

    t = time.perf_counter()
    cands = get_candidates(question)
    row["retrieve_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    top = rerank(question, cands, k=TOP_K)
    row["rerank_ms"] = (time.perf_counter() - t) * 1000

    key = cache_key(question, top)
    if use_cache and key in CACHE:
        t = time.perf_counter()
        _ = CACHE[key]
        row.update(cache_hit=True, ttft_ms=(time.perf_counter() - t) * 1000, gen_ms=0.0,
                   in_tokens=0, out_tokens=0, retries=0, throttle_wait_ms=0.0)
    else:
        docs = generate.CORPUS.loc[top].reset_index().to_dict("records")
        prompt = generate.USER_TEMPLATE.format(
            question=question, context=generate.format_context(docs),
            exact_rule=generate.EXACT_ANSWER_RULES[qtype],
        )
        retries, waited = 0, 0.0
        while True:
            waited += throttle.wait()
            try:
                text, ttft, gen_ms, (tin, tout) = timed_generate(generate.SYSTEM_STRICT, prompt)
                break
            except Exception as e:
                if _status_of(e) not in RETRYABLE_STATUS or retries >= MAX_RETRIES:
                    raise
                retries += 1
                time.sleep(min(60, 2 ** retries) + random.random())
        CACHE[key] = text
        row.update(cache_hit=False, ttft_ms=ttft, gen_ms=gen_ms, in_tokens=tin,
                   out_tokens=tout, retries=retries, throttle_wait_ms=waited)

    row["e2e_ms"] = row["retrieve_ms"] + row["rerank_ms"] + (row["gen_ms"] or row["ttft_ms"])
    row["paid_equiv_usd"] = (row["in_tokens"] * PRICE_IN_PER_M + row["out_tokens"] * PRICE_OUT_PER_M) / 1e6
    return row


# ---------------------------------------------------------------- report


def pct(series, p):
    s = series.dropna().sort_values().tolist()
    if not s:
        return float("nan")
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def summarize(df):
    print(f"\nmodel={generate.MODEL}  queries={len(df)}  cache hits={int(df.cache_hit.sum())}  "
          f"queries that needed a retry (429/5xx)={int((df.retries > 0).sum())}")
    print(f"{'stage':<12}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}")
    for col in ["retrieve_ms", "rerank_ms", "ttft_ms", "gen_ms", "e2e_ms"]:
        sub = df[col] if col in ("retrieve_ms", "rerank_ms") else df.loc[~df.cache_hit, col]
        print(f"{col:<12}{pct(sub, 50):>10.0f}{pct(sub, 95):>10.0f}{sub.mean():>10.0f}")
    if df.cache_hit.any():
        print(f"{'cached e2e':<12}{pct(df.loc[df.cache_hit, 'e2e_ms'], 50):>10.0f}"
              f"{pct(df.loc[df.cache_hit, 'e2e_ms'], 95):>10.0f}")
    miss = df[~df.cache_hit]
    print(f"\ntokens/query (uncached): in {miss.in_tokens.mean():.0f}, out {miss.out_tokens.mean():.0f}")
    print(f"actual cost: $0 (free tier)")
    print(f"paid-tier equivalent: ${df.paid_equiv_usd.mean():.5f}/query "
          f"(${df.paid_equiv_usd.mean() * 1000:.2f} per 1k queries, prices as of {PRICES_CHECKED_ON})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="latency_runs.csv")
    ap.add_argument("--cache", action="store_true", help="run every question twice: cold then warm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qs = json.loads(Path(args.questions).read_text())
    qs = qs["questions"] if isinstance(qs, dict) else qs
    random.Random(args.seed).shuffle(qs)
    qs = qs[: args.n]

    throttle = Throttle(FREE_RPM, FREE_RPD)
    run_one(qs[0], throttle, use_cache=False)  # warm-up, dropped
    CACHE.clear()

    order = qs + qs if args.cache else qs
    rows = []
    for i, q in enumerate(order, 1):
        try:
            rows.append(run_one(q, throttle, use_cache=args.cache))
        except RuntimeError as e:
            print(e)
            break
        if i % 10 == 0:
            print(f"{i}/{len(order)}")
            pd.DataFrame(rows).to_csv(args.out, index=False)  # checkpoint

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summarize(df)


if __name__ == "__main__":
    main()
