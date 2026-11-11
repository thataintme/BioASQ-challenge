"""Latency benchmark against the GPU Space: every stage, one run, one n.

Per query (same sample as the CPU benchmarks: training14b shuffled with --seed, first --n,
plus one warm-up query that's run and dropped):
    bm25_ms, dense_ms, fusion_ms, retrieve_ms, rerank_ms   timed inside the Space's GPU call
    client_search_ms   wall time of the whole search request as seen from here (adds network,
                       queueing and ZeroGPU's GPU attach on top of the stages above)
    ttft_ms, gen_ms    Gemini generation over the reranked top 10, timed on the Space
    e2e_ms             retrieve_ms + rerank_ms + gen_ms
    rerank_device, dense_device   where the models actually ran (flags any CPU fallback)

Saves after every query and resumes from --out, so a run cut short by the daily ZeroGPU
quota can be continued later with the same command. Stops at the first quota error.

Run from the repo root (reads HF_TOKEN and BENCH_TOKEN from .env):
    python hf_space/latency_bench.py Dataset/BioASQ-training14b/training14b.json --n 100
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
from gradio_client import Client

SPACE_ID = "santhosh-m/bioasq-rag"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 5


class Throttle:
    """Keeps Gemini calls under the free tier's requests-per-minute limit."""

    def __init__(self, rpm):
        self.rpm, self.sent = rpm, []

    def wait(self):
        while True:
            now = time.time()
            self.sent = [t for t in self.sent if now - t < 60]
            if len(self.sent) < self.rpm:
                break
            time.sleep(60 - (now - self.sent[0]) + 0.05)
        self.sent.append(time.time())


def pct(values, p):
    s = sorted(v for v in values if pd.notna(v))
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))] if s else float("nan")


def run_one(client, token, q, throttle):
    row = {"qid": q["id"], "qtype": q["type"]}
    t0 = time.perf_counter()
    s = client.predict(q["body"], token, api_name="/bench_search")
    row["client_search_ms"] = (time.perf_counter() - t0) * 1000
    pmids = s.pop("pmids")
    row.update(s)

    retries = 0
    while True:
        throttle.wait()
        g = client.predict(q["body"], q["type"], pmids, token, api_name="/bench_generate")
        if "error" not in g:
            break
        if g.get("status") not in RETRYABLE_STATUS or retries >= MAX_RETRIES:
            raise RuntimeError(f"generation failed: {g['error']}")
        retries += 1
        time.sleep(min(60, 2 ** retries) + random.random())
    row.update(ttft_ms=g["ttft_ms"], gen_ms=g["gen_ms"], in_tokens=g["in_tokens"],
               out_tokens=g["out_tokens"], gen_model=g["model"], gen_retries=retries)
    row["e2e_ms"] = row["retrieve_ms"] + row["rerank_ms"] + row["gen_ms"]
    return row


def summarize(df):
    print(f"\nqueries={len(df)}  devices: rerank={sorted(df.rerank_device.unique())} "
          f"dense={sorted(df.dense_device.unique())}  generation retries={int(df.gen_retries.sum())}")
    print(f"{'stage':<18}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}")
    for col in ["bm25_ms", "dense_ms", "fusion_ms", "retrieve_ms", "rerank_ms",
                "client_search_ms", "ttft_ms", "gen_ms", "e2e_ms"]:
        print(f"{col:<18}{pct(df[col], 50):>10.1f}{pct(df[col], 95):>10.1f}{df[col].mean():>10.1f}")
    print(f"\ntokens/query: in {df.in_tokens.mean():.0f}, out {df.out_tokens.mean():.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="latency_runs_gpu.csv")
    ap.add_argument("--rpm", type=int, default=10, help="Gemini requests/minute to stay under")
    args = ap.parse_args()

    qs = json.loads(Path(args.questions).read_text())
    qs = qs["questions"] if isinstance(qs, dict) else qs
    random.Random(args.seed).shuffle(qs)
    qs = qs[: args.n]

    out = Path(args.out)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {r["qid"] for r in rows}
    todo = [q for q in qs if q["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to go")

    env = dotenv_values(".env")
    client = Client(SPACE_ID, token=env["HF_TOKEN"], verbose=False)
    token, throttle = env["BENCH_TOKEN"], Throttle(args.rpm)

    if todo:
        # warm-up, dropped: first call pays GPU attach + moving weights to the GPU
        t0 = time.perf_counter()
        client.predict(qs[0]["body"], token, api_name="/bench_search")
        print(f"warm-up: {(time.perf_counter() - t0) * 1000:.0f} ms (dropped)")

    for q in todo:
        try:
            row = run_one(client, token, q, throttle)
        except Exception as e:
            msg = str(e)
            print(f"\nSTOPPED after {len(rows)} queries: {type(e).__name__}: {msg[:300]}")
            if "quota" in msg.lower():
                print("ZeroGPU quota reached - rerun the same command later to resume.")
            break
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)

        prev = [r["rerank_ms"] for r in rows[:-1]]
        flag = ""
        if row["rerank_device"] != "cuda:0" or row["dense_device"] != "cuda:0":
            flag = "  <-- NOT ON GPU"
        elif len(prev) >= 5 and row["rerank_ms"] > 5 * statistics.median(prev):
            flag = "  <-- rerank >5x the median so far"
        print(f"{len(rows):>3}/{len(qs)} rerank={row['rerank_ms']:7.1f}ms retrieve={row['retrieve_ms']:6.1f}ms "
              f"search(client)={row['client_search_ms']:7.0f}ms gen={row['gen_ms']:6.0f}ms "
              f"[{row['rerank_device']}]{flag}", flush=True)

    if rows:
        summarize(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
