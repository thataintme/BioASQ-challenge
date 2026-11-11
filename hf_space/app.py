"""Hugging Face Spaces (ZeroGPU) backend for the web showcase (web/index.html).

Same pipeline and same two endpoints as server.py (the Flask version, for any plain
CPU host), exposed through Gradio because ZeroGPU only runs Gradio apps. The static
page calls these with the @gradio/client JS library.

Retrieval and reranking both run inside one short @spaces.GPU call (see the note at
the .to("cuda") moves below for why the dense encoder can't stay on CPU), with a 10s
reservation so each search costs visitors little of their ZeroGPU quota.

Deployed by hf_space/deploy.py, which uploads this file as the Space's app.py next to
the pipeline modules. Local run (CPU, no ZeroGPU), from the repo root:
    PYTHONPATH=. python hf_space/app.py
"""

import hmac
import os
import time
from collections import defaultdict

import spaces  # must be imported before anything initialises CUDA

import gradio as gr

import bm25_search
import dense_search
import hybrid_search
import reranker
import generate_response
from corpus import CORPUS

ON_ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))

POOL = 100
TOP_K = 10
MAX_QUESTION_LEN = 500

# Same reasoning as server.py: flash-lite's free tier survives public traffic.
PUBLIC_GENERATION_MODEL = "gemini-3.5-flash-lite"
GENERATE_RATE_LIMIT = 5       # max generate calls
GENERATE_RATE_WINDOW = 3600   # per this many seconds, per IP
_generate_log = defaultdict(list)

# On ZeroGPU there's no GPU at startup, so both modules picked "cpu" at import time.
# Moving the models to "cuda" here is what ZeroGPU expects: the move is deferred and
# applied whenever a @spaces.GPU function runs.
#
# Every model forward pass has to happen inside the @spaces.GPU function: running the
# dense query encoder on CPU in the main process first (even with the GPU call after it)
# makes ZeroGPU's GPU worker fail with "RuntimeError: No CUDA GPUs are available".
# Found by bisecting on the Space itself; imports alone, BM25, and the .to("cuda")
# moves were all fine.
if ON_ZERO_GPU:
    dense_search.DEVICE = reranker.DEVICE = "cuda"
    dense_search.q_model.to("cuda")
    reranker.model.to("cuda")


def _client_ip(request: gr.Request):
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(ip):
    now = time.time()
    _generate_log[ip] = [t for t in _generate_log[ip] if now - t < GENERATE_RATE_WINDOW]
    if len(_generate_log[ip]) >= GENERATE_RATE_LIMIT:
        return True
    _generate_log[ip].append(now)
    return False


def _ms_since(t0):
    return (time.perf_counter() - t0) * 1000


@spaces.GPU(duration=10)
def _retrieve_and_rerank(question):
    """Returns (top PMIDs, per-stage timings). Stages are timed inside the GPU call, so
    they exclude network, queueing and GPU attach time. The .tolist()/.numpy() at the end
    of each model call forces a GPU sync, so the timings aren't cut short by async CUDA."""
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    bm25 = bm25_search.search(question, k=POOL)
    bm25_ms = _ms_since(t0)

    t0 = time.perf_counter()
    dense = dense_search.search(question, k=POOL)
    dense_ms = _ms_since(t0)

    t0 = time.perf_counter()
    candidates = [pmid for pmid, _ in hybrid_search.fuse(bm25, dense)[:POOL]]
    fusion_ms = _ms_since(t0)

    t0 = time.perf_counter()
    top = reranker.rerank(question, candidates, k=TOP_K)
    rerank_ms = _ms_since(t0)

    timings = dict(
        bm25_ms=bm25_ms, dense_ms=dense_ms, fusion_ms=fusion_ms,
        retrieve_ms=bm25_ms + dense_ms + fusion_ms, rerank_ms=rerank_ms,
        gpu_fn_ms=_ms_since(t_all),
        # which device the models really ran on, so a CPU fallback can't hide in the numbers
        rerank_device=str(next(reranker.model.parameters()).device),
        dense_device=str(next(dense_search.q_model.parameters()).device),
    )
    return top, timings


def _docs_for(top):
    docs = CORPUS.loc[[str(p) for p in top]].reset_index().to_dict("records")
    return [
        {"pmid": d["pmid"], "title": d["title"], "abstract": d["abstract"],
         "url": f"https://pubmed.ncbi.nlm.nih.gov/{d['pmid']}/"}
        for d in docs
    ]


def search(question: str) -> list:
    """Hybrid retrieval (BM25 + MedCPT dense, RRF) for 100 candidates, then MedCPT
    cross-encoder rerank to the top 10."""
    question = (question or "").strip()
    if not question:
        raise gr.Error("question is required")
    if len(question) > MAX_QUESTION_LEN:
        raise gr.Error(f"question is too long (max {MAX_QUESTION_LEN} characters)")

    top, _ = _retrieve_and_rerank(question)
    return _docs_for(top)


def generate(question: str, docs: list, request: gr.Request) -> dict:
    """Cited Gemini answer over the given docs. The exact-answer shape (yes/no, list,
    or none) is inferred from the question, since visitors don't label its type."""
    if _rate_limited(_client_ip(request)):
        raise gr.Error("Rate limit reached for this demo - try again in a bit.")
    question = (question or "").strip()
    if not question or not docs:
        raise gr.Error("question and docs are required")

    result = generate_response.answer(
        question, qtype="auto",
        docs=[{"pmid": d["pmid"], "title": d["title"], "abstract": d["abstract"]} for d in docs],
        model=PUBLIC_GENERATION_MODEL,
    )
    return {"exact_answer": result["exact_answer"], "ideal_answer": result["ideal_answer"]}


# ---------------------------------------------------------------- latency benchmark
# Used by hf_space/latency_bench.py. Gated by the BENCH_TOKEN Space secret (disabled
# when it's unset): bench_generate deliberately skips the per-IP rate limit, so it must
# not be callable by the public.

def _check_bench_token(token):
    expected = os.getenv("BENCH_TOKEN")
    if not expected or not hmac.compare_digest(str(token), expected):
        raise gr.Error("not authorized")


def bench_search(question: str, token: str) -> dict:
    _check_bench_token(token)
    top, timings = _retrieve_and_rerank(question)
    return {"pmids": [str(p) for p in top], **timings}


def _usage_from(event):
    for obj in (getattr(event, "usage", None), getattr(getattr(event, "interaction", None), "usage", None)):
        if obj is None:
            continue
        tin = getattr(obj, "total_input_tokens", None) or getattr(obj, "input_tokens", None)
        tout = getattr(obj, "total_output_tokens", None) or getattr(obj, "output_tokens", None)
        if tin or tout:
            return tin, tout
    return None, None


def bench_generate(question: str, qtype: str, pmids: list, token: str) -> dict:
    """Streams one generation and times it: ttft_ms = request sent -> first answer token,
    gen_ms = request sent -> last token. On an API error, returns its HTTP status instead
    of raising, so the caller can retry 429/5xx with backoff."""
    _check_bench_token(token)
    docs = CORPUS.loc[[str(p) for p in pmids]].reset_index().to_dict("records")
    prompt = generate_response.USER_TEMPLATE.format(
        question=question, context=generate_response.format_context(docs),
        exact_rule=generate_response.EXACT_ANSWER_RULES.get(qtype, generate_response.EXACT_ANSWER_RULES["auto"]),
    )
    t0 = time.perf_counter()
    ttft = None
    usage = (None, None)
    try:
        stream = generate_response.CLIENT.interactions.create(
            model=PUBLIC_GENERATION_MODEL, system_instruction=generate_response.SYSTEM_STRICT,
            input=prompt, stream=True,
        )
        for event in stream:
            if event.event_type == "step.delta" and event.delta.type == "text" and ttft is None:
                ttft = _ms_since(t0)
            u = _usage_from(event)
            if u != (None, None):
                usage = u
    except Exception as e:
        status = next((getattr(e, a) for a in ("code", "status_code", "status")
                       if isinstance(getattr(e, a, None), int)), None)
        return {"error": f"{type(e).__name__}: {e}"[:300], "status": status}
    return {"ttft_ms": ttft, "gen_ms": _ms_since(t0), "in_tokens": usage[0], "out_tokens": usage[1],
            "model": PUBLIC_GENERATION_MODEL}


with gr.Blocks(title="BioASQ Biomedical Literature RAG") as demo:
    gr.Markdown(
        "# BioASQ Biomedical Literature RAG\n"
        "API backend for the project showcase page. Source, benchmarks and methodology: "
        "[github.com/thataintme/BioASQ-challenge](https://github.com/thataintme/BioASQ-challenge)."
    )
    gr.api(search, api_name="search")
    gr.api(generate, api_name="generate")
    gr.api(bench_search, api_name="bench_search", api_visibility="undocumented")
    gr.api(bench_generate, api_name="bench_generate", api_visibility="undocumented")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
