"""Flask API behind the live web showcase (web/index.html): retrieval + generation
over HTTP, so the static page can be hosted anywhere (your website) and just call
this server wherever it's actually running.

Run:
    python server.py
"""

import time
from collections import defaultdict

from flask import Flask, jsonify, request
from flask_cors import CORS

import hybrid_search
import reranker
import generate_response
from corpus import CORPUS

app = Flask(__name__)
CORS(app)

POOL = 100
TOP_K = 10

# gemini-3.5-flash-lite, not generate_response.MODEL (gemini-3.6-flash): this endpoint
# is open to the public internet, and 3.6-flash's free tier (20 req/day, confirmed) would
# be exhausted by a handful of visitors. Flash-lite's much higher reported daily cap can
# actually survive real traffic. See README "Latency with generation" for why.
PUBLIC_GENERATION_MODEL = "gemini-3.5-flash-lite"

# Retrieval isn't rate-limited the same way - it costs this server's CPU, not a shared
# daily API quota, and each request already takes ~1 minute of that CPU, which is
# self-limiting enough. Generation hits a real external quota, so it gets a real cap.
GENERATE_RATE_LIMIT = 5       # max /api/generate calls
GENERATE_RATE_WINDOW = 3600   # per this many seconds, per IP
_generate_log = defaultdict(list)


def _rate_limited(ip):
    now = time.time()
    _generate_log[ip] = [t for t in _generate_log[ip] if now - t < GENERATE_RATE_WINDOW]
    if len(_generate_log[ip]) >= GENERATE_RATE_LIMIT:
        return True
    _generate_log[ip].append(now)
    return False


def pubmed_url(pmid):
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


@app.post("/api/search")
def search():
    body = request.get_json(force=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify(error="question is required"), 400
    if len(question) > 500:
        return jsonify(error="question is too long (max 500 characters)"), 400

    candidates = [pmid for pmid, _ in hybrid_search.search(question, k=POOL)]
    top = reranker.rerank(question, candidates, k=TOP_K)
    docs = CORPUS.loc[[str(p) for p in top]].reset_index().to_dict("records")
    return jsonify(results=[
        {"pmid": d["pmid"], "title": d["title"], "abstract": d["abstract"], "url": pubmed_url(d["pmid"])}
        for d in docs
    ])


@app.post("/api/generate")
def generate():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    if _rate_limited(ip):
        return jsonify(error="Rate limit reached for this demo - try again in a bit."), 429

    body = request.get_json(force=True) or {}
    question = (body.get("question") or "").strip()
    qtype = body.get("qtype") or "summary"
    docs = body.get("docs") or []
    if not question or not docs:
        return jsonify(error="question and docs are required"), 400
    if qtype not in generate_response.EXACT_ANSWER_RULES:
        qtype = "summary"

    result = generate_response.answer(
        question, qtype=qtype,
        docs=[{"pmid": d["pmid"], "title": d["title"], "abstract": d["abstract"]} for d in docs],
        model=PUBLIC_GENERATION_MODEL,
    )
    return jsonify(exact_answer=result["exact_answer"], ideal_answer=result["ideal_answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
