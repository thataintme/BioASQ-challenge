"""Upload the Hugging Face Space: hf_space/ files + the pipeline modules + the corpus data.

Run from the repo root (reads HF_TOKEN and GEMINI_API_KEY from .env):
    python hf_space/deploy.py              # code only
    python hf_space/deploy.py --data       # also upload corpus.jsonl + corpus_embeddings.pt (~270 MB)
    python hf_space/deploy.py --secrets    # also (re)set GEMINI_API_KEY and BENCH_TOKEN as Space secrets
"""

import argparse

from dotenv import dotenv_values
from huggingface_hub import CommitOperationAdd, HfApi

SPACE_ID = "santhosh-m/bioasq-rag"

# local path -> path in the Space repo
CODE_FILES = {
    "hf_space/app.py": "app.py",
    "hf_space/requirements.txt": "requirements.txt",
    "hf_space/README.md": "README.md",
    "corpus.py": "corpus.py",
    "bm25_search.py": "bm25_search.py",
    "dense_search.py": "dense_search.py",
    "hybrid_search.py": "hybrid_search.py",
    "reranker.py": "reranker.py",
    "generate_response.py": "generate_response.py",
}
DATA_FILES = {
    "corpus.jsonl": "corpus.jsonl",
    "corpus_embeddings.pt": "corpus_embeddings.pt",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store_true", help="also upload the corpus data files")
    parser.add_argument("--secrets", action="store_true", help="also set GEMINI_API_KEY and BENCH_TOKEN as Space secrets")
    args = parser.parse_args()

    env = dotenv_values(".env")
    api = HfApi(token=env["HF_TOKEN"])

    if args.secrets:
        for name in ("GEMINI_API_KEY", "BENCH_TOKEN"):  # BENCH_TOKEN gates the benchmark endpoints
            if env.get(name):
                api.add_space_secret(SPACE_ID, name, env[name])
                print("set secret", name)

    files = dict(CODE_FILES, **(DATA_FILES if args.data else {}))
    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src) for src, dst in files.items()]
    commit = api.create_commit(
        SPACE_ID, repo_type="space", operations=ops,
        commit_message="Deploy from BioASQ-challenge" + (" (with data)" if args.data else ""),
    )
    print("uploaded:", ", ".join(files.values()))
    print(commit.commit_url)


if __name__ == "__main__":
    main()
