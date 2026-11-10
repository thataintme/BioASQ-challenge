# Run this after collecting cross-encoder retrieval pmids on Colab (or any GPU environment)
#
# Usage:
#   python cross-encoder_eval.py --input retrievals/reranked_top10_ml256.csv \
#       --output search_evals/cross-encoder-reranked_ml256.csv --label "Cross-Encoder ml=256"

import argparse
import json
import re
import ast

import pandas as pd
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="retrievals/reranked_top10_ml512.csv")
ap.add_argument("--output", default="search_evals/cross-encoder-reranked_ml512.csv")
ap.add_argument("--label", default="Cross-Encoder ml=512")
args = ap.parse_args()

with open('./Dataset/BioASQ-training14b/training14b.json', encoding='utf-8') as f:
    questions = json.load(f)
questions = questions['questions']

cx_enc_retrievals = pd.read_csv(args.input)
cx_enc_retrievals['reranked'] = cx_enc_retrievals['reranked'].apply(ast.literal_eval)   # converts string representation of the list (reranked) back to list


rows, metrics_per_q = [], []
for q in tqdm(questions):
    gold = set()
    for i in q['documents']:
        gold.add(int(re.search(r'(\d+)/?$', i).group(1)))

    retrieval = cx_enc_retrievals[cx_enc_retrievals['qid'] == q['id']].iloc[0]
    top = retrieval['reranked']
    hits = [i for i, p in enumerate(top, 1) if p in gold]
    metrics_per_q.append({"qid": q["id"], "type": q["type"],
                    "recall@10": len(gold & set(top)) / len(gold),
                    "rr": 1 / hits[0] if hits else 0.0})
df = pd.DataFrame([r for r in metrics_per_q])
rows.append({"method": args.label, "n": len(df),
                "mean_recall@10": df["recall@10"].mean(), "mrr@10": df["rr"].mean()})
print(rows[-1])


pd.DataFrame(metrics_per_q).to_csv(args.output, index=False)
