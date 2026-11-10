import pandas as pd
import bm25_search, dense_search, hybrid_search
from tqdm import tqdm
import json
import re

METHODS = {"bm25": bm25_search.search, "dense": dense_search.search, "hybrid": hybrid_search.search}
K = 10

with open('./Dataset/BioASQ-training14b/training14b.json', encoding='utf-8') as f:
    questions = json.load(f)
questions = questions['questions']

rows, metrics_per_q = [], []
for name, search in METHODS.items():
    for q in tqdm(questions, desc=name):
        gold = set()
        for i in q['documents']:
            gold.add(int(re.search(r'(\d+)/?$', i).group(1)))
        top = [p for p, _ in search(q["body"], k=K)]
        hits = [i for i, p in enumerate(top, 1) if p in gold]
        metrics_per_q.append({"method": name, "qid": q["id"], "type": q["type"],
                      "recall@10": len(gold & set(top)) / len(gold),
                      "rr": 1 / hits[0] if hits else 0.0})
    df = pd.DataFrame([r for r in metrics_per_q if r["method"] == name])
    rows.append({"method": name, "n": len(df),
                 "mean_recall@10": df["recall@10"].mean(), "mrr@10": df["rr"].mean()})
    print(rows[-1])

pd.DataFrame(metrics_per_q).to_csv("search_evals/per_question_all.csv", index=False)
pd.DataFrame(rows).to_csv("search_evals/summary.csv", index=False)