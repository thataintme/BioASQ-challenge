"""
Generates 100 retrievals per question using hybrid retrieval and stores retrieved document pmids into ./bi-encoder_retrievals/

Adjust number of retrievals with K
"""

import pandas as pd
import hybrid_search
from tqdm import tqdm
import json
import re

# METHODS = {"bm25": bm25_search.search, "dense": dense_search.search, "hybrid": hybrid_search.search}
K = 100     # Using high K to get a generous amount of retrievals which the reranker can then apply cross-encoder on for top-10 results
search = hybrid_search.search

with open('./Dataset/BioASQ-training14b/training14b.json', encoding='utf-8') as f:
    questions = json.load(f)
questions = questions['questions']


retrievals_list = []
for q in tqdm(questions):
    search_results = [p for p, _ in search(q["body"], k=K)]
    retrievals_list.append({"question": q["body"], "qid": q["id"], "type": q["type"],
                    "retrievals": search_results})
df = pd.DataFrame([r for r in retrievals_list])

pd.DataFrame(retrievals_list).to_csv("retrievals/hybrid_search_retrievals_k100.csv", index=False)
