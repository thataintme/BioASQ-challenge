import bm25_search
import pandas as pd
import json
import re

selected_search_fn = bm25_search.search
output_fname = 'bm25_evals.csv'
k = 10

test_ds = None
with open ('Dataset/BioASQ-training14b/training14b.json', encoding='utf-8') as f:
    test_ds = json.load(f)

test_ds = pd.DataFrame(test_ds['questions'])

def reciprocal_rank(captured_pmids, gold_pmids):
    gold = set(gold_pmids)
    for rank, pmid in enumerate(captured_pmids, start=1):
        if pmid in gold:
            return 1.0 / rank
    return 0.0 

result = []
for idx, q in test_ds.iterrows():
    question = q.body
    search_results = selected_search_fn(question, k=k)

    captured_pmids = [x[0] for x in search_results]

    gold_pmids = [int(re.search(r'(\d+)/?$', url).group(1)) for url in q.documents]

    gold_captured = list(set(gold_pmids) & set(captured_pmids))
    rr = reciprocal_rank(captured_pmids, gold_pmids)
    

    result.append({'question':q.body,
                    'n_gold':len(q.documents),
                    'n_gold_captured':len(gold_captured),
                    f'recall@{k}':len(gold_captured)/len(gold_pmids),
                    'rr':rr})
    
df = pd.DataFrame(result)
df.to_csv(f'./search_evals/{output_fname}')

print('MRR = ', sum(df['rr'])/len(df['rr']))
print('Average recall@10 = ', sum(df[f'recall@{k}'])/len(df[f'recall@{k}']))