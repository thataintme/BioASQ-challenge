import bm25s
import Stemmer
import json
import pandas as pd
import re

from corpus import PMIDS, TEXTS

stemmer = Stemmer.Stemmer("english")
texts = TEXTS

# Normally bm25s.tokenize splits on space and punctuation, which is not good in medical search
# # example Vitamin-D, Covid-19, etc. So use custom regex
# The below pattern just tokenizes by split by spaces, and ignores trailing special characters (capture word character, and end with )
token_pattern = r"\w(?:\S*\w)?"

corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, token_pattern=token_pattern)

retriever = bm25s.BM25()
retriever.index(corpus_tokens)

def search(question, k=10):
    q = bm25s.tokenize(question, stopwords="en", stemmer=stemmer, token_pattern=token_pattern)
    idx, scores = retriever.retrieve(q, k=k)
    return [(int(PMIDS[i]), s) for i, s in zip(idx.tolist()[0], scores.tolist()[0])]




# Debug test out the above search. Pickicking questions from dataset, perform the search, and see how many are gold
# test_ds = None
# with open('./Dataset/Task14BGoldenEnriched/14B1_golden.json', encoding='utf-8') as f:
#     test_ds = json.load(f)
# test_data = pd.DataFrame(test_ds['questions'])
# question_indices = [3,26,75]
# questions = test_data.loc[question_indices].copy()
# result = []
# for idx, q in questions.iterrows():
#     search_results = search(q['body'], len(q.documents))
#     captured_pmids = [x[0] for x in search_results]
#     gold_pmids = [int(d.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '')) for d in q.documents]
#     gold_captured = list(set(gold_pmids) & set(captured_pmids))

#     result.append({'question':q.body,
#                     'n_gold':len(q.documents),
#                     'n_gold_captured':len(gold_captured),
#                     'percentage_captured':100*len(gold_captured)/len(q.documents)})
# print(pd.DataFrame(result))
