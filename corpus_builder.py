import pandas as pd
import numpy as np
import json
import requests
import time
import xml.etree.ElementTree as ET
import re

with open('Dataset/BioASQ-training14b/training14b.json', encoding="utf-8") as f:
    train_ds = json.load(f)

df = pd.DataFrame(train_ds['questions'])

selection = df[:5]

# counts by question type:
values, counts = np.unique_counts(df['type'])
print("Question counts by type:\n",dict(zip(values, counts.tolist())))


# Collect all gold PMIDs and get their paper titles
gold_pmids = set()
for i in df['documents']:
    for j in i:
        gold_pmids.add(re.search(r'(\d+)/?$', j).group(1))
gold_pmids = list(gold_pmids)

# Connecting to pubmed API
pubmed = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

def text_of(el):
    # If we get .text, we lose tags like <i>itallicised text</i>. so this helper function uses itertext()
    return "".join(el.itertext()).strip() if el is not None else ""

def first_author(art): # PubMed lists authors in publication order so first found author should be fine
    author = art.find(".//MedlineCitation/Article/AuthorList/Author")
    if author is None:
        return None
    # Some papers are authored by a group/consortium instead of a person
    collective = author.findtext("CollectiveName")
    if collective:
        return f'(group) {collective}'
    last = author.findtext("LastName", default="")
    initials = author.findtext("Initials", default="")
    return f"{last} {initials}".strip()

docs = []

i = 0
while i<len(gold_pmids): # The API has a silent max of 10000 pmids per query so doing it in batches of 5000
    selected_gold = gold_pmids[i:i+5000]

    # POST puts the IDs in the request body, so the URL stays short (no 414)
    resp = requests.post(pubmed + "efetch.fcgi",
                            data={"db": "pubmed", "id": ",".join(selected_gold), "retmode": "xml"})
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID")
        title = text_of(art.find(".//ArticleTitle"))
        author = first_author(art)
        parts = []
        for ab in art.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")  # e.g. BACKGROUND, METHODS in structured abstracts
            parts.append(f"{label}: {text_of(ab)}" if label else text_of(ab))
        docs.append({
                    "pmid": pmid,
                    "author": author,
                    "title": title,
                    "abstract": " ".join(parts)
                    })
    i = i + 5000

with open("corpus.jsonl", "w") as f:
    for d in docs:
        f.write(json.dumps(d) + "\n")

found = {d["pmid"] for d in docs}
# known gap: a handful of gold PMIDs are <PubmedBookArticle> entries, not <PubmedArticle>,
# so they're skipped by the findall above and always end up here (see future_work/missing_pmids.txt)
missing = [p for p in gold_pmids if p not in found]
no_abstract = [d["pmid"] for d in docs if not d["abstract"]]
print(f"{len(docs)} fetched, {len(missing)} missing, {len(no_abstract)} without abstract")

print()
