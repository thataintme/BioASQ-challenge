# select 15 random questions from the training dataset, generate responses, and rate them manually

import pandas as pd
import random
import json

random.seed(10)

questions = None
with open('Dataset/BioASQ-training14b/training14b.json') as f:
    questions = json.load(f)

questions = pd.DataFrame(questions['questions'])
dev_set = random.sample(sorted(questions['id']), 20)
dev_questions = questions[questions['id'].isin(dev_set)]
dev_questions.to_json('./manual_eval/dev_questions.jsonl', orient="records", lines=True)

