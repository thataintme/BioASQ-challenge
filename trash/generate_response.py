from google import genai
import requests
import xml.etree.ElementTree as ET
import bm25_search
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

GENERATOR_API_KEY = os.getenv('GEMINI_API_KEY')



# Now calling the LLM. First use the selected search algorithm and get a list of documents
# Then use PubMed API to get their titles and abstracts.


selected_search_fn = bm25_search.search
# user_question = "What is CPAP and does it cure sleep apnea?"
user_question = "What is the difference between Cushing Syndrome and Cushing Disease?"
k = 20

search_results = selected_search_fn(user_question, k=k)
pmids = [x[0] for x in search_results]



print()
corpus = pd.read_json('./corpus.jsonl', lines=True)
# docs = corpus[corpus['pmid'].isin(pmids)] # Rookie mistake: This loses the order in which bm25 returns results
docs = corpus.set_index("pmid").loc[pmids].reset_index()

prompt = f"""
{user_question}

Use the below documents as context:
{docs.to_json(orient="records")}"""


client = genai.Client(api_key=GENERATOR_API_KEY)
stream = client.interactions.create(
    # model="gemini-3.8-flash",
    model="gemini-3.6-flash",
    system_instruction=
        "You are a helpful assistant. Answer the user's question accurately using ONLY "
        "the provided context. The context is a list of articles each with their author name, title, abstract, and pmid "
        "For each sentence that you write, include a list of citations that cover all your claims in that paragraph. "
        "Citations should be given as a list of pmids in square brackets [] "
        "If the answer cannot be found in the context, state that "
        "you do not know and that the answer was not found in the corpus.\n"
        "Provide your answer as a JSON with two fields: {\"exact_answer\": ..., \"ideal_answer\": \"...\"}\n"
        "exact_answer is to the point. It could be a yes/no; or a list of items. "
        "ideal_answer is more natural human-readable.\n"
        "Answers that need a summary (not yes/no, and not a list) do not require a value for the exact_answer field.",

    input=prompt,
    stream=True
)

for event in stream:
    if event.event_type == "step.delta":
        if event.delta.type == "text":
            print(event.delta.text, end="", flush=True)
        elif event.delta.type == "image":
            print(f"\n[Image chunk: {len(event.delta.data)} bytes]", end="", flush=True)



