import json
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import random
import json

from generate_response import answer
random.seed(10)

MODEL='gemini-3.1-flash-lite'

NORMAL, SABOTAGED = 60, 40	# a bit more than half the questions should be sabotaged, by giving the model the wrong search results.
							# This is to ensure that we are able to evaluate the faithfulness of the model.
							# If not for this step, we may not notice if our judge-LLM fell to mode collapse on evaluating the truthfullness of the generator-LLM

search_results = []		# To enumerate during normal questions, and then use during the next sabotaged questions

def main():
	script_dir = Path(__file__).resolve().parent
	input_path = script_dir / "manual_eval/dev_questions.jsonl"
	output_path = script_dir / "manual_eval/generated_answers.jsonl"

	answers = []

	with input_path.open("r", encoding="utf-8") as source:
		input_lines = source.readlines()

	normal_count = int(len(input_lines) * NORMAL / (NORMAL + SABOTAGED))

	for line in tqdm(input_lines[:normal_count]):				# Normal questions (not sabotaged)
			if not line.strip():
				continue

			question = json.loads(line)
			docs, result = answer(question["body"], qtype="summary", model=MODEL, return_docs=True)
			if isinstance(result, str):
				result = json.loads(result)

			record = {
				"id": question["id"],
				"sabotaged": "no",
				"exact_answer": result.get("exact_answer"),
				"ideal_answer": result.get("ideal_answer"),
			}
			search_results.append(docs)
			json_record = json.dumps(record, ensure_ascii=False)
			answers.append(json_record)
			# time.sleep(10)

	for line in tqdm(input_lines[normal_count:]):			# Sabotaged questions with wrong search retrievals
			if not line.strip():
				continue

			question = json.loads(line)
			random_docs_list = random.choice(search_results)
			result = answer(question["body"], qtype="summary", model=MODEL, docs=random_docs_list)
			if isinstance(result, str):
				result = json.loads(result)

			record = {
				"id": question["id"],
				"sabotaged": "yes",
				"exact_answer": result.get("exact_answer"),
				"ideal_answer": result.get("ideal_answer"),
			}
			json_record = json.dumps(record, ensure_ascii=False)
			answers.append(json_record)
			# time.sleep(10)

	with output_path.open("w", encoding="utf-8") as destination:
		destination.write("\n".join(answers))
		if answers:
			destination.write("\n")


if __name__ == "__main__":
	main()
