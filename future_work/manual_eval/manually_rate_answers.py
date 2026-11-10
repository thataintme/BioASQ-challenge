import pandas as pd


dev_questions = pd.read_json("./manual_eval/dev_questions.jsonl", lines=True, encoding="utf-8")
generated_answers = pd.read_json("./manual_eval/generated_answers.jsonl", lines=True, encoding="utf-8")
corpus = pd.read_json("./corpus.jsonl", lines=True, encoding="utf-8")

generated_answers = generated_answers.rename(
	columns={
		"exact_answer": "generated_exact_answer",
		"ideal_answer": "generated_ideal_answer",
	}
)

merged_answers = dev_questions.merge(
	generated_answers,
	on="id",
	suffixes=("", "_generated"),
)
merged_answers = merged_answers[
	[
		"body",
		"type",
		"id",
		"exact_answer",
		"ideal_answer",
		"generated_exact_answer",
		"generated_ideal_answer",
	]
]

ratings = []


def collect_pmids():
    pmids = []
    while True:
        entered_pmids = input(
            "Enter PMID(s), separated by commas or spaces (-1 when finished): "
        ).strip()
        if not entered_pmids:
            continue
        entered_pmids = entered_pmids.replace(",", " ").split()
        finished = "-1" in entered_pmids
        pmids.extend(pmid for pmid in entered_pmids if pmid != "-1")
        if finished:
            break

    for pmid in pmids:
        matching_articles = corpus[corpus["pmid"].astype(str) == pmid]
        if matching_articles.empty:
            print("No corpus entry found for PMID", pmid)
        else:
            for abstract in matching_articles["abstract"]:
                print(f"Abstract [{pmid}]: {abstract}")
    return pmids


for _, answer in merged_answers.iterrows():
    print("\nQuestion:", answer["body"])
    print("Generated exact answer:", answer["generated_exact_answer"])
    print("Generated ideal answer:", answer["generated_ideal_answer"])
    print("----------------")
    is_summary = answer["type"] == "summary"
    if is_summary:
        workflow_choice = input(
            "Press Enter for summary workflow, or -1 to switch to claims: "
        ).strip()
        is_summary = workflow_choice != "-1"
        if is_summary:
            claim_count = None

    if is_summary:
        pmids = collect_pmids()
        while True:
            try:
                summary_rating = int(input("Rate the summary from 1 to 5: "))
                if 1 <= summary_rating <= 5:
                    break
            except ValueError:
                pass
            print("Enter a rating from 1 to 5.")
        ratings.append(
            {
                "id": answer["id"],
                "type": answer["type"],
                "body": answer["body"],
                "workflow": "summary",
                "claim_count": claim_count,
                "pmids": pmids,
                "correct_claims": None,
                "wrong_claims": None,
                "summary_rating": summary_rating,
            }
        )
        continue

    pmids = collect_pmids()

    while True:
        try:
            claim_count = int(input("How many claims are there? (enter -2 for summary): "))
            if claim_count >= 0 or claim_count == -2:
                break
        except ValueError:
            pass
        print("Enter a non-negative number of claims, or -2 for summary.")
    is_summary = claim_count == -2

    if is_summary:
        workflow_choice = input(
            "Press Enter for summary workflow, or -1 to switch to claims: "
        ).strip()
        if workflow_choice != "-1":
            while True:
                try:
                    summary_rating = int(input("Rate the summary from 1 to 5: "))
                    if 1 <= summary_rating <= 5:
                        break
                except ValueError:
                    pass
                print("Enter a rating from 1 to 5.")
            ratings.append(
                {
                    "id": answer["id"],
                    "type": answer["type"],
                    "body": answer["body"],
                    "workflow": "summary",
                    "claim_count": claim_count,
                    "pmids": pmids,
                    "correct_claims": None,
                    "wrong_claims": None,
                    "summary_rating": summary_rating,
                }
            )
            continue
        while True:
            try:
                claim_count = int(input("How many claims are there? "))
                if claim_count >= 0:
                    break
            except ValueError:
                pass
            print("Enter a non-negative number of claims.")

    while True:
        try:
            correct_claims = int(input("How many claims were correct? "))
            if 0 <= correct_claims <= claim_count:
                break
        except ValueError:
            pass
        print("Enter a number of correct claims from 0 to the total claims.")
    wrong_claims = claim_count - correct_claims
    ratings.append(
        {
            "id": answer["id"],
            "type": answer["type"],
            "body": answer["body"],
            "workflow": "claims",
            "claim_count": claim_count,
            "pmids": pmids,
            "correct_claims": correct_claims,
            "wrong_claims": wrong_claims,
            "summary_rating": None,
        }
    )

ratings_df = pd.DataFrame(ratings)
print("\nRatings:")
print(ratings_df)
ratings_df.to_csv('human_evaluations.csv')