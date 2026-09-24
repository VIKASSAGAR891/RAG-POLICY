import json
import random
from pathlib import Path


# Change this path if your CUAD_v1.json is somewhere else.
CUAD_PATH = Path(r"C:\Users\Vikas\Desktop\CUAD_v1\CUAD_v1.json")

OUTPUT_PATH = Path("data/evaluation/questions.jsonl")

NUM_QUESTIONS = 30
SEED = 42


def load_cuad(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_answerable_questions(data):
    rows = []

    for contract in data["data"]:
        contract_title = contract.get("title", "")

        for paragraph in contract.get("paragraphs", []):
            for qa in paragraph.get("qas", []):

                answers = qa.get("answers", [])

                if not answers:
                    continue

                # Ignore unanswerable questions.
                if qa.get("is_impossible", False):
                    continue

                answer_texts = [
                    a["text"].strip()
                    for a in answers
                    if a.get("text", "").strip()
                ]

                if not answer_texts:
                    continue

                rows.append(
                    {
                        "question": qa["question"],
                        "ground_truth": " ".join(
                            dict.fromkeys(answer_texts)
                        ),
                        "document": contract_title,
                        "qa_id": qa.get("id", ""),
                    }
                )

    return rows


def main():

    if not CUAD_PATH.exists():
        raise FileNotFoundError(
            f"CUAD file not found: {CUAD_PATH.resolve()}\n"
            "Place CUAD_v1.json at that location or update CUAD_PATH."
        )

    data = load_cuad(CUAD_PATH)

    rows = collect_answerable_questions(data)

    print(f"Total answerable CUAD questions found: {len(rows)}")

    random.seed(SEED)
    random.shuffle(rows)

    selected = rows[:NUM_QUESTIONS]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for row in selected:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Created evaluation set with "
        f"{len(selected)} questions."
    )

    print(
        f"Saved to: {OUTPUT_PATH.resolve()}"
    )

    print("\nFirst 5 evaluation questions:")

    for i, row in enumerate(selected[:5], start=1):
        print(f"\n{i}. {row['question']}")
        print(f"   Document: {row['document']}")
        print(f"   Ground truth: {row['ground_truth'][:250]}")


if __name__ == "__main__":
    main()