
import json
from retrieval import Retriever

retriever = Retriever("corpus/corpus.db")

strategies = {
    "Baseline": {
        "remove_stopwords": False,
        "diversify": False,
        "and_first": False
    },
    "Filtered + Diversity": {
        "remove_stopwords": True,
        "diversify": True,
        "and_first": False
    },
    "Raw BM25 + Diversity": {
        "remove_stopwords": False,
        "diversify": True,
        "and_first": False
    },
    "AND-first + Diversity": {
        "remove_stopwords": True,
        "diversify": True,
        "and_first": True
    }
}

with open("dev_eval/single_turn.jsonl") as f:
    questions = [
        json.loads(line)
        for line in f if line.strip()
    ]


def hit_rate(predicted, gold):
    if not gold:
        return 1.0

    return len(set(predicted) & set(gold)) / len(set(gold))


for name, settings in strategies.items():

    scores = []

    for item in questions:
        results = retriever.retrieve(
            item["question"],
            k=5,
            **settings
        )

        predicted = [
            r["paper_id"] for r in results
        ]

        gold = item["gold_paper_ids"]

        score = hit_rate(predicted, gold)
        scores.append(score)

    avg = sum(scores) / len(scores)

    print(f"{name}: {avg:.1%}")
