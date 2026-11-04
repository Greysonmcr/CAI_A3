
from retrieval import Retriever

retriever = Retriever("corpus/corpus.db")

question = "How are conversational agents evaluated?"

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

for name, settings in strategies.items():
    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)

    results = retriever.retrieve(
        question,
        k=5,
        **settings
    )

    for result in results:
        print(
            f"{result['paper_id']}: "
            f"{result['title']}"
        )
