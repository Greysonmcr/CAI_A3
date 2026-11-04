
import sqlite3
import re


class Retriever:

    STOPWORDS = {
        "what", "how", "why", "when", "where",
        "who", "is", "are", "was", "were",
        "does", "do", "did", "the", "a", "an",
        "in", "on", "of", "for", "to", "and",
        "or", "with", "it", "this", "that",
        "work", "works"
    }

    def __init__(self, db_path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row

    def retrieve(
        self,
        query,
        k=5,
        remove_stopwords=False,
        diversify=True,
        and_first=False
    ):
        if k <= 0:
            return []

        # 1. Tokenize query
        terms = re.findall(r"\w+", query.lower())

        # 2. Optionally remove stopwords
        if remove_stopwords:
            filtered = [
                word for word in terms
                if word not in self.STOPWORDS
            ]
            if filtered:
                terms = filtered

        if not terms:
            return []

        # 3. Construct FTS5 queries
        and_query = " AND ".join(
            f'"{term}"' for term in terms
        )

        or_query = " OR ".join(
            f'"{term}"' for term in terms
        )

        # 4. Choose candidate count
        candidate_k = max(k * 20, 100) if diversify else k

        # 5. SQL BM25 retrieval
        sql = """
            SELECT
                f.chunk_id,
                f.paper_id,
                p.title,
                f.text,
                bm25(chunks_fts, 0, 0, 2, 1, 1) AS score
            FROM chunks_fts AS f
            JOIN papers AS p
                ON f.paper_id = p.paper_id
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """

        # 6. Execute AND query first (if enabled)
        if and_first:
            rows = self.conn.execute(
                sql, (and_query, candidate_k)
            ).fetchall()

            # Count distinct papers
            unique_papers = {
                row["paper_id"] for row in rows
            }

            # Fall back to OR when not enough papers match
            if len(unique_papers) < k:
                rows = self.conn.execute(
                    sql, (or_query, candidate_k)
                ).fetchall()

        else:
            rows = self.conn.execute(
                sql, (or_query, candidate_k)
            ).fetchall()

        # 7. Return original chunk-level retrieval
        if not diversify:
            return [dict(row) for row in rows[:k]]

        # 8. Paper-level diversification
        results = []
        seen_papers = set()

        for row in rows:
            paper_id = row["paper_id"]

            if paper_id in seen_papers:
                continue

            seen_papers.add(paper_id)
            results.append(dict(row))

            if len(results) >= k:
                break

        return results


if __name__ == "__main__":
    retriever = Retriever("corpus/corpus.db")

    question = "How are conversational agents evaluated?"

    results = retriever.retrieve(question, k=5)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   Paper ID: {result['paper_id']}")
        print(f"   BM25 Score: {result['score']:.4f}")
        print(f"   Text: {result['text'][:200]}...")
