import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.example_index import build_query_search_text, infer_domain, infer_intent


STOPWORDS = {
    "the", "a", "an", "all", "me", "show", "give", "list", "of", "for",
    "in", "on", "to", "and", "or", "with", "by", "is", "are", "was",
    "were", "this", "that", "have", "has", "do", "does", "using",
}


def extract_keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9_./:-]+", text.lower())
    return {token for token in tokens if len(token) >= 3 and token not in STOPWORDS}


class HybridExampleRetriever:
    """
    Hybrid Retrieval = TF-IDF + Dense Embeddings + Structured Signals.

    TF-IDF catches exact technical matches:
    - dim_campaign
    - /data/foundation/catalog/batches
    - mergePolicies

    Dense embeddings catch semantic matches:
    - successful batches ≈ batches with status success
    - audience evaluation jobs ≈ segment jobs

    Structured signals stabilize the final score:
    - same domain
    - same intent
    - keyword overlap
    """

    def __init__(
        self,
        example_index_path: str | Path = "data/example_index.json",
        retrieval_dir: str | Path = "data/retrieval_index",
        use_dense: bool = True,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.example_index_path = Path(example_index_path)
        self.retrieval_dir = Path(retrieval_dir)
        self.use_dense = use_dense

        self.weights = weights or {
            "tfidf": 0.30,
            "dense": 0.50,
            "structured": 0.20,
        }

        self._validate_paths()
        self._load_examples()
        self._load_config()
        self._load_tfidf()
        self._load_dense_if_enabled()

    def _validate_paths(self) -> None:
        if not self.example_index_path.exists():
            raise FileNotFoundError(
                f"Example index not found: {self.example_index_path}. "
                f"Run: python -m src.example_index"
            )

        if not self.retrieval_dir.exists():
            raise FileNotFoundError(
                f"Retrieval index folder not found: {self.retrieval_dir}. "
                f"Run: python -m src.example_index"
            )

    def _load_examples(self) -> None:
        with self.example_index_path.open("r", encoding="utf-8") as f:
            self.examples: List[Dict[str, Any]] = json.load(f)

    def _load_config(self) -> None:
        config_path = self.retrieval_dir / "retrieval_config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Retrieval config not found: {config_path}. "
                f"Run: python -m src.example_index"
            )

        with config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

    def _load_tfidf(self) -> None:
        self.vectorizer = joblib.load(self.retrieval_dir / "tfidf_vectorizer.joblib")
        self.tfidf_matrix = joblib.load(self.retrieval_dir / "tfidf_matrix.joblib")

    def _load_dense_if_enabled(self) -> None:
        self.dense_enabled = (
            self.use_dense
            and self.config.get("dense_enabled", False)
            and (self.retrieval_dir / "dense_embeddings.npy").exists()
        )

        self.embedding_model = None
        self.dense_embeddings = None

        if not self.dense_enabled:
            return

        from sentence_transformers import SentenceTransformer

        model_name = self.config["embedding_model"]

        print(f"Loading dense retrieval model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)

        self.dense_embeddings = np.load(self.retrieval_dir / "dense_embeddings.npy")

    def _tfidf_scores(self, query_search_text: str) -> np.ndarray:
        query_vector = self.vectorizer.transform([query_search_text])
        scores = (self.tfidf_matrix @ query_vector.T).toarray().ravel()
        return scores.astype(np.float32)

    def _dense_scores(self, query_search_text: str) -> np.ndarray:
        if not self.dense_enabled or self.embedding_model is None:
            return np.zeros(len(self.examples), dtype=np.float32)

        query_embedding = self.embedding_model.encode(
            [query_search_text],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(query_embedding, dtype=np.float32)[0]

        scores = self.dense_embeddings @ query_embedding

        # Cosine similarity may theoretically be negative.
        # Clamp to [0, 1] to make scoring easier.
        scores = np.clip(scores, 0.0, 1.0)

        return scores.astype(np.float32)

    def _structured_score(
        self,
        query: str,
        query_intent: str,
        query_domain: str,
        example: Dict[str, Any],
    ) -> float:
        score = 0.0

        example_domain = example.get("domain", "general")
        example_intent = example.get("intent", "lookup")

        if query_domain != "general" and query_domain == example_domain:
            score += 0.45

        if query_intent == example_intent:
            score += 0.25

        query_keywords = extract_keywords(query)
        example_keywords = extract_keywords(example.get("query", ""))

        if query_keywords:
            overlap = len(query_keywords & example_keywords) / max(len(query_keywords), 1)
            score += min(overlap, 1.0) * 0.30

        return min(score, 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        query_search_text = build_query_search_text(query)
        query_intent = infer_intent(query)
        query_domain = infer_domain(query, api_endpoints=[], sql_tables=[])

        tfidf_scores = self._tfidf_scores(query_search_text)
        dense_scores = self._dense_scores(query_search_text)

        structured_scores = np.array(
            [
                self._structured_score(
                    query=query,
                    query_intent=query_intent,
                    query_domain=query_domain,
                    example=example,
                )
                for example in self.examples
            ],
            dtype=np.float32,
        )

        combined_scores = (
            self.weights["tfidf"] * tfidf_scores
            + self.weights["dense"] * dense_scores
            + self.weights["structured"] * structured_scores
        )

        ranked_indices = np.argsort(combined_scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []

        for rank, idx in enumerate(ranked_indices, start=1):
            idx = int(idx)
            example = self.examples[idx]

            results.append(
                {
                    "rank": rank,
                    "combined_score": float(combined_scores[idx]),
                    "tfidf_score": float(tfidf_scores[idx]),
                    "dense_score": float(dense_scores[idx]),
                    "structured_score": float(structured_scores[idx]),

                    "query": example["query"],
                    "route": example["route"],
                    "intent": example["intent"],
                    "domain": example["domain"],

                    "sql_tables": example["sql_tables"],
                    "api_endpoints": example["api_endpoints"],
                    "api_params": example["api_params"],

                    "gold_sql": example["gold_sql"],
                    "gold_api": example["gold_api"],
                }
            )

        return results

    def route_votes(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        votes: Dict[str, float] = {}

        for result in results:
            route = result["route"]
            score = result["combined_score"]
            votes[route] = votes.get(route, 0.0) + score

        total = sum(votes.values())

        if total <= 0:
            return votes

        return {
            route: round(score / total, 4)
            for route, score in sorted(votes.items(), key=lambda item: item[1], reverse=True)
        }

    def domain_votes(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        votes: Dict[str, float] = {}

        for result in results:
            domain = result["domain"]
            score = result["combined_score"]
            votes[domain] = votes.get(domain, 0.0) + score

        total = sum(votes.values())

        if total <= 0:
            return votes

        return {
            domain: round(score / total, 4)
            for domain, score in sorted(votes.items(), key=lambda item: item[1], reverse=True)
        }


if __name__ == "__main__":
    retriever = HybridExampleRetriever()

    test_queries = [
        "How many successful batches are there?",
        "List all journeys",
        "Show me all tags",
        "Which segment evaluation jobs are queued?",
        "Show datasets that use the same schema",
        "Give me recently modified destinations",
    ]

    for query in test_queries:
        print("\n" + "=" * 90)
        print("Query:", query)

        results = retriever.search(query, top_k=3)

        for item in results:
            print(
                f"[{item['rank']}] combined={item['combined_score']:.4f} "
                f"tfidf={item['tfidf_score']:.4f} "
                f"dense={item['dense_score']:.4f} "
                f"structured={item['structured_score']:.4f}"
            )
            print(
                f"    route={item['route']} "
                f"domain={item['domain']} "
                f"intent={item['intent']}"
            )
            print("    similar query:", item["query"])
            print("    endpoints:", item["api_endpoints"])
            print("    tables:", item["sql_tables"])

        print("Route votes:", retriever.route_votes(results))
        print("Domain votes:", retriever.domain_votes(results))