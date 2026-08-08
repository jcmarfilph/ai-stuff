import math
from typing import List, Dict, Tuple

class LocalRAGEngine:
    """Pure Python Retrieval-Augmented Generation indexing and matching engine."""
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().replace("?", "").replace(".", "").replace(",", "").split()

    @classmethod
    def _compute_tf(cls, tokens: List[str]) -> Dict[str, float]:
        tf = {}
        if not tokens:
            return tf
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        for token in tf:
            tf[token] = tf[token] / len(tokens)
        return tf

    @classmethod
    def retrieve_context(cls, query: str, filepath: str = "knowledge_base.txt") -> Tuple[str, float]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except FileNotFoundError:
            return "", 0.0

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            return "", 0.0

        query_tokens = cls._tokenize(query)
        query_tf = cls._compute_tf(query_tokens)

        best_match = ""
        best_score = 0.0

        for paragraph in paragraphs:
            para_tokens = cls._tokenize(paragraph)
            para_tf = cls._compute_tf(para_tokens)

            dot_product = 0.0
            for word in query_tf:
                if word in para_tf:
                    dot_product += query_tf[word] * para_tf[word]

            query_mag = math.sqrt(sum(v**2 for v in query_tf.values())) + 1e-9
            para_mag = math.sqrt(sum(v**2 for v in para_tf.values())) + 1e-9
            similarity = dot_product / (query_mag * para_mag)

            if similarity > best_score:
                best_score = similarity
                best_match = paragraph

        # If similarity is too weak, reject the match by returning empty content
        if best_score < 0.05:
            return "", 0.0

        return best_match, best_score
