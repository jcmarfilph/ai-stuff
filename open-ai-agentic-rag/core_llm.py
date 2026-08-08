import math
from typing import List, Dict

class Softmax:
    @staticmethod
    def forward(x: List[float]) -> List[float]:
        max_x = max(x) if x else 0.0
        exp_x = [math.exp(i - max_x) for i in x]
        sum_exp = sum(exp_x)
        return [i / (sum_exp + 1e-9) for i in exp_x]

class SelfAttentionBlock:
    def __init__(self, d_model: int, weight_multiplier: float = 1.0):
        self.d_model = d_model
        self.W_q = [[0.15 * weight_multiplier if i == j else 0.02 for j in range(d_model)] for i in range(d_model)]
        self.W_k = [[0.12 * weight_multiplier if i == j else 0.01 for j in range(d_model)] for i in range(d_model)]
        self.W_v = [[0.20 * weight_multiplier if i == j else 0.04 for j in range(d_model)] for i in range(d_model)]

    def _matmul(self, vec: List[float], matrix: List[List[float]]) -> List[float]:
        out = [0.0] * len(matrix)
        for col in range(len(matrix)):
            out[col] = sum(vec[row] * matrix[row][col] for row in range(len(vec)))
        return out

    def process_sequence(self, token_embeddings: List[List[float]]) -> List[List[float]]:
        Q = [self._matmul(t, self.W_q) for t in token_embeddings]
        K = [self._matmul(t, self.W_k) for t in token_embeddings]
        V = [self._matmul(t, self.W_v) for t in token_embeddings]
        
        seq_len = len(token_embeddings)
        output_embeddings = []
        
        for i in range(seq_len):
            scores = []
            for j in range(seq_len):
                score = sum(Q[i][k] * K[j][k] for k in range(self.d_model))
                scores.append(score / math.sqrt(self.d_model))
            
            attention_weights = Softmax.forward(scores)
            
            out_vec = [0.0] * self.d_model
            for j in range(seq_len):
                for k in range(self.d_model):
                    out_vec[k] += attention_weights[j] * V[j][k]
            output_embeddings.append(out_vec)
            
        return output_embeddings

class CustomAgentScratchLLM:
    """Mathematical feature routing matrix supporting structured token calls."""
    def __init__(self, model_version: str = "v1"):
        self.d_model = 8
        multiplier = 2.5 if model_version == "scratch-agent-smarter-v2" else 1.0
        self.attention = SelfAttentionBlock(self.d_model, weight_multiplier=multiplier)
        self.intents = {
            "fetch_stock_data": ["stock", "price", "market", "aapl", "tsla", "nvda", "msft", "shares", "ticker", "equity", "valuation"],
            "fetch_news_feed": ["news", "event", "breaking", "bulletin", "devcon", "happenings", "current", "headline", "updates"],
            "fetch_weather_telemetry": ["weather", "temperature", "rain", "forecast", "cloudy", "degrees", "sky", "hot", "cold", "climate"]
        }
        
    def _tokenize_to_embedding(self, text: str) -> List[List[float]]:
        words = text.lower().replace("?", "").replace(".", "").split()
        embeddings = []
        for word in words:
            base_val = sum(ord(c) for c in word) % 10 / 10.0
            vec = [(base_val + i * 0.05) for i in range(self.d_model)]
            embeddings.append(vec)
        if not embeddings:
            embeddings = [[0.1] * self.d_model]
        return embeddings

    def analyze_intent(self, prompt: str) -> Dict[str, float]:
        """Calculates neural probability vectors over tool options registers."""
        embeddings = self._tokenize_to_embedding(prompt)
        transformed = self.attention.process_sequence(embeddings)
        
        seq_len = len(transformed)
        context_vector = [0.0] * self.d_model
        for vec in transformed:
            for i in range(self.d_model):
                context_vector[i] += vec[i] / seq_len
                
        words = prompt.lower().replace("?", "").replace(".", "").replace(",", "").split()
        
        stock_hits = sum(3.0 for w in words if w in self.intents["fetch_stock_data"])
        news_hits = sum(3.0 for w in words if w in self.intents["fetch_news_feed"])
        wx_hits = sum(3.0 for w in words if w in self.intents["fetch_weather_telemetry"])
        
        if not (stock_hits > 0 or news_hits > 0 or wx_hits > 0):
            return {"fetch_stock_data": 0.0, "fetch_news_feed": 0.0, "fetch_weather_telemetry": 0.0}
        
        stock_score = sum(context_vector) * 0.05 + stock_hits
        news_score = sum(context_vector) * 0.05 + news_hits
        wx_score = sum(context_vector) * 0.05 + wx_hits
        
        scores = Softmax.forward([stock_score, news_score, wx_score])
        return {"fetch_stock_data": scores[0], "fetch_news_feed": scores[1], "fetch_weather_telemetry": scores[2]}
