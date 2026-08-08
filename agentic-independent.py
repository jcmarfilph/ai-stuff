import time
import math
import sys
from typing import List, Dict, Any

# =====================================================================
# 1. THE FOUNDATIONAL LLM BUILT FROM SCRATCH (Micro-Transformer Blocks)
# =====================================================================

class Softmax:
    @staticmethod
    def forward(x: List[float]) -> List[float]:
        max_x = max(x) if x else 0.0
        exp_x = [math.exp(i - max_x) for i in x]
        sum_exp = sum(exp_x)
        return [i / (sum_exp + 1e-9) for i in exp_x]

class SelfAttentionBlock:
    """A self-contained matrix multiplication and attention mechanism."""
    def __init__(self, d_model: int):
        self.d_model = d_model
        # Deterministic weights for scratch transformer layer projections
        self.W_q = [[0.15 if i == j else 0.02 for j in range(d_model)] for i in range(d_model)]
        self.W_k = [[0.12 if i == j else 0.01 for j in range(d_model)] for i in range(d_model)]
        self.W_v = [[0.20 if i == j else 0.04 for j in range(d_model)] for i in range(d_model)]

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

class CustomScratchLLM:
    """A language model using custom text evaluation logic to prevent misrouting."""
    def __init__(self):
        self.d_model = 8
        self.attention = SelfAttentionBlock(self.d_model)
        
        # Explicit intent classification keywords
        self.intents = {
            "stocks": ["stock", "price", "market", "aapl", "tsla", "nvda", "msft", "shares", "ticker"],
            "news": ["news", "event", "breaking", "bulletin", "devcon", "happenings", "current"],
            "weather": ["weather", "temperature", "rain", "forecast", "cloudy", "degrees", "sky", "hot", "cold"]
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

    def generate_routing_logits(self, prompt: str) -> Dict[str, float]:
        embeddings = self._tokenize_to_embedding(prompt)
        transformed = self.attention.process_sequence(embeddings)
        
        # Calculate mean context pool vector from attention blocks
        seq_len = len(transformed)
        context_vector = [0.0] * self.d_model
        for vec in transformed:
            for i in range(self.d_model):
                context_vector[i] += vec[i] / seq_len
                
        # Parse strings explicitly to calculate math weights on exact intent
        words = prompt.lower().replace("?", "").replace(".", "").replace(",", "").split()
        
        # Establish dynamic routing bases based on exact word matches
        stock_hits = sum(2.0 for w in words if w in self.intents["stocks"])
        news_hits = sum(2.0 for w in words if w in self.intents["news"])
        wx_hits = sum(2.0 for w in words if w in self.intents["weather"])
        
        # Track whether any keywords matched at all
        has_matches = (stock_hits > 0 or news_hits > 0 or wx_hits > 0)
        
        # Combine hidden context embedding vectors with our dynamic hits
        stock_score = sum(context_vector) * 0.05 + stock_hits
        news_score = sum(context_vector) * 0.05 + news_hits
        wx_score = sum(context_vector) * 0.05 + wx_hits
        
        p_stock, p_news, p_wx = Softmax.forward([stock_score, news_score, wx_score])
        
        # If absolutely nothing matched the vocabulary, force low scores to trigger fallback
        if not has_matches:
            return {"stocks": 0.0, "news": 0.0, "weather": 0.0}
            
        return {"stocks": p_stock, "news": p_news, "weather": p_wx}

# =====================================================================
# 2. INDEPENDENT AGENT ENVIRONMENT (Strictly Isolated File Execution)
# =====================================================================

class DataEnvironment:
    """Simulates real system databases via locally maintained plain text files."""
    @staticmethod
    def setup_mock_files():
        with open("stocks.txt", "w") as f:
            f.write("AAPL:175.50\nTSLA:180.20\nNVDA:875.12\nMSFT:420.30")
        with open("news.txt", "w") as f:
            f.write("BREAKING: Tech sector index hits record high.\nEVENT: Annual DevCon starting tonight.")
        with open("weather.txt", "w") as f:
            f.write("CITY: New York\nTEMP: 72F\nCOND: Partly Cloudy\nRAIN: 10%")

    @staticmethod
    def read_file(filename: str) -> str:
        try:
            with open(filename, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return f"Error: {filename} database could not be loaded."

class IndependentAgent:
    """Autonomous core executing cognitive iterations entirely on the local system."""
    def __init__(self):
        self.llm = CustomScratchLLM()
        self.history: List[Dict[str, Any]] = []

    def execute_plan(self, user_request: str) -> str:
        # 1. Thought Cycle (Forward Pass Routing via keyword-strengthened attention)
        routing_signals = self.llm.generate_routing_logits(user_request)
        
        # 2. Hard Routing Filter: Determine the absolute highest probability option
        chosen_intent = max(routing_signals, key=routing_signals.get)
        highest_score = routing_signals[chosen_intent]
        
        # Fallback Check: If confidence is zero due to no keyword matching, return target string
        if highest_score == 0.0:
            return "I do not know the answer, please use google for that."
        
        # 3. Isolated Tool Execution Phase (Only reads from one single target file path)
        if chosen_intent == "stocks":
            raw_data = DataEnvironment.read_file("stocks.txt")
            parsed_output = f"📊 [Single Answer: Stocks Only]\n{raw_data}"
        elif chosen_intent == "news":
            raw_data = DataEnvironment.read_file("news.txt")
            parsed_output = f"📰 [Single Answer: News Only]\n{raw_data}"
        else:
            raw_data = DataEnvironment.read_file("weather.txt")
            parsed_output = f"🌤️ [Single Answer: Weather Only]\n{raw_data}"
            
        # 4. Reflexive Logging Update
        self.history.append({
            "query": user_request,
            "routing": chosen_intent,
            "timestamp": time.time()
        })
        
        return parsed_output

# =====================================================================
# 3. INTERACTIVE REPL RUNNER LOOP
# =====================================================================

if __name__ == "__main__":
    DataEnvironment.setup_mock_files()
    agent = IndependentAgent()
    
    print("=====================================================================")
    print("       AGENT CORE ACTIVE WITH UNRELATED QUERY FALLBACK DETECTOR      ")
    print("       Type your query below. Type '/exit' or Ctrl+C to close.       ")
    print("=====================================================================")
    
    try:
        while True:
            user_input = input("\n💡 Ask Agent -> ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == "/exit":
                print("\nShutting down Agent engine gracefully. Goodbye!")
                break
                
            # Evaluates context and routes to either an isolated tool or the google fallback response
            result = agent.execute_plan(user_input)
            print(result)
            print("-" * 69)
            
    except KeyboardInterrupt:
        print("\n\n[Interrupt Detected] Session closed via Ctrl+C. Goodbye!")
        sys.exit(0)
