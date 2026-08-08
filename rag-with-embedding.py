import sys
import os
import json
import numpy as np
from google import genai
from openai import OpenAI
from enum import Enum
from colorama import init, Fore, Back, Style
from fastembed import TextEmbedding

init(autoreset=True)

class ASSISTANTS(Enum):
    GEMINI = 1
    OPEN_AI = 2

AI_ASSISTANT = ASSISTANTS.OPEN_AI
EMBEDDINGS_FILE = "knowledge_base_fastembed.json"
CACHE_FILE = "semantic_cache.json"

# Semantic cache match configuration
CACHE_THRESHOLD = 0.95  # 95% similarity required to trigger a cache hit

print(Fore.CYAN + "Initializing local FastEmbed engine...")
embedding_model = TextEmbedding() 

match AI_ASSISTANT:
    case ASSISTANTS.GEMINI:
        # Initialize the client (automatically uses GEMINI_API_KEY environment variable
        client = genai.Client(api_key="AQ................")
    case ASSISTANTS.OPEN_AI:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-v1..............",
            default_headers={
                "HTTP-Referer": "http://localhost:3000", # Optional but recommended by OpenRouter
                "X-Title": "Local Barebone RAG App"
            }
        )

KNOWLEDGE_BASE = [
    "Company name is called Karen and Ken LLC",
    "The secret project code name for the new electric vehicle is 'Project Zephyr'.",
    "Company policy states that all employees get 25 days of annual paid leave or PTO, 12 federal holidays and if female, will get additional 10.",
    "The office espresso machine requires a deep cleaning cycle every Friday at 4 PM. Chocolate is available too.",
    "The finance team expects Q3 budget submissions by August 15th. Company performance last year was 150%.",
    "Company provides unlimited sick leave."
]

def get_embedding(text: str) -> list:
    """Generates a text embedding using local FastEmbed."""
    embeddings_generator = embedding_model.embed([text])
    vector = next(embeddings_generator)
    return vector.tolist()

def load_or_create_embeddings(documents: list) -> dict:
    """Loads knowledge base embeddings from local JSON."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r") as f:
            return json.load(f)
            
    print(Fore.CYAN + "Generating local embeddings for Knowledge Base...")
    embeddings_store = {doc: get_embedding(doc) for doc in documents}
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings_store, f, indent=2)
    return embeddings_store

def load_cache() -> list:
    """Loads the history of queries, their embeddings, and answers."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return []

def save_to_cache(query: str, query_vector: list, no_rag_ans: str, rag_ans: str, context: str):
    """Appends a new query, its vector, and responses to the cache file."""
    cache = load_cache()
    cache.append({
        "query": query,
        "embedding": query_vector,
        "context_matched": context,
        "no_rag_response": no_rag_ans,
        "rag_response": rag_ans
    })
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def cosine_similarity(v1: list, v2: list) -> float:
    """Calculates the cosine similarity between two vectors."""
    arr1, arr2 = np.array(v1), np.array(v2)
    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))

def check_semantic_cache(query_vector: list, cache: list) -> dict:
    """Checks if a similar question was asked before."""
    best_hit = None
    highest_score = -1.0
    
    for item in cache:
        score = cosine_similarity(query_vector, item["embedding"])
        if score > highest_score:
            highest_score = score
            best_hit = item
            
    if highest_score >= CACHE_THRESHOLD:
        return best_hit
    return None

def retrieve_context_semantic(query_vector: list, embeddings_store: dict, threshold: float = 0.3) -> str:
    """Finds the document with the highest semantic similarity using pre-calculated query vector."""
    best_match = ""
    highest_score = -1.0
    
    for doc, doc_vector in embeddings_store.items():
        score = cosine_similarity(query_vector, doc_vector)
        if score > highest_score:
            highest_score = score
            best_match = doc
            
    return best_match if highest_score >= threshold else "No relevant context found."

def generate_answer_no_rag(query: str) -> str:
    """Generates an answer via LLM with no local context."""
    prompt = f"Answer the following request. CRITICAL RULE: Your entire response must be exactly 2 to 3 sentences long. Do not exceed this limit.\n\nRequest: {query}"
    match AI_ASSISTANT:
        case ASSISTANTS.GEMINI:
            response = client.models.generate_content(model='gemini-3.6-flash', contents=prompt)
            return response.text
        case ASSISTANTS.OPEN_AI:
            response = client.chat.completions.create(
                model="google/gemma-4-26b-a4b-it:free",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

def generate_answer_with_rag(query: str, context: str) -> str:
    """Augments prompt with retrieved corporate context."""
    prompt = f"""
    You are an intelligent corporate assistant. 
    1. First, look at the "Retrieved Local Context" below to get specific company-specific data.
    2. Then, combine that context with your own general knowledge, professional reasoning, and explanation skills to give a thorough, comprehensive, and helpful answer to the user.
    3. If the local context does not mention the topic, rely entirely on your own general knowledge to answer, but politely note that you couldn't find corporate records on it.
    4. Make the response 2-3 sentences only please.

    Retrieved Local Context:
    {context}
    User Question:
    {query}
    Comprehensive Answer:
    """
    match AI_ASSISTANT:
        case ASSISTANTS.GEMINI:
            response = client.models.generate_content(model='gemini-3.6-flash', contents=prompt)
            return response.text
        case ASSISTANTS.OPEN_AI:
            response = client.chat.completions.create(
                model="google/gemma-4-26b-a4b-it:free",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content


# --- Interactive Loop ---
if __name__ == "__main__":
    print("\n=== Interactive Corporate Knowledge Semantic RAG Pipeline ===")
    print("Ask questions about company policy, projects, or office operations.")
    print("Type '/quit' or press Ctrl+C to exit.\n")
    
    loaded_vectors = load_or_create_embeddings(KNOWLEDGE_BASE)
    
    try:
        while True:
            user_question = input(Style.BRIGHT + Fore.YELLOW + "You: ").strip()
            
            if not user_question:
                continue
                
            if user_question.lower() == '/quit':
                print("\nGoodbye!")
                break
            
            # 1. Compute current query vector once
            current_query_vector = get_embedding(user_question)
            
            # 2. Check the Semantic Cache
            cached_pipeline = load_cache()
            cache_hit = check_semantic_cache(current_query_vector, cached_pipeline)
            
            if cache_hit:
                print(Fore.MAGENTA + f"\n[CACHE HIT] Found a matching previous query: '{cache_hit['query']}'")
                print(Style.BRIGHT + Fore.RED + f"Chatbot Context Match: {cache_hit['context_matched']}\n")
                print(Style.BRIGHT + Fore.GREEN + f"Chatbot AI response with no RAG:\n{cache_hit['no_rag_response']}\n")
                print(Style.BRIGHT + Fore.BLUE + f"Chatbot AI response with RAG:\n{cache_hit['rag_response']}\n")
                continue
            
            # 3. Cache Miss - Proceed with normal retrieval and generation
            print(Fore.LIGHTBLACK_EX + "\n[CACHE MISS] Fetching fresh answers from live LLM pipeline...")
            retrieved_data = retrieve_context_semantic(current_query_vector, loaded_vectors)

            print(Style.BRIGHT + Fore.RED + f"Chatbot Context Match: {retrieved_data}\n")
            
            ai_response = generate_answer_no_rag(user_question)
            print(Style.BRIGHT + Fore.GREEN + f"Chatbot AI response with no RAG:\n{ai_response}\n")            
            
            rag_response = generate_answer_with_rag(user_question, retrieved_data)
            print(Style.BRIGHT + Fore.BLUE + f"Chatbot AI response with RAG:\n{rag_response}\n")            
            
            # 4. Save newly generated generation vectors to file
            save_to_cache(user_question, current_query_vector, ai_response, rag_response, retrieved_data)
            
    except KeyboardInterrupt:
        print("\n\nSession closed via Ctrl+C. Goodbye!")
        sys.exit(0)
