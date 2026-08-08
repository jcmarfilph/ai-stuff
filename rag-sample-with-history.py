import sys
import json
import os
from google import genai
from openai import OpenAI
from enum import Enum
from colorama import init, Fore, Back, Style

class ASSISTANTS(Enum):
    GEMINI = 1
    OPEN_AI = 2

AI_ASSISTANT = ASSISTANTS.OPEN_AI

# Initialize Client Router
match AI_ASSISTANT:
    case ASSISTANTS.GEMINI:
        # Initialize the client (automatically uses GEMINI_API_KEY environment variable
        client = genai.Client(api_key="AQ........................")
    case ASSISTANTS.OPEN_AI:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-v1................................",
            default_headers={
                "HTTP-Referer": "http://localhost:3000", # Optional but recommended by OpenRouter
                "X-Title": "Local Barebone RAG App"
            }
        )
        
KNOWLEDGE_BASE = [
    "The secret project code name for the new electric vehicle is 'Project Zephyr'.",
    "Company policy states that all employees get 25 days of annual paid leave, 12 federal holidays and if female, will get additional 10.",
    "The office espresso machine requires a deep cleaning cycle every Friday at 4 PM.",
    "The finance team expects Q3 budget submissions by August 15th.",
    "Company provides unlimited sick leave."
]

# --- Persistent File Configuration ---
CACHE_FILE = "rag_chat_cache.json"

def load_cache() -> dict:
    """Loads the cache from a local JSON file if it exists and is valid."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f: #
                return json.load(f) #
        except (json.JSONDecodeError, IOError): #
            # If file is empty or corrupted, return a clean empty cache
            print(Style.BRIGHT + Fore.RED + "Warning: Cache file corrupted. Resetting local cache.")
            return {}
    return {}

def save_cache(cache_data: dict):
    """Saves the current in-memory cache back to the local JSON file."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f: #
            json.dump(cache_data, f, indent=2, ensure_ascii=False) #
    except IOError as e:
        print(Style.BRIGHT + Fore.RED + f"Error saving cache to disk: {e}")

# Initialize the cache by reading from disk
CHAT_CACHE = load_cache()

def check_history_cache(user_query: str, threshold: float = 0.7) -> str | None:
    """Checks the local cache for past user queries with high keyword similarity."""
    query_words = set(user_query.lower().split())
    if not query_words:
        return None
        
    best_cached_response = None
    highest_match_ratio = 0.0

    for cached_query, cached_response in CHAT_CACHE.items():
        cached_words = set(cached_query.lower().split())
        if not cached_words:
            continue
            
        overlap_count = len(query_words.intersection(cached_words))
        match_ratio = overlap_count / len(query_words)
        
        if match_ratio > highest_match_ratio:
            highest_match_ratio = match_ratio
            best_cached_response = cached_response

    if highest_match_ratio >= threshold:
        return best_cached_response
        
    return None

def retrieve_context(user_query: str, documents: list) -> str:
    query_words = set(user_query.lower().split())
    best_match = ""
    highest_score = 0
    for doc in documents:
        doc_words = set(doc.lower().split())
        score = len(query_words.intersection(doc_words))
        if score > highest_score:
            highest_score = score
            best_match = doc
    return best_match if highest_score > 0 else "No relevant context found."

def generate_answer_no_rag(query: str) -> str:
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
    prompt = f"""
    You are an intelligent corporate assistant.
    1. First, look at the "Retrieved Local Context" below to get specific company-specific data.
    2. Then, combine that context with your own general knowledge, professional reasoning, and explanation skills to give a thorough, comprehensive, and helpful answer to the user.
    3. If the local context does not mention the topic, rely entirely on your own general knowledge to answer, but politely note that you couldn't find corporate records on it.
    4. Make the response 2-3 sentences only please.
    
    Retrieved Local Context: {context}
    User Question: {query}
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
    init(autoreset=True)
    print("=== Interactive Corporate Knowledge RAG Pipeline ===")
    print(f"Loaded {len(CHAT_CACHE)} cached responses from '{CACHE_FILE}'.")
    print("Ask questions about company policy, projects, or office operations.")
    print("Type '/quit' or press Ctrl+C to exit.\n")
    
    try:
        while True:
            user_question = input(Style.BRIGHT + Fore.YELLOW + "You: ").strip()
            
            if not user_question:
                continue
                
            if user_question.lower() == '/quit':
                print("\nGoodbye!")
                break
            
            # Step 1: Check local cache first
            cached_answer = check_history_cache(user_question, threshold=0.7)
            
            if cached_answer:
                print(Style.BRIGHT + Fore.MAGENTA + f"Chatbot [LOCAL CACHE HIT]: {cached_answer}\n")
                continue 
            
            # Step 2: RAG Pipeline Fallback
            retrieved_data = retrieve_context(user_question, KNOWLEDGE_BASE)
            print(Style.BRIGHT + Fore.RED + f"Chatbot: No AI assistance: {retrieved_data}\n")
            
            ai_response = generate_answer_no_rag(user_question)
            print(Style.BRIGHT + Fore.GREEN + f"Chatbot: AI response with no RAG: {ai_response}\n")
            
            rag_response = generate_answer_with_rag(user_question, retrieved_data)
            print(Style.BRIGHT + Fore.BLUE + f"Chatbot: AI response with RAG: {rag_response}\n")
            
            # Step 3: Save newly generated answer to memory AND update disk storage
            CHAT_CACHE[user_question] = rag_response
            save_cache(CHAT_CACHE) #
            
    except KeyboardInterrupt:
        print("\n\nSession closed via Ctrl+C. Goodbye!")
        sys.exit(0)
