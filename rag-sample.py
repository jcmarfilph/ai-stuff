import sys
from google import genai
from openai import OpenAI
from enum import Enum
from colorama import init, Fore, Back, Style


class ASSISTANTS(Enum):
    GEMINI = 1
    OPEN_AI = 2

AI_ASSISTANT = ASSISTANTS.OPEN_AI


match AI_ASSISTANT:
    case ASSISTANTS.GEMINI:
        # Initialize the client (automatically uses GEMINI_API_KEY environment variable
        client = genai.Client(api_key="AQ....................")
    case ASSISTANTS.OPEN_AI:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-v1-........................",
            default_headers={
                "HTTP-Referer": "http://localhost:3000", # Optional but recommended by OpenRouter
                "X-Title": "Local Barebone RAG App"
            }
        )

# Our Knowledge Base (The Data Store)
KNOWLEDGE_BASE = [
    "The secret project code name for the new electric vehicle is 'Project Zephyr'.",
    "Company policy states that all employees get 25 days of annual paid leave, 12 federal holidays and if female, will get additional 10.",
    "The office espresso machine requires a deep cleaning cycle every Friday at 4 PM.",
    "The finance team expects Q3 budget submissions by August 15th.",
    "Company provides unlimited sick leave."
]

def retrieve_context(user_query: str, documents: list) -> str:
    """
    A barebones keyword retriever.
    Finds the document containing the most overlapping words with the query.
    """
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
    """
    Just generates an answer via LLM and no context
    """
    
    prompt = f"Answer the following request. CRITICAL RULE: Your entire response must be exactly 2 to 3 sentences long. Do not exceed this limit.\n\nRequest: {query}"
            
    match AI_ASSISTANT:
        case ASSISTANTS.GEMINI:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
            )
            return response.text
        case ASSISTANTS.OPEN_AI:
            response = client.chat.completions.create(
                model="google/gemma-4-26b-a4b-it:free",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content

def generate_answer_with_rag(query: str, context: str) -> str:
    """
    Augments the prompt, instructing LLM to blend its own knowledge 
    with the retrieved corporate context.
    """
    
    # Instructs the AI to merge local data with external intelligence
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
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
            )
            return response.text
        case ASSISTANTS.OPEN_AI:
            # Pick any OpenRouter model slug (e.g., Llama 3.3 70B, Qwen 2.5, etc.)
            response = client.chat.completions.create(
                model="google/gemma-4-26b-a4b-it:free",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
                 
            return response.choices[0].message.content


# --- Interactive Loop ---
if __name__ == "__main__":
    print("=== Interactive Corporate Knowledge RAG Pipeline ===")
    print("Ask questions about company policy, projects, or office operations.")
    print("Type '/quit' or press Ctrl+C to exit.\n")
    
    try:
        while True:
            # Capture input from user
            user_question = input(Style.BRIGHT + Fore.YELLOW + "You: ").strip()
            
            # Skip empty inputs
            if not user_question:
                continue
                
            # Check for quit command
            if user_question.lower() == '/quit':
                print("\nGoodbye!")
                break
            
            # Retrieve context
            retrieved_data = retrieve_context(user_question, KNOWLEDGE_BASE)

            # Display results
            print(Style.BRIGHT + Fore.RED + f"Chatbot: No AI assistance: {retrieved_data}\n")
            ai_response = generate_answer_no_rag(user_question)
            print(Style.BRIGHT + Fore.GREEN + f"Chatbot: AI response with no RAG: {ai_response}\n")            
            rag_response = generate_answer_with_rag(user_question, retrieved_data)
            print(Style.BRIGHT + Fore.BLUE + f"Chatbot: AI response with RAG: {rag_response}\n")            
            
    except KeyboardInterrupt:
        # Catch Ctrl+C cleanly without throwing raw Python error traces
        print("\n\nSession closed via Ctrl+C. Goodbye!")
        sys.exit(0)
