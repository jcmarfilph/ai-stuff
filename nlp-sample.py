import os
import pandas as pd
from google import genai
from openai import OpenAI
from enum import Enum
from colorama import init, Fore, Back, Style


class ASSISTANTS(Enum):
    GEMINI = 1
    OPEN_AI = 2

AI_ASSISTANT = ASSISTANTS.OPEN_AI


# 1. Load the CSV data into Pandas DataFrames
customers_df = pd.read_csv("customers.csv")
orders_df = pd.read_csv("orders.csv")


# 2. Initialize the Gemini client
match AI_ASSISTANT:
    case ASSISTANTS.GEMINI:
        # Initialize the client (automatically uses GEMINI_API_KEY environment variable
        client = genai.Client(api_key="AQ.................")
    case ASSISTANTS.OPEN_AI:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-v1.....................7",
            default_headers={
                "HTTP-Referer": "http://localhost:3000", # Optional but recommended by OpenRouter
                "X-Title": "Local Barebone NLP App"
            }
        )


# 3. Define the NLP system prompt with schema context
sys_instruction = f"""
You are a Python data analytics assistant. Your job is to translate human natural language questions into executable Pandas code.
You have access to two DataFrames:

'customers_df' with columns: {list(customers_df.columns)}
'orders_df' with columns: {list(orders_df.columns)}

Rules:
1. ONLY return the valid executable Python code block.
2. Do not explain the code. Do not wrap it in markdown block tags like ```python.
3. Assume the variables 'customers_df' and 'orders_df' are already loaded in memory.
4. The last line of your code must evaluate to the final answer or DataFrame so it prints out.
"""

def query_csv_with_nlp(user_question: str):
    print(f"\n🙋 Question: {user_question}")
    
    match AI_ASSISTANT:
        case ASSISTANTS.GEMINI:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=user_question,
                config={
                    "system_instruction": sys_instruction,
                    "temperature": 0.0 # Low temperature makes generation deterministic and precise
                }
            )

            pandas_code = response.text.strip()
  
        case ASSISTANTS.OPEN_AI:
            response = client.chat.completions.create(
                model="google/gemma-4-26b-a4b-it:free",
                messages=[
                    {"role": "system", "content": sys_instruction},
                    {"role": "user", "content": user_question}
                ],
                temperature=0.0 # Keeps deterministic and precise code generation
            )

            pandas_code = response.choices[0].message.content.strip()  
    
    print(f"🤖 Generated Pandas Code:\n{pandas_code}\n")
    
    # 5. Safely execute the generated code in the local environment
    try:
        # Using local execution environment context
        local_vars = {'customers_df': customers_df, 'orders_df': orders_df}
        
        # Execute all lines except the last one, evaluate the last line to grab the result
        lines = pandas_code.split('\n')
        if len(lines) > 1:
            exec('\n'.join(lines[:-1]), {}, local_vars)
        
        result = eval(lines[-1], {}, local_vars)
        
        print("📊 Output Result:")
        print(result)
    except Exception as e:
        print(f"❌ Failed to run generated code: {e}")

# --- Example Query Scenarios ---

# Example 1: Filtering and Selection
query_csv_with_nlp("Show me all customers who live in New York")

# Example 2: Joins and Aggregations
query_csv_with_nlp("What is the total amount spent by Alice Smith?")

# Example 3: Complex Join query
query_csv_with_nlp("List the order products and names of customers who spent more than $100")
