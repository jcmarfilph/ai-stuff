import os
import json
from enum import Enum
import ollama
from openai import OpenAI
from google import genai
from google.genai import types

# =====================================================================
# 1. PROVIDER SELECTION SETUP
# =====================================================================

class ASSISTANTS(Enum):
    GEMINI = 1
    OPEN_AI = 2  # OpenRouter
    OLLAMA = 3

# Choose your provider here
AI_ASSISTANT = ASSISTANTS.OLLAMA

# Configure model targets per provider
MODEL_MAP = {
    ASSISTANTS.GEMINI: "gemini-2.5-flash",
    ASSISTANTS.OPEN_AI: "meta-llama/llama-3.1-8b-instruct:free",  # Example OpenRouter free model
    ASSISTANTS.OLLAMA: "llama3.1"  # Changed from llama3 to support tools
}

SELECTED_MODEL = MODEL_MAP[AI_ASSISTANT]

# =====================================================================
# 2. DEFINE YOUR TOOLS (Functions with type hints and docstrings)
# =====================================================================

def get_current_weather(location: str) -> str:
    """
    Get the current weather for a given city location.
    
    Args:
        location: The name of the city, e.g., 'London, UK' or 'New York'
    """
    print(f"   [Tool Executing] Fetching weather data for: {location}")
    loc = location.lower()
    if "new york" in loc:
        return "It is currently 72°F and sunny in New York."
    return f"The weather in {location} is 68°F with a light breeze."

AVAILABLE_TOOLS = {
    'get_current_weather': get_current_weather
}

# =====================================================================
# 3. AGENT loop DEFINITION
# =====================================================================

def run_agent(user_prompt: str):
    print(f"\n[Provider: {AI_ASSISTANT.name}] User Request: '{user_prompt}'")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use your weather tool if asked about the weather."},
        {"role": "user", "content": user_prompt}
    ]
    
    # -----------------------------------------------------------------
    # PATH A: GOOGLE GEMINI EXECUTION
    # -----------------------------------------------------------------
    if AI_ASSISTANT == ASSISTANTS.GEMINI:
        # Note: Do not hardcode API keys in production; use os.environ instead
        client = genai.Client(api_key="AQ.................................")
        
        # Convert our system message format to Gemini's expected layout
        gemini_messages = [{"role": m["role"], "parts": [m["content"]]} for m in messages if m["role"] != "system"]
        system_instruction = messages[0]["content"]

        response = client.models.generate_content(
            model=SELECTED_MODEL,
            contents=gemini_messages,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[get_current_weather]
            )
        )
        
        # Handle Gemini tool calls
        if response.function_calls:
            print("🤖 Gemini decided to use a tool!")
            for call in response.function_calls:
                if call.name in AVAILABLE_TOOLS:
                    # Extract arguments as a dictionary
                    args = dict(call.args)
                    tool_result = AVAILABLE_TOOLS[call.name](**args)
                    print(f"   [Tool Result] {tool_result}")
                    
                    # Gemini handles follow-ups by appending the call and the response to history
                    gemini_messages.append(response.candidates[0].content)
                    gemini_messages.append(types.Content(
                        role="tool",
                        parts=[types.Part.from_function_response(name=call.name, response={"result": tool_result})]
                    ))
                    
                    final_response = client.models.generate_content(
                        model=SELECTED_MODEL,
                        contents=gemini_messages,
                        config=types.GenerateContentConfig(system_instruction=system_instruction)
                    )
                    print(f"\nFinal AI Answer:\n{final_response.text}")
        else:
            print(f"\nFinal AI Answer:\n{response.text}")

    # -----------------------------------------------------------------
    # PATH B: OPENROUTER (OPENAI SDK) EXECUTION
    # -----------------------------------------------------------------
    elif AI_ASSISTANT == ASSISTANTS.OPEN_AI:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-v1.........................",
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Local Barebone RAG App"
            }
        )
        
        # Format tools into OpenAI JSON schemas
        openai_tools = [{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a given city location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"}
                    },
                    "required": ["location"]
                }
            }
        }]

        response = client.chat.completions.create(
            model=SELECTED_MODEL,
            messages=messages,
            tools=openai_tools
        )
        
        assistant_message = response.choices[0].message
        if assistant_message.tool_calls:
            print("🤖 OpenRouter decided to use a tool!")
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name in AVAILABLE_TOOLS:
                    tool_result = AVAILABLE_TOOLS[function_name](**function_args)
                    print(f"   [Tool Result] {tool_result}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result
                    })
            
            final_response = client.chat.completions.create(model=SELECTED_MODEL, messages=messages)
            print(f"\nFinal AI Answer:\n{final_response.choices[0].message.content}")
        else:
            print(f"\nFinal AI Answer:\n{assistant_message.content}")

    # -----------------------------------------------------------------
    # PATH C: OLLAMA LOCAL EXECUTION
    # -----------------------------------------------------------------
    elif AI_ASSISTANT == ASSISTANTS.OLLAMA:
        response = ollama.chat(
            model=SELECTED_MODEL,
            messages=messages,
            tools=[get_current_weather]
        )
        
        if response.message.tool_calls:
            print("🤖 Ollama decided to use a tool!")
            messages.append(response.message)
            
            for tool_call in response.message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments
                
                if function_name in AVAILABLE_TOOLS:
                    tool_result = AVAILABLE_TOOLS[function_name](**function_args)
                    print(f"   [Tool Result] {tool_result}")
                    
                    messages.append({
                        'role': 'tool',
                        'name': function_name,
                        'content': tool_result
                    })
                    
            final_response = ollama.chat(model=SELECTED_MODEL, messages=messages)
            print(f"\nFinal AI Answer:\n{final_response.message.content}")
        else:
            print(f"\nFinal AI Answer:\n{response.message.content}")

# =====================================================================
# 4. EXECUTION
# =====================================================================
if __name__ == "__main__":
    run_agent("What should I wear outside in New York right now?")
