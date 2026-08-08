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

# Initialize clients using standard configurations
match AI_ASSISTANT:
    case ASSISTANTS.GEMINI:
        # Initialize the client (automatically uses GEMINI_API_KEY environment variable
        client = genai.Client(api_key="AQ.............")
    case ASSISTANTS.OPEN_AI:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-v1...................",
            default_headers={
                "HTTP-Referer": "http://localhost:3000", # Optional but recommended by OpenRouter
                "X-Title": "Local Barebone RAG App"
            }
        )

# Define your input data
job_title = "Senior Data Analyst"
company_name = "TechCorp"
work_experience = """
- 4 years of experience analyzing large datasets using Python and SQL.
- Built a predictive dashboard that reduced churn by 15%.
- Led a team of 2 junior analysts and automated weekly reporting.
"""

# Craft a clear prompt
prompt = f"""
Write a professional, concise cover letter for the position of {job_title} at {company_name}.
Base the letter on the following work experience:
{work_experience}
Keep it under 300 words and tone it to be enthusiastic yet professional.
"""

# Generate the content using the recommended model
match AI_ASSISTANT:
    case ASSISTANTS.GEMINI:
        response = client.models.generate_content(model='gemini-3.6-flash', contents=prompt)
        print(response.text)
    case ASSISTANTS.OPEN_AI:
        response = client.chat.completions.create(
            model="google/gemma-4-26b-a4b-it:free",
            messages=[{"role": "user", "content": prompt}]
        )
        print(response.choices[0].message.content)
