from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# 1. Initialize the local model (no API key or internet required)
# Make sure the model name matches what you downloaded via Ollama
model = ChatOllama(model="llama3", temperature=0.7)

# 2. Create your dynamic prompt template (same as before)
prompt_template = PromptTemplate.from_template(
    "What are three good names for a {business_type} that sells {product}?"
)

# 3. Combine the prompt and model into a chain using LCEL
chain = prompt_template | model

# 4. Run the chain locally
response = chain.invoke({"business_type": "bakery", "product": "sourdough bread"})

# 5. Print the local AI's response
print(response.content)
