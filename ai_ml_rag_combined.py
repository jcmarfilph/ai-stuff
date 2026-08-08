import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline

# 1. KNOWLEDGE BASE (The internal data sources)
documents = [
    "ACME Corp's internal travel policy states that employees can claim up to $50 per day for meals without receipts.",
    "The official remote work framework allows staff to work from another country for a maximum of 30 calendar days per year.",
    "Project Nova is scheduled for deployment on October 15, 2026, under the supervision of Dr. Sarah Jenkins.",
    "HR requires all annual performance self-evaluations to be submitted via the TalentPortal by December 5th annually."
]

# 2. CONFIGURATION & MODELS (Using lightweight, open-source models)
# Using a popular, tiny embedding model for vector search
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Using a lightweight, fast text-generation model
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading internal AI/ML models...")
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
generator = pipeline("text-generation", model=LLM_MODEL_NAME, max_new_tokens=150)

# 3. EMBEDDING PIPELINE (Converts text into internal vector representations)
def compute_embeddings(texts):
    inputs = embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = embed_model(**inputs)
    # Perform mean pooling to get fixed-sized sentence vectors
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

print("Indexing internal documents...")
document_embeddings = compute_embeddings(documents)

# 4. RETRIEVAL PIPELINE (Vector Search via Cosine Similarity)
def retrieve_relevant_context(query, top_k=1):
    query_embedding = compute_embeddings([query])
    # Calculate cosine similarities between query and documents
    dot_product = np.dot(document_embeddings, query_embedding.T).squeeze()
    doc_norms = np.linalg.norm(document_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    similarities = dot_product / (doc_norms * query_norm)
    
    # Get the index of the most similar document
    best_match_idx = np.argmax(similarities)
    return documents[best_match_idx]

# 5. GENERATION PIPELINE (Augmenting the prompt with retrieved context)
def answer_internal_query(user_query):
    # Step A: Retrieve facts
    context = retrieve_relevant_context(user_query, top_k=1)
    print(f"\n[Retrieved Context]: {context}")
    
    # Step B: Construct the augmented prompt
    prompt = (
        f"You are an internal company assistant. Use the following piece of context to answer the question. "
        f"If you do not know the answer based on the context, say you don't know.\n\n"
        f"Context: {context}\n\n"
        f"Question: {user_query}\n\n"
        f"Answer:"
    )
    
    # Step C: Generate response using local LLM
    response = generator(prompt)
    generated_text = response[0]['generated_text']
    
    # Clean up the output to only show the generated answer
    answer = generated_text.split("Answer:")[-1].strip()
    return answer

# 6. EXECUTION EXAMPLE
if __name__ == "__main__":
    query = "What is the daily budget limit for meals when I travel for work?"
    print(f"\nUser Question: {query}")
    
    final_answer = answer_internal_query(query)
    print(f"\n[System Output]: {final_answer}")
