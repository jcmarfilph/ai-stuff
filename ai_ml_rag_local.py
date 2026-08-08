import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# ==========================================
# 1. DATA AND VOCABULARY SETUP
# ==========================================
documents = [
    "ACME daily meal budget is 50 dollars.",
    "Remote work is allowed for 30 days.",
    "Project Nova deploys on October 15."
]

# Mini training dataset pairing user queries with the index of the correct context document
training_queries = [
    {"query": "What is the daily meal budget?", "correct_doc_idx": 0},
    {"query": "How much can I spend on food?", "correct_doc_idx": 0},
    {"query": "How long can I work remotely?", "correct_doc_idx": 1},
    {"query": "What is the remote work policy?", "correct_doc_idx": 1},
    {"query": "When is Project Nova deploying?", "correct_doc_idx": 2},
    {"query": "Who is deploying on October 15?", "correct_doc_idx": 2}
]

# Vocabulary setup
chars = sorted(list(set("".join(documents) + " \n?abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:,;!=-_")))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# ==========================================
# 2. ARCHITECTURES FROM SCRATCH
# ==========================================
class ScratchEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.char_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, text):
        indices = torch.tensor([char_to_idx.get(c, 0) for c in text], dtype=torch.long)
        embedded = self.char_embeddings(indices)
        mean_pooled = embedded.mean(dim=0)
        return torch.tanh(self.fc1(mean_pooled))

class ScratchRNNGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(vocab_size, hidden_size)
        self.rnn_cell = nn.GRUCell(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_idx, hidden):
        embedded = self.encoder(input_idx)
        hidden = self.rnn_cell(embedded, hidden)
        output = self.decoder(hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# Instantiate models
embed_model = ScratchEmbeddingModel(vocab_size, embed_dim=32)
generator_model = ScratchRNNGenerator(vocab_size, hidden_size=128)

# ==========================================
# 3. THE TRAINING LOOPS (TRAINING MODE)
# ==========================================
def train_models(epochs=150):
    print(">>> Starting Internal Model Training Mode...")
    
    # Optimizers
    embed_optimizer = optim.Adam(embed_model.parameters(), lr=0.01)
    gen_optimizer = optim.Adam(generator_model.parameters(), lr=0.005)
    
    # Loss Criteria
    criterion_gen = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        # --- PART A: Train the Embedding Network (Contrastive Ranking Loss) ---
        embed_model.train()
        embed_loss_total = 0
        embed_optimizer.zero_grad()  # Reset gradients once at the start of the epoch
        
        # Pre-compute document embeddings for this step
        doc_vectors = torch.stack([embed_model(doc) for doc in documents])
        
        epoch_embed_loss = 0
        for q_data in training_queries:
            q_vec = embed_model(q_data["query"])
            correct_idx = q_data["correct_doc_idx"]
            
            # Choose a random wrong document as a negative example
            wrong_indices = [i for i in range(len(documents)) if i != correct_idx]
            wrong_idx = random.choice(wrong_indices)
            
            # Calculate cosine similarities
            pos_sim = F.cosine_similarity(q_vec.unsqueeze(0), doc_vectors[correct_idx].unsqueeze(0))
            neg_sim = F.cosine_similarity(q_vec.unsqueeze(0), doc_vectors[wrong_idx].unsqueeze(0))
            
            # Margin loss calculation (No immediate backward here to avoid in-place runtime errors)
            loss_embed = F.relu(0.3 - (pos_sim - neg_sim)) 
            epoch_embed_loss += loss_embed

        # Step weights exactly once after accumulating the entire batch loss
        if epoch_embed_loss.item() > 0:
            epoch_embed_loss.backward()  
            embed_optimizer.step()
            embed_loss_total = epoch_embed_loss.item()

        # --- PART B: Train the Text Generator (Autoregressive Token Prediction) ---
        generator_model.train()
        gen_loss_total = 0
        
        for doc in documents:
            gen_optimizer.zero_grad()
            hidden = generator_model.init_hidden()
            loss_seq = 0
            
            # Train the network to recreate its internal knowledge sequence text
            for i in range(len(doc) - 1):
                input_idx = torch.tensor([char_to_idx[doc[i]]])
                target_idx = torch.tensor([char_to_idx[doc[i+1]]])
                
                output, hidden = generator_model(input_idx, hidden)
                loss_seq += criterion_gen(output, target_idx)
                
            loss_seq.backward()
            gen_optimizer.step()
            gen_loss_total += loss_seq.item() / len(doc)

        if epoch % 30 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Embed Loss: {embed_loss_total:.4f} | Gen Loss: {gen_loss_total:.4f}")

    print(">>> Training complete! Models switched to evaluation mode.\n")
    embed_model.eval()
    generator_model.eval()

# ==========================================
# 4. INFERENCE PIPELINE
# ==========================================
def retrieve(query, doc_vectors):
    with torch.no_grad():
        query_vec = embed_model(query)
    similarities = F.cosine_similarity(query_vec.unsqueeze(0), doc_vectors)
    return torch.argmax(similarities).item()

def generate_response(prompt, max_len=45):
    hidden = generator_model.init_hidden()
    generated_text = prompt
    
    with torch.no_grad():
        # Prime internal states
        for char in prompt[:-1]:
            input_idx = torch.tensor([char_to_idx.get(char, 0)])
            _, hidden = generator_model(input_idx, hidden)
            
        input_char = prompt[-1]
        for _ in range(max_len):
            input_idx = torch.tensor([char_to_idx.get(input_char, 0)])
            output, hidden = generator_model(input_idx, hidden)
            
            # Greedy prediction decoding
            predicted_idx = torch.argmax(output, dim=1).item()
            next_char = idx_to_char[predicted_idx]
            generated_text += next_char
            input_char = next_char
            
            # Break if network generates a newline character or stop signal
            if next_char == "\n":
                break
    return generated_text

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Execute the training mode function
    train_models(epochs=100)
    
    # Re-calculate index matrix vectors using newly optimized weights
    with torch.no_grad():
        final_doc_vectors = torch.stack([embed_model(doc) for doc in documents])
    
    # Test Question
    user_query = "How much can I spend on food?"
    print(f"[User Query]: {user_query}")
    
    # 1. RAG Action: Retrieve context index
    best_doc_idx = retrieve(user_query, final_doc_vectors)
    retrieved_fact = documents[best_doc_idx]
    print(f"[Retrieved Context]: {retrieved_fact}")
    
    # 2. RAG Action: Seed the model generator directly with factual context text
    constructed_prompt = f"Fact: {retrieved_fact} Answer: "
    
    system_output = generate_response(constructed_prompt, max_len=50)
    print(f"[Trained System Output]:\n{system_output}")
