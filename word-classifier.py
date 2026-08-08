import os
import pickle
import re
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline
import nltk
from nltk.corpus import wordnet

# Download the WordNet database (run once)
nltk.download('wordnet')

# Configuration Constants
EMBEDDING_DIM = 8
TAG_MAPPING = {0: "Noun", 1: "Verb", 2: "Pronoun"}   # <-- Pronoun added
MODEL_FILE = "word_persist_model.pth"
VOCAB_FILE = "word_persist_vocab.pkl"
SUFFIX_FEATURES = ['ing', 'ed', 'ate', 'ize', 'tion', 'ism', 'er', 'ment']

# Pronoun Lexical List
PRONOUNS = {
    "i","me","my","mine","you","your","yours","he","him","his",
    "she","her","hers","it","its","we","us","our","ours",
    "they","them","their","theirs","this","that","these","those"
}

print("[System] Initializing DistilBERT local LLM for synonym generation...")
# The batch_size parameter tells the engine it can safely parallelize inputs
llm_mask_filler = pipeline("fill-mask", model="distilbert-base-uncased", batch_size=16)

# ==========================================
# 1. CORE NETWORK ARCHITECTURE
# ==========================================
class HybridWordClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, fallback_dim):
        super(HybridWordClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim + fallback_dim, 3)
        
    def forward(self, x_idx, x_char_features):
        embedded = self.embedding(x_idx)
        combined = torch.cat((embedded, x_char_features), dim=1)
        return self.fc(combined)

def extract_char_features(word):
    word = word.lower()
    vector = [1.0 if suffix in word else 0.0 for suffix in SUFFIX_FEATURES]
    return torch.tensor(vector, dtype=torch.float32)

def load_system():
    if os.path.exists(MODEL_FILE) and os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, "rb") as f:
            vocab = pickle.load(f)
        model = HybridWordClassifier(len(vocab), EMBEDDING_DIM, len(SUFFIX_FEATURES))
        model.load_state_dict(torch.load(MODEL_FILE))
    else:
        vocab = {"<unk>": 0} 
        model = HybridWordClassifier(len(vocab), EMBEDDING_DIM, len(SUFFIX_FEATURES))
    return model, vocab

def tokenize_text(text):
    clean_text = text.replace('\n', ' ').replace('\r', ' ')
    return re.findall(r'\b\w+\b', clean_text)


def get_wordnet_synonyms(word: str) -> list:
    synonyms = set()
    
    # Loop through synsets (semantic concept groups)
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Clean up multi-word phrases (e.g., 'car_pool' -> 'car pool')
            clean_synonym = lemma.name().replace('_', ' ')
            synonyms.add(clean_synonym)
            
    # Remove the target word itself from the results
    synonyms.discard(word)
    return sorted(list(synonyms))



def get_single_word_synonym(word):
    """Fallback function for isolated single word predictions."""
    word_lower = word.lower()
    generic_prompt = f"The word '{word_lower}' means the same thing as [MASK]."
    results = llm_mask_filler(generic_prompt)
    
    if results and isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
        results = results[0]  # Handle variable array nesting
        
    for res in results:
        syn = res['token_str'].strip()
        if syn.isalpha() and syn.lower() != word_lower and len(syn) > 1:
            return syn.lower()
    return None

# ==========================================
# 2. BATCH INFERENCE PIPELINE
# ==========================================
def predict_sentence(sentence_text):
    """
    Tokenizes text blocks and batches prompts directly into the LLM transformer.
    This architecture utilizes maximum parallel processing speed and removes sequential warnings.
    """
    model, vocab = load_system()
    model.eval()
    words_to_predict = tokenize_text(sentence_text)
    
    print("\n" + "="*115)
    print(f"Running Multi-Line Sentence Inference (GPU Batch Mode Optimized):")
    print(f"\"\"\"\n{sentence_text.strip()}\n\"\"\"")
    print("="*115)
    
    # Step A: Pre-build masked strings for the entire text block upfront
    masked_prompts = []
    for word in words_to_predict:
        masked_sentence = re.sub(rf'\b{re.escape(word)}\b', '[MASK]', sentence_text, flags=re.IGNORECASE)
        if '[MASK]' in masked_sentence:
            masked_prompts.append(masked_sentence)
        else:
            masked_prompts.append(f"The word '{word.lower()}' is the synonym of [MASK].")

    # Step B: Feed everything to the LLM pipeline at the same time to trigger optimized batch math
    batch_results = llm_mask_filler(masked_prompts)
    
    # Step C: Evaluate classifications using parallel outputs
    with torch.no_grad():
        for i, word in enumerate(words_to_predict):
            word_lower = word.lower()
            method = "Memory Vector (Direct Match)"
            label = "Classification"
            

            if word_lower in PRONOUNS:
                method = "Direct Pronoun Match"
                pred_class = 2
            else:
                # 1. Direct Lookup Pass
                if word_lower in vocab:
                    word_idx = torch.tensor([vocab[word_lower]], dtype=torch.long)
                else:
                    # 2. Extract context-aware synonym from pre-calculated batch collection matrix
                    synonym = None
                    word_results = batch_results[i]
                    
                    # Dynamic normalization wrapper for nested pipeline dictionary lists
                    if word_results and isinstance(word_results, list) and len(word_results) > 0 and isinstance(word_results[0], list):
                        word_results = word_results[0]
                        
                    for res in word_results:
                        syn = res['token_str'].strip()
                        if syn.isalpha() and syn.lower() != word_lower and len(syn) > 1:

                            #synonym = get_single_word_synonym(syn)
                            #for syno in get_wordnet_synonyms(syn):
                            #    if (syno in vocab):
                            #        synonym = syno
                            #        break
                            
                            synonym = syn.lower()
                            break
                    
                    #print(f"Synonym for word {word} is {synonym}")

                    # Synonym Proxy Lookup Pass
                    if synonym and synonym in vocab:
                        word_idx = torch.tensor([vocab[synonym]], dtype=torch.long)
                        method = f"LLM Proxy Vector (via synonym: '{synonym}')"
                        label = "Classification"
                    else:
                        # 3. Structural Analysis Trait Fallback
                        word_idx = torch.tensor([vocab.get("<unk>", 0)], dtype=torch.long)
                        method = f"Character Analysis (No matches for word or synonym '{synonym}')"
                        label = "Prediction"

                char_features = extract_char_features(word_lower).unsqueeze(0)
                prediction = model(word_idx, char_features)
                pred_class = torch.argmax(prediction, dim=1).item()
                
            print(f"Token: '{word:<15}' -> {label:<15}: {TAG_MAPPING[pred_class]:<10} | Strategy: {method:<50}")

# ==========================================
# 3. INTERFACE WRAPPERS & TRAINING
# ==========================================
def predict_word(word):
    """Evaluates an isolated token using local synonym proxy mapping checks."""
    model, vocab = load_system()
    model.eval()
    word_lower = word.lower()

    if word_lower in PRONOUNS:
        method = "Direct Pronoun Match"
        pred_class = 2
        label = "Classification"

    with torch.no_grad():
        if word_lower in vocab:
            word_idx = torch.tensor([vocab[word_lower]], dtype=torch.long)
            method = "Memory Vector (Direct Match)"
            label = "Classification"
        else:
            synonym = get_single_word_synonym(word)
            #synonyms = get_wordnet_synonyms(word)
            #synonym =  synonyms[0]
            #for syn in synonyms:
            #    if (syn in vocab):
            #        synonym = syn
            #        break

            print(f"Synonym for word {word} is {synonym}")

            if synonym and synonym in vocab:
                word_idx = torch.tensor([vocab[synonym]], dtype=torch.long)
                method = f"LLM Proxy Vector (via synonym: '{synonym}')"
                label = "Classification"
            else:
                word_idx = torch.tensor([vocab.get("<unk>", 0)], dtype=torch.long)
                method = f"Character Analysis (No matches for word or synonym '{synonym}')"
                label = "Prediction"
                
        char_features = extract_char_features(word_lower).unsqueeze(0)
        prediction = model(word_idx, char_features)
        pred_class = torch.argmax(prediction, dim=1).item()
        
    print(f"Word: '{word:<15}' -> {label:<15}: {TAG_MAPPING[pred_class]:<10} | Strategy: {method:<50}")
    return TAG_MAPPING[pred_class]


def train_on_new_words(new_samples):
    """Grows model storage allocations and tunes parameters for novel items."""
    model, vocab = load_system()
    truly_new = [(w.lower(), t) for w, t in new_samples if w.lower() not in vocab]
    
    if len(truly_new) == 0:
        return
        
    for word, _ in truly_new:
        vocab[word] = len(vocab)
        
    old_embeddings = model.embedding.weight.data
    model.embedding = nn.Embedding(len(vocab), EMBEDDING_DIM)
    with torch.no_grad():
        if len(old_embeddings) > 1:
            model.embedding.weight.data[:len(old_embeddings)] = old_embeddings
            
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    idx_list = torch.tensor([vocab[w] for w, _ in truly_new], dtype=torch.long)
    char_list = torch.stack([extract_char_features(w) for w, _ in truly_new])
    labels = torch.tensor([tag for _, tag in truly_new], dtype=torch.long)
    
    model.train()
    for epoch in range(40):
        optimizer.zero_grad()
        loss = criterion(model(idx_list, char_list), labels)
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), MODEL_FILE)
    with open(VOCAB_FILE, "wb") as f:
        pickle.dump(vocab, f)


# ==========================================
# 4. RUNTIME CONFIRMATION LOOP
# ==========================================
if __name__ == "__main__":
    #for f in [MODEL_FILE, VOCAB_FILE]:
    #    if os.path.exists(f): os.remove(f)

    print("--- Stage 1: Training Core Vocabulary Base ---")
    baseline_data = [
        ("apple", 0), ("device", 0), ("instructor", 0),
        ("run", 1), ("operate", 1), ("program", 1),
        ("start", 1),
        ("he", 2), ("she", 2), ("they", 2), ("it", 2), ("our", 2)
    ]
    train_on_new_words(baseline_data)
    print("[System] Core vocabulary trained successfully.")
    
    print("\n--- Stage 2: Standalone Isolated Prediction Metrics ---")
    predict_word("apple")       # Exact match -> Stored Memory 
    predict_word("mandarin")
    predict_word("master")     # Resolves to synonym "instructor" via LLM proxy mapping
    predict_word("instrument")
    predict_word("she")
    predict_word("they")
    
    print("\n--- Stage 3: Running Efficient Batch Text Operations ---")
    text_block = """
    The instructor will execute the program.
    We will try and run to get an apple and orange.
    """
    # Excecuting this will not generate any Hugging Face sequential loop warning markers
    predict_sentence(text_block)

    text_block = """
    The mentor will run the show so try to catch up if you can or else they will take over and beat us.
    """
    # Excecuting this will not generate any Hugging Face sequential loop warning markers
    predict_sentence(text_block)    
