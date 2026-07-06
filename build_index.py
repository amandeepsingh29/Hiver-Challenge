import os
import json
import time
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API Key not found.")
    exit(1)

client = genai.Client(api_key=api_key)

def get_embedding(client: genai.Client, text: str) -> np.ndarray:
    """Helper function to get text embedding."""
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents=text
    )
    return np.array(response.embeddings[0].values, dtype=np.float32)

print("Loading dataset...")
with open("dataset.json", "r") as f:
    dataset = json.load(f)

embeddings = []
mapping = {}

print(f"Building index for {len(dataset)} items...")
index = None

for i, item in enumerate(dataset):
    print(f"Embedding item {i+1}/{len(dataset)}...")
    vec = get_embedding(client, item['incoming'])
    
    if index is None:
        dim = len(vec)
        print(f"Detected embedding dimension: {dim}")
        index = faiss.IndexFlatIP(dim)
        
    # Normalize for cosine similarity
    faiss.normalize_L2(np.array([vec]))
    embeddings.append(vec)
    
    # Store mapping from faiss ID to item ID
    mapping[str(i)] = item['id']
    
    time.sleep(1) # Be gentle on rate limits even for embeddings

# Add to index
embeddings_array = np.array(embeddings, dtype=np.float32)
faiss.normalize_L2(embeddings_array)
index.add(embeddings_array)

print("Saving FAISS index and mapping...")
faiss.write_index(index, "dataset.faiss")

with open("faiss_mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)

print("Build complete.")
