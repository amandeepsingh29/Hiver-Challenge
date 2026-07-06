import os
import json
import numpy as np
from google import genai
from typing import List, Dict, Tuple

# Assumes client is created in main.py

def get_embedding(client: genai.Client, text: str) -> np.ndarray:
    """Helper function to get text embedding."""
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents=text
    )
    return np.array(response.embeddings[0].values)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def generate_reply(client: genai.Client, incoming_email: str, dataset: List[Dict], model_name: str = "gemini-2.5-flash") -> Tuple[str, List[Dict]]:
    """
    Generates a suggested reply for an incoming email, grounded in past examples via RAG.
    Returns a tuple of (generated_reply, retrieved_examples).
    """
    # 1. RAG Retrieval Phase
    incoming_vec = get_embedding(client, incoming_email)
    
    # Calculate similarity for all items in the dataset (excluding exact matches to simulate real-world)
    similarities = []
    for item in dataset:
        if item['incoming'] == incoming_email:
            continue
        
        # In a production system, these embeddings would be pre-computed and stored in a Vector DB.
        # Here we compute on the fly for simplicity.
        item_vec = get_embedding(client, item['incoming'])
        sim = cosine_similarity(incoming_vec, item_vec)
        similarities.append((sim, item))
    
    # Sort by highest similarity and take top 2
    similarities.sort(key=lambda x: x[0], reverse=True)
    retrieved_examples = [item for sim, item in similarities[:2]]
    
    # 2. Generation Phase
    prompt = "You are an AI customer support assistant for Hiver. Your task is to draft a polite, helpful, and concise reply to an incoming customer email.\n\n"
    
    if retrieved_examples:
        prompt += "Here are the most relevant past customer emails and our approved replies to use as guidance:\n\n"
        for ex in retrieved_examples:
            prompt += f"--- Relevant Past Example ---\nIncoming: {ex['incoming']}\nReply: {ex['reference_reply']}\n\n"
            
    prompt += f"--- Now draft a reply for the following new incoming email ---\nIncoming: {incoming_email}\nReply:\n"
    
    # Call the model
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    
    return response.text.strip(), retrieved_examples
