import os
import json
import numpy as np
import faiss
from google import genai
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple

# Load FAISS index and mapping at module level for O(1) fast startup
try:
    index = faiss.read_index("dataset.faiss")
    with open("faiss_mapping.json", "r") as f:
        faiss_mapping = json.load(f)
except Exception as e:
    print(f"Warning: Could not load FAISS index. Run build_index.py first. Error: {e}")
    index, faiss_mapping = None, None

class GeneratorOutput(BaseModel):
    intent: str = Field(description="The primary intent of the customer (e.g., 'refund_request', 'bug_report').")
    requires_human: bool = Field(description="True if the request involves legal threats, account deletion, or requires human empathy. False otherwise.")
    suggested_reply: str = Field(description="The polite, helpful, and concise email draft. If requires_human is True, draft an escalation message.")

def get_embedding(client: genai.Client, text: str) -> np.ndarray:
    """Helper function to get text embedding."""
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents=text
    )
    return np.array(response.embeddings[0].values, dtype=np.float32)

def generate_reply(client: genai.Client, incoming_email: str, dataset: List[Dict], model_name: str = "gemini-2.5-flash") -> Tuple[GeneratorOutput, List[Dict]]:
    """
    Generates a structured reply using FAISS vector search for RAG.
    Implements a similarity threshold to gracefully fall back to zero-shot if no relevant past tickets exist.
    """
    retrieved_examples = []
    
    if index is not None and faiss_mapping is not None:
        # 1. FAISS RAG Retrieval (O(1) fast lookup)
        incoming_vec = get_embedding(client, incoming_email)
        faiss.normalize_L2(np.array([incoming_vec]))
        
        # Search top 3 (in case the query itself is in the dataset, we skip it)
        distances, indices = index.search(np.array([incoming_vec]), 3)
        
        SIMILARITY_THRESHOLD = 0.65
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            
            # Threshold Check - Defensive RAG
            if dist < SIMILARITY_THRESHOLD:
                continue
                
            # Retrieve from dataset via mapping
            item_id = faiss_mapping.get(str(idx))
            if item_id:
                # Find item in dataset
                item = next((x for x in dataset if x['id'] == item_id), None)
                if item and item['incoming'] != incoming_email:
                    retrieved_examples.append(item)
                if len(retrieved_examples) == 2:
                    break
    
    # 2. Generation Phase with HITL Structured Output and System Instruction
    system_instruction = "You are an elite, highly empathetic AI customer support assistant for Hiver. Draft a polite, helpful, and concise reply. Base your answers firmly on company policy if provided."
    
    prompt = ""
    
    if retrieved_examples:
        prompt += "--- RELEVANT PAST EXAMPLES (Knowledge Base) ---\n"
        for ex in retrieved_examples:
            prompt += f"Incoming: {ex['incoming']}\nApproved Reply: {ex['reference_reply']}\n\n"
            
    prompt += f"--- NEW INCOMING EMAIL ---\n{incoming_email}\n"
    
    # Call the model enforcing structured output and using the system prompt
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=GeneratorOutput,
            temperature=0.3
        )
    )
    
    # Parse structured output
    try:
        output = GeneratorOutput.model_validate_json(response.text)
    except Exception as e:
        # Fallback if parsing fails (rare with strict schema)
        output = GeneratorOutput(intent="unknown", requires_human=True, suggested_reply=f"Error parsing response: {e}")
        
    return output, retrieved_examples
