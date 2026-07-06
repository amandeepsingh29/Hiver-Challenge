import os
import json
from google import genai
from typing import List, Dict

# Assumes client is created in main.py

def generate_reply(client: genai.Client, incoming_email: str, dataset: List[Dict], model_name: str = "gemini-1.5-flash") -> str:
    """
    Generates a suggested reply for an incoming email, grounded in past examples (Few-Shot).
    """
    # We use a few-shot prompting approach. Let's take up to 3 random examples from the dataset
    few_shot_examples = [ex for ex in dataset if ex['incoming'] != incoming_email][:3]
    
    prompt = "You are an AI customer support assistant for Hiver. Your task is to draft a polite, helpful, and concise reply to an incoming customer email.\n\n"
    
    if few_shot_examples:
        prompt += "Here are some examples of past customer emails and our approved replies:\n\n"
        for ex in few_shot_examples:
            prompt += f"--- Example ---\nIncoming: {ex['incoming']}\nReply: {ex['reference_reply']}\n\n"
            
    prompt += f"--- Now draft a reply for the following new incoming email ---\nIncoming: {incoming_email}\nReply:\n"
    
    # Call the model
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    
    return response.text.strip()
