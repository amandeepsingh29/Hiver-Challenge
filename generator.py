import os
import json
import google.generativeai as genai
from typing import List, Dict

# Assumes genai.configure(api_key=...) has been called in main.py

def generate_reply(incoming_email: str, dataset: List[Dict], model_name: str = "gemini-1.5-flash") -> str:
    """
    Generates a suggested reply for an incoming email, grounded in past examples (Few-Shot).
    """
    model = genai.GenerativeModel(model_name)
    
    # We use a few-shot prompting approach. Let's take up to 3 random examples from the dataset
    # (excluding the one we might be testing if we were doing strict leave-one-out, 
    # but for simplicity we'll just use the first 3 that don't match the incoming exactly)
    few_shot_examples = [ex for ex in dataset if ex['incoming'] != incoming_email][:3]
    
    prompt = "You are an AI customer support assistant for Hiver. Your task is to draft a polite, helpful, and concise reply to an incoming customer email.\n\n"
    
    if few_shot_examples:
        prompt += "Here are some examples of past customer emails and our approved replies:\n\n"
        for ex in few_shot_examples:
            prompt += f"--- Example ---\nIncoming: {ex['incoming']}\nReply: {ex['reference_reply']}\n\n"
            
    prompt += f"--- Now draft a reply for the following new incoming email ---\nIncoming: {incoming_email}\nReply:\n"
    
    # Call the model
    response = model.generate_content(prompt)
    
    return response.text.strip()
