from google import genai
from google.genai import types
import json
from pydantic import BaseModel, Field

class EvaluationResult(BaseModel):
    tone_score: int = Field(description="Score from 1 to 5 for the tone (professional and polite).")
    helpfulness_score: int = Field(description="Score from 1 to 5 for how helpful the reply is in addressing the incoming email.")
    conciseness_score: int = Field(description="Score from 1 to 5 for how concise and to-the-point the reply is.")
    reasoning: str = Field(description="A brief explanation for the given scores.")
    overall_score: float = Field(description="The average of the three scores.")

def evaluate_reply(client: genai.Client, incoming: str, generated_reply: str, reference_reply: str, model_name: str = "gemini-2.5-flash") -> dict:
    """
    Evaluates a generated reply using an LLM-as-a-judge approach.
    """
    prompt = f"""
    You are an expert customer support QA manager. Your job is to evaluate a generated email reply.
    
    Incoming Customer Email:
    {incoming}
    
    Reference (Ideal) Reply:
    {reference_reply}
    
    Generated Reply to Evaluate:
    {generated_reply}
    
    Evaluate the generated reply on three axes (1-5 scale):
    1. Tone: Is it professional, empathetic, and polite?
    2. Helpfulness: Does it accurately address the customer's issue (using the reference as a ground truth for company policy/action)?
    3. Conciseness: Is it direct and without unnecessary fluff?
    
    Provide the scores and a brief reasoning.
    """
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EvaluationResult
        )
    )
    
    return json.loads(response.text)
