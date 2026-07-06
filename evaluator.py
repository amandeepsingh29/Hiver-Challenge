from google import genai
from pydantic import BaseModel, Field

class EvaluationResult(BaseModel):
    tone_score: int = Field(description="Score from 1 to 5 on politeness and professionalism.")
    helpfulness_score: int = Field(description="Score from 1 to 5 on whether it solves the user's problem.")
    conciseness_score: int = Field(description="Score from 1 to 5 on being concise without fluff.")
    hallucination_penalty: int = Field(description="Score from 0 to -5. Deduct points if the generated reply invents company policies, urls, or features not present in the reference reply.")
    reasoning: str = Field(description="A brief explanation for the given scores and penalties.")
    overall_score: float = Field(description="The average of the three positive scores plus the hallucination penalty (e.g., if avg is 4 and penalty is -2, overall is 2).")

def evaluate_reply(client: genai.Client, incoming: str, generated_reply: str, reference_reply: str, model_name: str = "gemini-2.5-flash") -> dict:
    """
    Evaluates a generated reply using an LLM-as-a-judge approach, now with strict hallucination checks.
    """
    prompt = f"""
    You are a strict QA evaluator for Hiver customer support.
    Evaluate the following GENERATED REPLY against the IDEAL REFERENCE REPLY.
    
    Customer Email: {incoming}
    Ideal Reference Reply: {reference_reply}
    Generated Reply: {generated_reply}
    
    Score Tone, Helpfulness, and Conciseness on a scale of 1-5.
    Critically, apply a Hallucination Penalty (0 to -5) if the Generated Reply invents any facts, URLs, integrations, or promises that are NOT present in the Ideal Reference Reply.
    Return the scores and your reasoning in JSON format.
    """
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EvaluationResult,
                temperature=0.1
            )
        )
        # Parse the JSON response
        result = EvaluationResult.model_validate_json(response.text)
        return result.model_dump()
    except Exception as e:
        return {
            "tone_score": 0,
            "helpfulness_score": 0,
            "conciseness_score": 0,
            "hallucination_penalty": 0,
            "overall_score": 0.0,
            "reasoning": f"Evaluation failed: {e}"
        }
