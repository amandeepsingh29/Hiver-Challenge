import os
import json
import asyncio
from google import genai
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.genai.errors import APIError

from generator import generate_reply
from evaluator import evaluate_reply

# Configuration
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Please set GEMINI_API_KEY in your .env file.")
    exit(1)

client = genai.Client(api_key=api_key)

# We use a semaphore to limit concurrent requests on the free tier
# Max 2 concurrent tasks to respect rate limits while still being much faster than serial
CONCURRENCY_LIMIT = 2
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(APIError)
)
def generate_with_retry(incoming: str, dataset: list):
    return generate_reply(client, incoming, dataset)

@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(APIError)
)
def evaluate_with_retry(incoming: str, generated: str, reference: str):
    return evaluate_reply(client, incoming, generated, reference)

async def process_email(item, dataset):
    async with semaphore:
        print(f"Processing Email ID: {item['id']}...", flush=True)
        try:
            # Run the synchronous generator in a thread
            output, retrieved = await asyncio.to_thread(generate_with_retry, item['incoming'], dataset)
            
            # Run the synchronous evaluator in a thread
            eval_result = await asyncio.to_thread(
                evaluate_with_retry, 
                item['incoming'], 
                output.suggested_reply, 
                item['reference_reply']
            )
            
            print(f"✅ Finished Email ID: {item['id']} | HITL Required: {output.requires_human} | Score: {eval_result['overall_score']}", flush=True)
            
            return {
                "id": item["id"],
                "incoming": item["incoming"],
                "generator_output": output.model_dump(),
                "evaluation": eval_result
            }
        except Exception as e:
            print(f"❌ Failed Email ID: {item['id']}: {e}", flush=True)
            return None

async def main():
    print("Loading dataset...")
    try:
        with open("dataset.json", "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("dataset.json not found.")
        return
        
    print(f"Loaded {len(dataset)} examples. Starting async batch processing...")
    
    # Process all emails concurrently with bounded concurrency
    tasks = [process_email(item, dataset) for item in dataset]
    results = await asyncio.gather(*tasks)
    
    # Filter out failures
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        total_score = sum(r['evaluation']['overall_score'] for r in valid_results)
        avg_score = total_score / len(valid_results)
        print(f"\n--- BATCH COMPLETE ---")
        print(f"Processed {len(valid_results)}/{len(dataset)} emails successfully.")
        print(f"System Average Score: {avg_score:.2f} / 5.0")
        
        # Save results
        with open("evaluation_results.json", "w") as f:
            json.dump(valid_results, f, indent=2)
        print("Detailed results saved to evaluation_results.json")
    else:
        print("No valid results were produced.")

if __name__ == "__main__":
    asyncio.run(main())
