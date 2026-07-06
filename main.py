import os
import json
from google import genai
from dotenv import load_dotenv
from generator import generate_reply
from evaluator import evaluate_reply

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        print("Please create a .env file and add GEMINI_API_KEY=your_key")
        return
        
    client = genai.Client(api_key=api_key)
    
    # 1. Load the dataset
    print("Loading dataset...", flush=True)
    try:
        with open("dataset.json", "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Error: dataset.json not found.")
        return
        
    print(f"Loaded {len(dataset)} examples.", flush=True)
    
    results = []
    total_score = 0.0
    
    # 2 & 3. Generate responses and Evaluate them
    for i, item in enumerate(dataset):
        print(f"\n--- Processing Email {i+1}/{len(dataset)} ---", flush=True)
        print(f"Category: {item['category']}", flush=True)
        print(f"Incoming: {item['incoming']}", flush=True)
        
        # Generate
        print("Generating reply...", flush=True)
        try:
            generated_reply = generate_reply(client, item['incoming'], dataset)
            print(f"Generated Reply:\n{generated_reply}\n", flush=True)
            
            # Evaluate
            print("Evaluating reply...", flush=True)
            eval_result = evaluate_reply(client, item['incoming'], generated_reply, item['reference_reply'])
            print(f"Scores - Tone: {eval_result['tone_score']}, Helpfulness: {eval_result['helpfulness_score']}, Conciseness: {eval_result['conciseness_score']}", flush=True)
            print(f"Overall Score: {eval_result['overall_score']}", flush=True)
            print(f"Reasoning: {eval_result['reasoning']}", flush=True)
            
            results.append({
                "id": item["id"],
                "incoming": item["incoming"],
                "generated_reply": generated_reply,
                "evaluation": eval_result
            })
            
            total_score += eval_result['overall_score']
        except Exception as e:
            print(f"Error during processing: {e}", flush=True)
            return
        
    # Final Reporting
    average_overall_score = total_score / len(dataset)
    print("\n==================================================", flush=True)
    print("FINAL EVALUATION REPORT", flush=True)
    print("==================================================", flush=True)
    print(f"Total Emails Processed: {len(dataset)}", flush=True)
    print(f"System Average Overall Score: {average_overall_score:.2f} / 5.00", flush=True)
    print("==================================================", flush=True)
    
    # Save results to a file for review
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to evaluation_results.json", flush=True)

if __name__ == "__main__":
    main()
