import os
import json
import google.generativeai as genai
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
        
    genai.configure(api_key=api_key)
    
    # 1. Load the dataset
    print("Loading dataset...")
    try:
        with open("dataset.json", "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Error: dataset.json not found.")
        return
        
    print(f"Loaded {len(dataset)} examples.")
    
    results = []
    total_score = 0.0
    
    # 2 & 3. Generate responses and Evaluate them
    for i, item in enumerate(dataset):
        print(f"\n--- Processing Email {i+1}/{len(dataset)} ---")
        print(f"Category: {item['category']}")
        print(f"Incoming: {item['incoming']}")
        
        # Generate
        print("Generating reply...")
        generated_reply = generate_reply(item['incoming'], dataset)
        print(f"Generated Reply:\n{generated_reply}\n")
        
        # Evaluate
        print("Evaluating reply...")
        eval_result = evaluate_reply(item['incoming'], generated_reply, item['reference_reply'])
        print(f"Scores - Tone: {eval_result['tone_score']}, Helpfulness: {eval_result['helpfulness_score']}, Conciseness: {eval_result['conciseness_score']}")
        print(f"Overall Score: {eval_result['overall_score']}")
        print(f"Reasoning: {eval_result['reasoning']}")
        
        results.append({
            "id": item["id"],
            "incoming": item["incoming"],
            "generated_reply": generated_reply,
            "evaluation": eval_result
        })
        
        total_score += eval_result['overall_score']
        
    # Final Reporting
    average_overall_score = total_score / len(dataset)
    print("\n==================================================")
    print("FINAL EVALUATION REPORT")
    print("==================================================")
    print(f"Total Emails Processed: {len(dataset)}")
    print(f"System Average Overall Score: {average_overall_score:.2f} / 5.00")
    print("==================================================")
    
    # Save results to a file for review
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to evaluation_results.json")

if __name__ == "__main__":
    main()
