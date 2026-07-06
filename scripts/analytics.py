import json
import collections

def generate_report(results_file="data/evaluation_results.json"):
    try:
        with open(results_file, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {results_file} not found. Run main.py first.")
        return

    if not results:
        print("No results to analyze.")
        return

    total_emails = len(results)
    
    # Metrics
    total_score = 0
    total_hallucination = 0
    intents = collections.Counter()
    hitl_count = 0
    
    for r in results:
        eval_data = r.get("evaluation", {})
        gen_data = r.get("generator_output", {})
        
        total_score += eval_data.get("overall_score", 0)
        
        # Hallucination Penalty is negative, so a penalty < 0 means it hallucinated
        if eval_data.get("hallucination_penalty", 0) < 0:
            total_hallucination += 1
            
        intents[gen_data.get("intent", "unknown")] += 1
        
        if gen_data.get("requires_human", False):
            hitl_count += 1
            
    avg_score = total_score / total_emails
    hallucination_rate = (total_hallucination / total_emails) * 100
    hitl_rate = (hitl_count / total_emails) * 100

    print("=" * 50)
    print(" 🚀 HIVER AI EVALUATION REPORT 🚀 ")
    print("=" * 50)
    print(f"Total Emails Processed : {total_emails}")
    print(f"System Average Score   : {avg_score:.2f} / 5.0")
    print(f"Hallucination Rate     : {hallucination_rate:.1f}%")
    print(f"Escalation (HITL) Rate : {hitl_rate:.1f}%")
    print("-" * 50)
    print("Top Detected Intents:")
    for intent, count in intents.most_common(5):
        print(f"  - {intent}: {count} ({(count/total_emails)*100:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    generate_report()
