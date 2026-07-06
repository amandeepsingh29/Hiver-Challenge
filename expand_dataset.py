import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API Key not found.")
    exit(1)

client = genai.Client(api_key=api_key)

print("Loading existing dataset...")
with open("dataset.json", "r") as f:
    dataset = json.load(f)

# We want 50 total. We have 5. We need 45 more.
needed = 50 - len(dataset)
print(f"Need {needed} more items.")

if needed <= 0:
    print("Already have 50 or more items.")
    exit(0)

# We will generate in batches of 15 to avoid hitting output token limits or parsing errors.
schema = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "category": {"type": "STRING", "description": "e.g., bug_report, feature_request, billing, how_to, feedback, outage"},
            "incoming": {"type": "STRING", "description": "The customer's email"},
            "reference_reply": {"type": "STRING", "description": "The ideal, polite, and helpful support reply from Hiver"}
        },
        "required": ["category", "incoming", "reference_reply"]
    }
}

batch_size = 15
current_id = len(dataset) + 1

for i in range(0, needed, batch_size):
    amount_to_generate = min(batch_size, needed - i)
    print(f"Generating {amount_to_generate} items... (Batch {i//batch_size + 1})")
    
    prompt = f"Generate {amount_to_generate} diverse customer support emails for a SaaS product (Hiver) along with ideal reference replies. Make sure the scenarios are varied (e.g. bugs, billing, angry customers, confused users, feature requests, etc)."
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                # Pass the schema directly since the new SDK supports it this way or via BaseModel
            )
        )
        
        # Parse the output
        try:
            new_items = json.loads(response.text)
            for item in new_items:
                item['id'] = str(current_id)
                dataset.append(item)
                current_id += 1
            print(f"Successfully added {len(new_items)} items.")
        except Exception as parse_e:
            print(f"Failed to parse JSON: {parse_e}")
            print(f"Raw output: {response.text}")
            
    except Exception as e:
        print(f"API Error: {e}")
    
    # Sleep to respect rate limits
    print("Sleeping for 15 seconds to respect rate limits...")
    time.sleep(15)

print(f"Dataset now has {len(dataset)} items.")
with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
print("Updated dataset.json")
