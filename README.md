# Hiver 100-Minute Challenge: AI Email Responder

This repository contains my solution to the Hiver 100-Minute Challenge. The goal is to build an AI system that suggests email replies and evaluates its own accuracy.

## Approach

### 1. Dataset
I created a synthetic dataset (`dataset.json`) containing 5 high-quality examples of customer support emails and ideal "reference" replies. These cover common scenarios like refund requests, technical issues, feature requests, and billing inquiries. This dataset serves both as few-shot context for generation and as ground-truth for evaluation.

### 2. Gen-AI Response Generator
I used **Google's Gemini 1.5 Flash** (via `google-generativeai`) as the LLM. 
To ground the generation in the dataset, I used **Few-Shot Prompting**. The `generator.py` script pulls a few examples from `dataset.json` and includes them in the system prompt.
**Trade-offs**: 
- *Why Few-Shot?* It's incredibly fast to implement (perfect for a 100-minute challenge) and provides the LLM with immediate context on the expected tone and format without the overhead of setting up a vector database for RAG or the cost/time of fine-tuning. 
- *Limitations*: Context window limits how many examples can be passed. For a production system with thousands of past emails, RAG (Retrieval-Augmented Generation) would be a better approach to fetch the most semantically relevant past tickets.

### 3. Measuring Accuracy (The Core Challenge)
For evaluating text generation like emails, exact-match or BLEU/ROUGE scores are often inadequate because they penalize paraphrasing and different sentence structures even if the semantic meaning and tone are perfect.

I implemented an **LLM-as-a-Judge** evaluation metric in `evaluator.py`, using `gemini-1.5-flash` with Structured Outputs (Pydantic schemas) to score the generated reply. 

**What "accurate" means:**
The system evaluates each reply on three axes (1-5 scale):
1. **Tone**: Is it professional, empathetic, and polite?
2. **Helpfulness**: Does it accurately address the customer's issue according to the reference reply?
3. **Conciseness**: Is it direct and without unnecessary fluff?

**Why this is the right metric:**
This accurately reflects human judgment of email quality. It goes beyond keyword matching and understands *nuance*. The evaluator outputs a JSON with these three scores, an overall average score, and a *reasoning* string explaining why it gave those scores.

## How to Run

1. Clone this repository.
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Set up your API key:
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
4. **Command-Line Runner:**
   Run the end-to-end evaluation pipeline in the terminal:
   ```bash
   python main.py
   ```
   The script will iterate over the dataset, generate replies, evaluate them, print the per-response scores and reasoning, and output a final system average score. A detailed JSON log will also be saved to `evaluation_results.json`.

5. **Interactive Dashboard:**
   To interactively test the generation and evaluation, run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Critical Analysis & Production Trade-offs

Because this was built within a 100-minute constraint, several naive shortcuts were taken. If this were a real production system for Hiver, the current architecture would be highly inadequate. Here is a critical breakdown of its flaws and what **must** be done for production:

### 1. Grounding is Naive (Random Few-Shot vs. RAG)
* **The Flaw**: The system currently selects few-shot examples from the dataset *at random*. If a customer asks for a refund, and the prompt includes 3 random examples about "login issues", the context is wasted and the model has no reference for the refund policy.
* **The Fix**: Implement a **Retrieval-Augmented Generation (RAG)** pipeline. We must embed all historical, high-CSAT support tickets into a vector database (e.g., Pinecone, Qdrant). When a new email arrives, we perform a semantic search to retrieve the top 3 most similar past cases and use *those* as the few-shot examples.

### 2. Evaluation is Biased ("Grading Your Own Homework")
* **The Flaw**: We are using `gemini-2.5-flash` to generate the reply, and the exact same model to act as the QA Judge. LLMs have a known bias toward their own output style. Worse, if the model hallucinates a policy during generation, it might fail to dock points during evaluation because it inherently believes its own hallucination.
* **The Fix**: The Generator and the Evaluator must be decoupled. Use a faster, cheaper model for generation (e.g., a fine-tuned Llama 3 or Gemini Flash), but use a larger, more capable "Teacher" model (e.g., `GPT-4o` or `Gemini 1.5 Pro`) exclusively for the evaluation step. 

### 3. High Risk of Dangerous Hallucinations
* **The Flaw**: The prompt blindly asks the LLM to write a reply. As seen in test cases, if asked about a "Slack Integration", the AI enthusiastically invents a step-by-step guide for a feature that doesn't exist.
* **The Fix**: We need **Guardrails**. 
    1. A Knowledge Base document must be injected into the context. 
    2. We need a pre-generation classification step: *Does the AI have enough context to answer this?*
    3. We must implement **Human-in-the-Loop (HITL) routing**. High-risk intents (e.g., "delete my account", "legal action") should bypass the generator entirely and be routed to a human queue.

### 4. Brittle Infrastructure
* **The Flaw**: To bypass rate limits, the code uses a synchronous `time.sleep(15)`. In production, a burst of 1,000 emails would completely lock the thread and crash the system.
* **The Fix**: Move to asynchronous processing (`asyncio`) or a message broker (Celery/Redis). Rate limits should be handled via exponential backoff (e.g., the `tenacity` library) rather than hardcoded sleeps.
