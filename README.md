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
