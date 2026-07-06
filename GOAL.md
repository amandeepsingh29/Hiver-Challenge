The task
Build an AI email suggested-response system
Build a system that, given an incoming email, generates a suggested reply — learning from a dataset of past emails and their responses. Then build a way to measure how good each generated response actually is.

1. Build a dataset
You own the data. Create (or source) a dataset of past emails paired with the replies that were sent. It can be synthetic, from public corpora, or hand-authored — your call. Explain in your README where it came from and why it's representative.

2. Generate suggested responses (Gen AI)
Build a system that takes a new incoming email and produces a suggested reply using a generative AI model (an LLM) — not a classical ML classifier. Ground the generation in your dataset. You choose how — prompting, RAG/retrieval over past emails, few-shot examples, fine-tuning an LLM, or a mix. Justify the trade-offs.

3. Measure accuracy — the core of this challenge
This is what we care about most. Build an accuracy system that, for a generated response, tells us how accurate/good it is and why. Think about:

What "accurate" even means for a suggested reply (exact match is too strict)
The metric(s) you use and why they're the right ones
How you validate the metric reflects real quality, not just a number
Reporting: per-response scores and an overall system score
What we're evaluating
Clarity of thinking about accuracy/evaluation (weighted heaviest)
Quality and honesty of the dataset
Whether the response generator is sensible and runs
A README covering your approach, trade-offs, and how to run it
Ship something that runs end-to-end. Tell us in the README how you used AI tools.

Deliverables
A public GitHub repository URL.
The dataset (or a script that generates/fetches it) and how you built it.
The Gen-AI response generator, runnable end-to-end.
The accuracy/evaluation system, with per-response and overall scores.
A README: your approach, why your accuracy metric is right, and how to run it.