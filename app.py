import streamlit as st
import os
import json
import time
from google import genai
from dotenv import load_dotenv
from generator import generate_reply
from evaluator import evaluate_reply

# Load environment and configure API
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Hiver AI Email Responder", page_icon="📧", layout="wide")

st.title("📧 Hiver Challenge: AI Email Responder")
st.markdown("This dashboard demonstrates an end-to-end AI email response generator and evaluator using **Google Gemini 2.5 Flash**.")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found. Please add it to your `.env` file.")
    st.stop()

# Initialize Client
client = genai.Client(api_key=api_key)

# Load dataset
@st.cache_data
def load_dataset():
    with open("dataset.json", "r") as f:
        return json.load(f)

dataset = load_dataset()

# Sidebar for Selection
st.sidebar.header("Select an Email")
email_options = [f"Email {item['id']}: {item['category'].replace('_', ' ').title()}" for item in dataset]
selected_option = st.sidebar.selectbox("Choose a customer email from the dataset:", email_options)
selected_index = email_options.index(selected_option)
selected_item = dataset[selected_index]

st.sidebar.markdown("---")
st.sidebar.info("💡 **How it works:**\n1. Select an email.\n2. The system generates a suggested reply using few-shot prompting.\n3. An LLM acts as a QA Judge to score the reply on Tone, Helpfulness, and Conciseness.")

# Main Display
st.subheader("📨 Incoming Customer Email")
st.info(selected_item['incoming'])

if st.button("Generate Reply & Evaluate", type="primary"):
    with st.spinner("🧠 Generating intelligent reply..."):
        try:
            output, retrieved_examples = generate_reply(client, selected_item['incoming'], dataset)
            
            if retrieved_examples:
                st.subheader("⚡ FAISS Vector Search Results")
                for i, ex in enumerate(retrieved_examples):
                    with st.expander(f"Top Match {i+1}: Past {ex['category'].replace('_', ' ').title()} Email"):
                        st.markdown(f"**Incoming:** {ex['incoming']}\n\n**Reply Used:** {ex['reference_reply']}")
            
            st.subheader("🤖 AI Response Generation")
            st.write(f"**Detected Intent:** `{output.intent}`")
            if output.requires_human:
                st.error("⚠️ HIGH RISK: Flagged for Human-in-the-Loop Review")
            else:
                st.success("✅ LOW RISK: Safe for Automated Delivery")
            
            st.markdown("### Suggested Reply:")
            st.info(output.suggested_reply)
            
            st.markdown("---")
            st.subheader("⚖️ AI Judge Evaluation")
            with st.spinner("Grading response with Hallucination Checks..."):
                eval_result = evaluate_reply(client, selected_item['incoming'], output.suggested_reply, selected_item['reference_reply'])
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Overall Score", f"{eval_result['overall_score']:.1f}/5")
                col2.metric("Tone", f"{eval_result['tone_score']}/5")
                col3.metric("Helpfulness", f"{eval_result['helpfulness_score']}/5")
                col4.metric("Conciseness", f"{eval_result['conciseness_score']}/5")
                col5.metric("Hallucination Penalty", f"{eval_result['hallucination_penalty']}", delta_color="inverse")
                
                st.markdown("**Judge's Reasoning:**")
                st.write(f"> {eval_result['reasoning']}")
                
        except Exception as e:
            if "429" in str(e):
                st.error("⚠️ **Rate Limit Exceeded (429):** The free tier limits you to 5 requests per minute. Please wait 30 seconds and try again!")
            else:
                st.error(f"❌ An error occurred: {e}")
