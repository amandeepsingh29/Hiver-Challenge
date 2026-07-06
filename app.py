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
            generated_reply = generate_reply(client, selected_item['incoming'], dataset)
            
            st.subheader("🤖 Generated Reply")
            st.success(generated_reply)
            
            with st.spinner("⚖️ Evaluating reply..."):
                # Pass the reference reply for evaluation
                eval_result = evaluate_reply(
                    client, 
                    selected_item['incoming'], 
                    generated_reply, 
                    selected_item['reference_reply']
                )
                
                st.subheader("📊 Evaluation Report")
                
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Overall Score", f"{eval_result['overall_score']:.2f}/5")
                col2.metric("Tone", f"{eval_result['tone_score']}/5")
                col3.metric("Helpfulness", f"{eval_result['helpfulness_score']}/5")
                col4.metric("Conciseness", f"{eval_result['conciseness_score']}/5")
                
                st.markdown(f"**📝 Reasoning:**\n> {eval_result['reasoning']}")
                
        except Exception as e:
            if "429" in str(e):
                st.error("⚠️ **Rate Limit Exceeded (429):** The free tier limits you to 5 requests per minute. Please wait 30 seconds and try again!")
            else:
                st.error(f"❌ An error occurred: {e}")
