import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# --- Watsonx Setup ---
API_KEY = "IBM API Key"
PROJECT_ID = "Project Id"
MODEL_ID = "ibm/granite-3-8b-instruct"

credentials = Credentials(
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=API_KEY
)

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 500,
    "min_new_tokens": 0,
    "repetition_penalty": 1
}

model = ModelInference(
    model_id=MODEL_ID,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID
)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("🤖 TravelBuddy-My Travel assistant")

# --- Chat Session History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- User Input ---
user_input = st.text_input("Ask a question:")

if st.button("Send") and user_input.strip() != "":
    # Send to watsonx
    with st.spinner("Thinking..."):
        try:
            response = model.generate_text(prompt=user_input, guardrails=True)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("AI", response))
        except Exception as e:
            st.error(f"Error: {e}")

# --- Display Chat History ---
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧍 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")
