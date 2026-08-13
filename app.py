"""
Customer Query Assistant
Main Streamlit application
"""

import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(
    page_title="Customer Query Assistant",
    page_icon="comment",
    layout="wide"
)

st.title("Customer Query Assistant")
st.write("Powered by Hugging Face Transformers")

# Initialize session state for conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load the question-answering pipeline
@st.cache_resource
def load_qa_pipeline():
    """Load the QA pipeline from Hugging Face"""
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

qa_pipeline = load_qa_pipeline()

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    context_text = st.text_area(
        "Enter context/document text:",
        height=200,
        placeholder="Paste the text you want the assistant to answer questions about..."
    )

# Main chat interface
if context_text:
    st.subheader("Ask a Question")
    user_query = st.text_input("Your question:")
    
    if user_query and st.button("Get Answer", type="primary"):
        try:
            # Get answer from the model
            result = qa_pipeline(question=user_query, context=context_text)
            
            # Display the result
            st.success("Answer found")
            st.write(f"**Answer:** {result['answer']}")
            st.write(f"**Confidence Score:** {result['score']:.2%}")
            
            # Store in session history
            st.session_state.messages.append({
                "question": user_query,
                "answer": result['answer'],
                "score": result['score']
            })
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Display conversation history
    if st.session_state.messages:
        st.subheader("Conversation History")
        for i, msg in enumerate(st.session_state.messages, 1):
            with st.expander(f"Q{i}: {msg['question'][:50]}..."):
                st.write(f"**Question:** {msg['question']}")
                st.write(f"**Answer:** {msg['answer']}")
                st.write(f"**Confidence:** {msg['score']:.2%}")
else:
    st.info("Please enter context text in the sidebar to begin asking questions.")
