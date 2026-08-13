import streamlit as st
import pandas as pd
from transformers import pipeline
import time
from vector_store import retrieve_documents, load_vector_store, get_all_documents


def load_dataset():
    """Load the customer queries dataset"""
    try:
        df = pd.read_csv("customer_queries.csv")
        return df
    except FileNotFoundError:
        return None


@st.cache_resource
def load_qa_pipeline():
    """Load the QA pipeline"""
    return pipeline("question-answering", model="deepset/roberta-base-squad2")


@st.cache_resource
def load_text_generation_pipeline():
    """Load the text generation pipeline"""
    return pipeline("text-generation", model="distilgpt2")


def get_sample_queries():
    """Get sample queries from dataset"""
    df = load_dataset()
    if df is not None:
        return df["query"].head(5).tolist()
    return ["What is the status of my order?", "How do I reset my password?"]


def main():
    st.set_page_config(
        page_title="Customer Query Assistant",
        page_icon="comment",
        layout="wide"
    )
    
    st.title("Customer Query Assistant")
    st.markdown("---")
    
    # Initialize session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "loading" not in st.session_state:
        st.session_state.loading = False
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        model_type = st.selectbox(
            "Select Model Type",
            ["Question Answering", "Text Generation", "RAG Assistant"],
            help="Choose between QA pipeline, generative model, or RAG-enhanced assistant"
        )
        
        prompt_template = st.selectbox(
            "Select Prompt Template",
            ["Direct Q&A", "Assistant Style", "Instruction Based"],
            help="Choose how the query is formatted"
        )
        
        use_rag = st.checkbox("Use RAG System", value=(model_type == "RAG Assistant"))
        
        if use_rag:
            retrieval_k = st.slider("Number of documents to retrieve", 1, 5, 3)
        
        st.divider()
        st.subheader("Dataset Preview")
        df = load_dataset()
        if df is not None:
            st.metric("Total Queries", len(df))
            if st.button("Show Dataset Sample"):
                st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("Show Knowledge Base"):
            with st.expander("FAQ & Documentation", expanded=False):
                docs = get_all_documents()
                for doc in docs:
                    st.write(f"**{doc['title']}**")
                    st.write(doc['content'])
                    st.divider()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Query Input")
        
        input_method = st.radio(
            "How would you like to input?",
            ["Text Input", "Sample Query"],
            horizontal=True
        )
        
        if input_method == "Text Input":
            user_query = st.text_input(
                "Enter your query:",
                placeholder="Type your customer query here..."
            )
        else:
            sample_queries = get_sample_queries()
            user_query = st.selectbox(
                "Select a sample query:",
                sample_queries
            )
    
    # Context input (if using QA mode without RAG)
    if not use_rag and model_type == "Question Answering":
        context_text = st.text_area(
            "Enter context text:",
            placeholder="Paste the document or context for the model to search within...",
            height=150
        )
    else:
        context_text = None
    
    # Processing button
    col1, col2, col3 = st.columns(3)
    
    with col1:
        submit_button = st.button("Process Query", type="primary", use_container_width=True)
    
    with col2:
        if st.button("Clear History", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
    
    with col3:
        if st.button("Show History", use_container_width=True):
            st.session_state.show_history = not st.session_state.get("show_history", False)
    
    # Process query
    if submit_button and user_query:
        if model_type == "Question Answering" and not context_text:
            st.error("Please provide context text for Question Answering mode")
        else:
            with st.spinner("Processing your query..."):
                try:
                    retrieved_context = None
                    retrieved_docs = []
                    
                    # Retrieve documents if using RAG
                    if use_rag:
                        with st.spinner("Retrieving relevant documents..."):
                            retrieved_docs = retrieve_documents(user_query, k=retrieval_k if 'retrieval_k' in locals() else 3)
                            retrieved_context = "\n\n".join(retrieved_docs)
                    
                    if model_type == "Question Answering":
                        qa_pipeline = load_qa_pipeline()
                        context_to_use = retrieved_context if use_rag else context_text
                        
                        result = qa_pipeline(question=user_query, context=context_to_use)
                        
                        response = result.get("answer", "No answer found")
                        confidence = result.get("score", 0)
                        
                        st.success("Answer Generated!")
                        st.subheader("Response")
                        st.write(f"**Answer:** {response}")
                        st.metric("Confidence Score", f"{confidence:.1%}")
                        
                        # Display retrieved documents if using RAG
                        if use_rag and retrieved_docs:
                            with st.expander("Retrieved Context Documents"):
                                for i, doc in enumerate(retrieved_docs, 1):
                                    st.write(f"**Document {i}:**")
                                    st.write(doc)
                                    st.divider()
                        
                    else:  # Text Generation or RAG Assistant
                        gen_pipeline = load_text_generation_pipeline()
                        
                        if use_rag:
                            if prompt_template == "Assistant Style":
                                formatted_query = f"Context: {retrieved_context}\n\nAssistant: Based on the above context, {user_query}"
                            elif prompt_template == "Instruction Based":
                                formatted_query = f"Context: {retrieved_context}\n\nTask: Answer the customer query below.\nQuery: {user_query}\nAnswer:"
                            else:
                                formatted_query = f"Context: {retrieved_context}\n\nQuery: {user_query}\nAnswer:"
                        else:
                            if prompt_template == "Assistant Style":
                                formatted_query = f"Assistant: Help me with this customer query: {user_query}"
                            elif prompt_template == "Instruction Based":
                                formatted_query = f"Task: Answer the customer query below.\nQuery: {user_query}\nAnswer:"
                            else:
                                formatted_query = user_query
                        
                        result = gen_pipeline(
                            formatted_query,
                            max_length=200,
                            num_return_sequences=1,
                            temperature=0.7
                        )
                        
                        response = result[0]["generated_text"]
                        
                        st.success("Response Generated!")
                        st.subheader("Response")
                        st.write(response)
                        
                        # Display retrieved documents if using RAG
                        if use_rag and retrieved_docs:
                            with st.expander("Retrieved Context Documents"):
                                for i, doc in enumerate(retrieved_docs, 1):
                                    st.write(f"**Document {i}:**")
                                    st.write(doc)
                                    st.divider()
                    
                    # Add to history
                    st.session_state.conversation_history.append({
                        "query": user_query,
                        "response": response,
                        "model": model_type,
                        "template": prompt_template,
                        "rag_used": use_rag,
                        "retrieved_docs": retrieved_docs
                    })
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.info("Make sure all required dependencies are installed: pip install -r requirements.txt")
    
    # Display conversation history
    if st.session_state.get("show_history", False):
        st.divider()
        st.subheader("Conversation History")
        
        if st.session_state.conversation_history:
            for i, item in enumerate(st.session_state.conversation_history, 1):
                with st.expander(f"Query {i}: {item['query'][:50]}..."):
                    st.write(f"**Model:** {item['model']}")
                    st.write(f"**Template:** {item['template']}")
                    if item.get('rag_used', False):
                        st.write("**RAG Enabled:** Yes")
                        if item.get('retrieved_docs'):
                            with st.expander("Retrieved Documents"):
                                for doc in item['retrieved_docs']:
                                    st.write(doc)
                                    st.divider()
                    st.write(f"**Query:** {item['query']}")
                    st.write(f"**Response:** {item['response']}")
        else:
            st.info("No queries processed yet")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        Customer Query Assistant | Powered by Hugging Face Transformers & Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
