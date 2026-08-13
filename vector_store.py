import os
import pickle
import numpy as np
from typing import List, Tuple
import streamlit as st


try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


VECTOR_STORE_PATH = "vector_store.pkl"
DOCUMENTS_PATH = "documents.pkl"


def get_sample_documents():
    """Return sample FAQ and product documentation."""
    return [
        {
            "id": 1,
            "title": "How to reset my password",
            "content": "To reset your password, visit the login page and click 'Forgot Password'. Enter your email address and follow the instructions sent to your inbox. If you don't receive an email within 5 minutes, check your spam folder."
        },
        {
            "id": 2,
            "title": "Order tracking",
            "content": "You can track your order using the order number sent to your email. Visit our website, go to 'My Orders', and enter your order number and email address. You'll see real-time updates on your shipment status."
        },
        {
            "id": 3,
            "title": "Return policy",
            "content": "We offer a 30-day return policy for most items. Items must be unused and in original packaging. To initiate a return, contact our support team with your order number. Refunds are processed within 5-7 business days."
        },
        {
            "id": 4,
            "title": "Product specifications",
            "content": "Our products are designed with high-quality materials and undergo rigorous testing. Each product comes with a 1-year warranty covering manufacturing defects. Check the product page for detailed specifications."
        },
        {
            "id": 5,
            "title": "Billing and payment",
            "content": "We accept all major credit cards, PayPal, and bank transfers. Your billing information is encrypted and secure. You can view your billing history in your account dashboard under 'Billing & Payments'."
        },
        {
            "id": 6,
            "title": "Customer support contact",
            "content": "Contact our customer support team via email at support@company.com or call 1-800-123-4567. Our hours are Monday to Friday, 9 AM to 6 PM EST. We also offer live chat on our website during business hours."
        },
        {
            "id": 7,
            "title": "Account management",
            "content": "You can update your account information, change your password, and manage notification preferences from your account settings. Enable two-factor authentication for enhanced security."
        },
        {
            "id": 8,
            "title": "Shipping information",
            "content": "We offer free shipping on orders over $50. Standard shipping takes 5-7 business days. Express shipping is available for an additional fee. International shipping is available to select countries."
        },
        {
            "id": 9,
            "title": "Product warranty",
            "content": "All products come with a standard 1-year warranty. Extended warranty options are available for purchase. The warranty covers manufacturing defects but not damage from misuse or accidents."
        },
        {
            "id": 10,
            "title": "Technical support",
            "content": "If you experience technical issues, first try restarting your device. Clear your browser cache and cookies. If the problem persists, contact technical support with details about the error and your browser version."
        }
    ]


def initialize_vector_store():
    """Initialize or load the vector store with document embeddings."""
    if not FAISS_AVAILABLE:
        return None, None
    
    # Load or create documents
    if os.path.exists(DOCUMENTS_PATH):
        with open(DOCUMENTS_PATH, "rb") as f:
            documents = pickle.load(f)
    else:
        documents = get_sample_documents()
        with open(DOCUMENTS_PATH, "wb") as f:
            pickle.dump(documents, f)
    
    # Initialize embedder
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create embeddings for all documents
    doc_texts = [f"{doc['title']}\n{doc['content']}" for doc in documents]
    embeddings = embedder.encode(doc_texts, convert_to_numpy=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    
    return index, documents, embedder


@st.cache_resource
def load_vector_store():
    """Load the vector store with caching."""
    if not FAISS_AVAILABLE:
        return None, None, None
    
    index, documents, embedder = initialize_vector_store()
    return index, documents, embedder


def retrieve_documents(query: str, k: int = 3) -> List[str]:
    """Retrieve the top k most relevant documents for a query."""
    if not FAISS_AVAILABLE:
        return [
            "This is a placeholder document about product features.",
            "Another document about troubleshooting."
        ]
    
    try:
        index, documents, embedder = load_vector_store()
        
        if index is None or documents is None:
            return ["Unable to retrieve documents. Vector store not available."]
        
        # Embed the query
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        
        # Search for nearest neighbors
        distances, indices = index.search(query_embedding.astype(np.float32), k)
        
        # Retrieve the actual documents
        retrieved_docs = []
        for idx in indices[0]:
            if idx < len(documents):
                doc = documents[idx]
                retrieved_docs.append(f"{doc['title']}: {doc['content']}")
        
        return retrieved_docs if retrieved_docs else ["No relevant documents found."]
    
    except Exception as e:
        return [f"Error retrieving documents: {str(e)}"]


def get_all_documents() -> List[dict]:
    """Get all indexed documents."""
    if os.path.exists(DOCUMENTS_PATH):
        with open(DOCUMENTS_PATH, "rb") as f:
            return pickle.load(f)
    return get_sample_documents()


if __name__ == "__main__":
    # Test the vector store
    if FAISS_AVAILABLE:
        print("Initializing vector store...")
        index, documents, embedder = initialize_vector_store()
        print(f"Vector store initialized with {len(documents)} documents")
        
        test_query = "How do I reset my password?"
        print(f"\nQuery: {test_query}")
        retrieved = retrieve_documents(test_query)
        print("Retrieved documents:")
        for doc in retrieved:
            print(f"- {doc}")
    else:
        print("FAISS or sentence-transformers not available. Install with:")
        print("pip install faiss-cpu sentence-transformers")
