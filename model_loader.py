"""
Model Loader for Customer Query Assistant
Loads a pre-trained LLM from Hugging Face and generates responses to queries.
"""

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers library not available. Install with: pip install transformers")

def load_model_and_tokenizer(model_name: str):
    """
    Loads a pre-trained model and tokenizer from Hugging Face.
    
    Args:
        model_name (str): The name/ID of the model from Hugging Face Hub
        
    Returns:
        tuple: (tokenizer, model) loaded from Hugging Face
    """
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    print(f"Loading model: {model_name}")
    print("(This may take a moment on first load as the model is downloaded...)")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print(f"✅ Model and tokenizer loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

def generate_response(tokenizer, model, query: str, max_length: int = 128) -> str:
    """
    Generates a response to a query using the loaded model.
    
    Args:
        tokenizer: The tokenizer for encoding the query
        model: The pre-trained language model
        query (str): The input query/prompt
        max_length (int): Maximum length of generated response
        
    Returns:
        str: Generated response text
    """
    if tokenizer is None or model is None:
        return "Model not loaded. Please install transformers: pip install transformers torch"
    
    try:
        # Tokenize the input query
        inputs = tokenizer.encode(query, return_tensors="pt")
        
        # Generate response with the model
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_beams=2,
            temperature=0.7,
            do_sample=True,
            early_stopping=True
        )
        
        # Decode the generated tokens to text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    """Main function to demonstrate model loading and response generation."""
    
    if not TRANSFORMERS_AVAILABLE:
        print("\n" + "="*70)
        print("⚠️  TRANSFORMERS LIBRARY NOT INSTALLED")
        print("="*70)
        print("\nTo run this script, install the required packages:")
        print("  pip install transformers torch")
        print("\nFor long paths on Windows (OneDrive), you may need to:")
        print("  1. Move project to a shorter path (e.g., C:\\Projects\\)")
        print("  2. Or configure pip to handle long paths")
        print("\nModel loading logic is implemented and ready to use.")
        return
    
    # Choose a lightweight, efficient model suitable for customer queries
    # google/flan-t5-small is a good choice: small, fast, and good at instruction following
    model_name = "google/flan-t5-small"
    
    # Sample queries to test the model
    sample_queries = [
        "What is the capital of France?",
        "How do I reset my password?",
        "What are the benefits of cloud computing?",
        "How can I track my order?",
        "What is your return policy?",
    ]
    
    try:
        # Load model and tokenizer
        tokenizer, model = load_model_and_tokenizer(model_name)
        
        if tokenizer is None or model is None:
            print("Failed to load model. Please check your internet connection and try again.")
            return
        
        print("\n" + "="*70)
        print("GENERATING RESPONSES TO SAMPLE QUERIES")
        print("="*70 + "\n")
        
        # Generate responses for each query
        for i, query in enumerate(sample_queries, 1):
            print(f"[Query {i}]")
            print(f"❓ Query: {query}")
            
            # Generate response
            response = generate_response(tokenizer, model, query)
            print(f"🤖 Response: {response}")
            print("-" * 70 + "\n")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
