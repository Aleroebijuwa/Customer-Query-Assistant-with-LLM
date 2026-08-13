"""
Prompt Engineering Module for Customer Query Assistant
Constructs effective prompts for model inference using query and context.
"""

def create_prompt(query: str, context: str, prompt_style: str = "assistant") -> str:
    """
    Constructs a prompt for a model using the given query and context.

    Args:
        query (str): The user's question or request.
        context (str): Relevant information to help the model answer the query.
        prompt_style (str): The style of prompt to generate. Options: 'assistant', 'qa', 'instruction'

    Returns:
        str: The formatted prompt string ready for model inference.
    """
    if prompt_style == "assistant":
        return _create_assistant_prompt(query, context)
    elif prompt_style == "qa":
        return _create_qa_prompt(query, context)
    elif prompt_style == "instruction":
        return _create_instruction_prompt(query, context)
    else:
        return _create_assistant_prompt(query, context)  # Default to assistant style


def _create_assistant_prompt(query: str, context: str) -> str:
    """
    Creates a conversational assistant-style prompt.
    
    This style treats the interaction as a natural conversation between
    a customer service representative and the user.
    
    Args:
        query (str): The user's question
        context (str): Relevant context information
        
    Returns:
        str: Formatted assistant-style prompt
    """
    prompt = f"""You are a helpful customer service assistant. Use the following information to answer the customer's question accurately and professionally.

**Context Information:**
{context}

**Customer Question:**
{query}

**Your Response:**
Please provide a clear, concise, and helpful answer based on the context provided. If the context doesn't contain relevant information, acknowledge this and provide general guidance."""
    
    return prompt


def _create_qa_prompt(query: str, context: str) -> str:
    """
    Creates a Question-Answer format prompt.
    
    This style emphasizes direct Q&A structure, useful for extractive
    or short-answer generation tasks.
    
    Args:
        query (str): The user's question
        context (str): Relevant context information
        
    Returns:
        str: Formatted Q&A style prompt
    """
    prompt = f"""Given the following context, answer the question below:

**Context:**
{context}

**Question:**
{query}

**Answer:**
"""
    
    return prompt


def _create_instruction_prompt(query: str, context: str) -> str:
    """
    Creates an instruction-following style prompt.
    
    This style emphasizes explicit instructions and structured output,
    useful for task-specific responses.
    
    Args:
        query (str): The user's question
        context (str): Relevant context information
        
    Returns:
        str: Formatted instruction-style prompt
    """
    prompt = f"""Task: Answer the following customer query using only the provided information.

**Instructions:**
1. Read the context carefully
2. Answer the query using only information from the context
3. Be specific and avoid speculation
4. Format your response clearly

**Relevant Information:**
{context}

**Customer Query:**
{query}

**Response:**
"""
    
    return prompt


def format_context(context_dict: dict) -> str:
    """
    Formats a dictionary of context information into a readable string.
    
    Args:
        context_dict (dict): Dictionary with context keys and values
        
    Returns:
        str: Formatted context string
    """
    if isinstance(context_dict, str):
        return context_dict
    
    formatted_lines = []
    for key, value in context_dict.items():
        formatted_lines.append(f"- {key}: {value}")
    
    return "\n".join(formatted_lines)


def create_prompt_with_examples(query: str, context: str, examples: list = None) -> str:
    """
    Creates a prompt with few-shot examples for better model guidance.
    
    Args:
        query (str): The user's question
        context (str): Relevant context information
        examples (list): List of (question, answer) tuples for examples
        
    Returns:
        str: Formatted prompt with examples
    """
    prompt = f"""You are a helpful customer service assistant. Learn from the examples below and apply the same approach to answer the new question.

**Examples:**
"""
    
    if examples:
        for i, (example_q, example_a) in enumerate(examples, 1):
            prompt += f"""
Example {i}:
Question: {example_q}
Answer: {example_a}
"""
    
    prompt += f"""
**Context Information:**
{context}

**Customer Question:**
{query}

**Your Response:**
Please provide a helpful answer following the style and approach shown in the examples above."""
    
    return prompt


def create_structured_prompt(query: str, context: str, desired_format: str = "text") -> str:
    """
    Creates a prompt that specifies desired output format.
    
    Args:
        query (str): The user's question
        context (str): Relevant context information
        desired_format (str): Desired output format ('text', 'list', 'json', 'bullet_points')
        
    Returns:
        str: Formatted prompt with structure guidance
    """
    format_instructions = {
        "text": "Please provide your answer as a clear, well-formatted paragraph.",
        "list": "Please provide your answer as a numbered list of key points.",
        "bullet_points": "Please provide your answer as bullet points (•).",
        "json": "Please format your answer as valid JSON with appropriate keys."
    }
    
    format_instruction = format_instructions.get(desired_format, format_instructions["text"])
    
    prompt = f"""You are a helpful customer service assistant. Use the following information to answer the customer's question.

**Context:**
{context}

**Question:**
{query}

**Format Requirement:**
{format_instruction}

**Answer:**
"""
    
    return prompt


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("PROMPT ENGINEERING EXAMPLES")
    print("="*70)
    
    # Sample query and context
    sample_query = "How do I reset my password?"
    sample_context = """
    To reset your password:
    1. Go to the login page and click "Forgot Password"
    2. Enter your email address associated with the account
    3. Check your email for a password reset link (arrives within 5 minutes)
    4. Click the link and create a new password (minimum 8 characters, must include letters and numbers)
    5. Log in with your new password
    
    Note: Password reset links expire after 24 hours. If you don't receive an email, check your spam folder or contact support.
    """
    
    # Test different prompt styles
    print("\n1. ASSISTANT STYLE PROMPT:")
    print("-" * 70)
    assistant_prompt = create_prompt(sample_query, sample_context, "assistant")
    print(assistant_prompt)
    
    print("\n2. Q&A STYLE PROMPT:")
    print("-" * 70)
    qa_prompt = create_prompt(sample_query, sample_context, "qa")
    print(qa_prompt)
    
    print("\n3. INSTRUCTION STYLE PROMPT:")
    print("-" * 70)
    instruction_prompt = create_prompt(sample_query, sample_context, "instruction")
    print(instruction_prompt)
    
    print("\n4. STRUCTURED PROMPT (BULLET POINTS):")
    print("-" * 70)
    structured_prompt = create_structured_prompt(sample_query, sample_context, "bullet_points")
    print(structured_prompt)
    
    print("\n" + "="*70)
    print("Prompt engineering module loaded")
    print("="*70)
