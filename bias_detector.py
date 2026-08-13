import re
from typing import Tuple, List, Dict


BIAS_PATTERNS = {
    "gender_bias": [
        r"\b(she|he|her|his|man|woman|male|female)\b",
        r"\b(nurse|secretary|doctor|engineer)\b.*\b(she|he)\b",
    ],
    "racial_bias": [
        r"\b(racial|ethnic|race|ethnicity|color|minority|majority)\b",
        r"\b(black|white|asian|hispanic|african|caucasian)\b",
    ],
    "age_bias": [
        r"\b(old|young|elderly|youth|aged|senior|junior)\b",
        r"\b(teenager|millennial|boomer|generation)\b",
    ],
    "ability_bias": [
        r"\b(disabled|handicapped|retarded|crazy|insane|mentally ill)\b",
        r"\b(wheelchair|blind|deaf|dumb|stupid)\b",
    ],
    "religious_bias": [
        r"\b(christian|muslim|jewish|hindu|buddhist|atheist)\b",
        r"\b(church|mosque|synagogue|temple|faith|religion)\b",
    ],
    "political_bias": [
        r"\b(democrat|republican|liberal|conservative|socialist|communist)\b",
        r"\b(trump|biden|election|vote|politics)\b",
    ]
}

HARMFUL_KEYWORDS = [
    "hate", "discriminate", "racist", "sexist", "ageist", "bias",
    "stereotyp", "prejudic", "intoleran", "bigot", "discriminator",
    "violence", "attack", "harm", "abuse", "assault", "kill", "die",
    "stupid", "idiot", "retard", "crazy", "insane", "mad"
]

POSITIVE_SENTIMENT_KEYWORDS = [
    "good", "great", "excellent", "positive", "helpful", "professional",
    "fair", "equal", "inclusive", "diverse", "respectful", "thoughtful"
]


def detect_sensitive_attributes(text: str) -> Tuple[bool, List[str]]:
    """
    Detect sensitive attributes (gender, race, age, ability, religion, politics) in text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Tuple of (has_bias, bias_categories) where has_bias is True if sensitive attributes found,
        and bias_categories is a list of detected categories
    """
    text_lower = text.lower()
    detected_categories = []
    
    for category, patterns in BIAS_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                detected_categories.append(category)
                break
    
    return len(detected_categories) > 0, detected_categories


def detect_harmful_content(text: str) -> Tuple[bool, List[str]]:
    """
    Detect harmful content in text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Tuple of (has_harmful, harmful_keywords_found) where has_harmful is True if harmful content detected
    """
    text_lower = text.lower()
    harmful_found = []
    
    for keyword in HARMFUL_KEYWORDS:
        if keyword in text_lower:
            harmful_found.append(keyword)
    
    return len(harmful_found) > 0, harmful_found


def check_context_balance(text: str) -> bool:
    """
    Check if the text has positive sentiment keywords to balance potential sensitive attributes.
    
    Args:
        text: The text to analyze
        
    Returns:
        True if positive sentiment keywords are present
    """
    text_lower = text.lower()
    positive_count = sum(1 for keyword in POSITIVE_SENTIMENT_KEYWORDS if keyword in text_lower)
    return positive_count > 0


def analyze_bias(text: str) -> Dict:
    """
    Comprehensive bias analysis of text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with bias analysis results
    """
    has_sensitive, sensitive_categories = detect_sensitive_attributes(text)
    has_harmful, harmful_keywords = detect_harmful_content(text)
    has_positive_balance = check_context_balance(text)
    
    bias_score = 0
    bias_flags = []
    
    # Calculate bias score
    if has_sensitive:
        bias_score += 3
        bias_flags.append(f"Sensitive attributes detected: {', '.join(set(sensitive_categories))}")
    
    if has_harmful:
        bias_score += 5
        bias_flags.append(f"Harmful content detected: {', '.join(set(harmful_keywords[:3]))}")
    
    if not has_positive_balance and (has_sensitive or has_harmful):
        bias_score += 2
        bias_flags.append("Lacks positive context balance")
    
    if has_positive_balance and (has_sensitive or has_harmful):
        bias_score -= 2
        bias_score = max(0, bias_score)
    
    is_biased = bias_score >= 5
    
    return {
        "is_biased": is_biased,
        "bias_score": min(bias_score, 10),
        "bias_flags": bias_flags,
        "has_sensitive_attributes": has_sensitive,
        "sensitive_categories": sensitive_categories,
        "has_harmful_content": has_harmful,
        "harmful_keywords": harmful_keywords,
        "has_positive_balance": has_positive_balance,
        "risk_level": "HIGH" if bias_score >= 8 else "MEDIUM" if bias_score >= 5 else "LOW"
    }


def get_bias_warning_message(analysis: Dict) -> str:
    """
    Generate a warning message based on bias analysis.
    
    Args:
        analysis: The analysis result from analyze_bias()
        
    Returns:
        A formatted warning message
    """
    message = f"Risk Level: {analysis['risk_level']}\n"
    
    if analysis['bias_flags']:
        message += "Issues detected:\n"
        for flag in analysis['bias_flags']:
            message += f"• {flag}\n"
    
    if analysis['has_sensitive_attributes']:
        message += f"\nSensitive attributes found in categories: {', '.join(set(analysis['sensitive_categories']))}\n"
    
    if analysis['has_harmful_content']:
        message += f"Harmful keywords detected: {', '.join(set(analysis['harmful_keywords'][:3]))}\n"
    
    return message.strip()


if __name__ == "__main__":
    test_text = "The nurse is very good at her job and provides excellent care."
    analysis = analyze_bias(test_text)
    print(f"Analysis: {analysis}")
    print(f"\nWarning Message:\n{get_bias_warning_message(analysis)}")
