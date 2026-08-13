import pandas as pd

def generate_synthetic_data(num_samples=50):
    """
    Generate synthetic customer queries and responses for retail/SaaS domains.
    
    Args:
        num_samples (int): Number of sample records to generate
        
    Returns:
        pd.DataFrame: DataFrame with 'query', 'context', and 'response' columns
    """
    data = []
    
    # Product Inquiry Queries
    product_queries = [
        {
            "query": "What are the features of the Premium plan?",
            "context": "User browsing pricing page, asked about Premium subscription tier",
            "response": "The Premium plan includes unlimited storage, priority support, advanced analytics, API access, and team collaboration features."
        },
        {
            "query": "Is your product compatible with Mac?",
            "context": "Mac user checking system requirements before purchase",
            "response": "Yes, our product is fully compatible with Mac OS 10.14 and later. You can download it from our website or the App Store."
        },
        {
            "query": "Do you offer a free trial?",
            "context": "New customer interested in testing before purchasing",
            "response": "Yes, we offer a 30-day free trial of all our plans with full access to features. No credit card required to sign up."
        },
    ]
    
    # Order Status Queries
    order_queries = [
        {
            "query": "Where is my order? It's been 5 days.",
            "context": "Order ID: ORD-2024-78932, Customer placed order on August 8th",
            "response": "Your order is currently in transit and should arrive within 2-3 business days. You can track your shipment using tracking number TRK-9876543 on our website."
        },
        {
            "query": "Can I change my delivery address?",
            "context": "Customer wants to change address for order ORD-2024-89234 placed today",
            "response": "If the order hasn't shipped yet, we can change the address. Please contact us immediately with your order ID and we'll update it for you."
        },
        {
            "query": "How long does shipping take?",
            "context": "New customer checking standard delivery times",
            "response": "Standard shipping takes 5-7 business days, Express shipping takes 2-3 business days, and Overnight shipping is available for urgent orders."
        },
    ]
    
    # Billing/Payment Queries
    billing_queries = [
        {
            "query": "Why was I charged twice for my subscription?",
            "context": "Customer account shows duplicate charge for monthly subscription",
            "response": "We apologize for the duplicate charge. We've identified the issue and will process a refund immediately. Please check your account within 3-5 business days."
        },
        {
            "query": "Can I get an invoice for my purchase?",
            "context": "Customer needs receipt for business expense documentation",
            "response": "Yes, you can download invoices from your account dashboard under 'Billing & Receipts' or we can email you a PDF copy directly."
        },
        {
            "query": "Do you accept PayPal?",
            "context": "Customer checkout, wants to use PayPal payment method",
            "response": "Yes, we accept PayPal, credit cards (Visa, Mastercard, Amex), Apple Pay, Google Pay, and bank transfers for larger amounts."
        },
    ]
    
    # Technical Support Queries
    support_queries = [
        {
            "query": "The app keeps crashing on my phone. What should I do?",
            "context": "Mobile app user experiencing crashes on iOS device, version 2.1.0",
            "response": "Try these steps: 1) Update to the latest version, 2) Clear app cache, 3) Restart your phone, 4) Reinstall the app if needed. Contact support if the issue persists."
        },
        {
            "query": "How do I reset my password?",
            "context": "User locked out of account, forgot password",
            "response": "Click 'Forgot Password' on the login page, enter your email, and you'll receive a password reset link within 5 minutes. Follow the link to create a new password."
        },
        {
            "query": "I'm getting an error code 404 when trying to log in.",
            "context": "User reports authentication error during login attempt",
            "response": "Error 404 usually indicates a connection issue. Try these: 1) Check your internet connection, 2) Clear browser cookies, 3) Try a different browser, 4) Wait 10 minutes and retry."
        },
    ]
    
    # Account Management Queries
    account_queries = [
        {
            "query": "How do I update my profile information?",
            "context": "User wants to change email, phone number, or address on their account",
            "response": "Go to Settings > Account > Profile Information. You can update your email, phone, address, and other details. Changes take effect immediately."
        },
        {
            "query": "Can I have multiple accounts under the same email?",
            "context": "User asking about creating multiple user profiles",
            "response": "Each email can have only one primary account. However, you can add team members or create sub-accounts within your workspace for team collaboration."
        },
        {
            "query": "How do I delete my account?",
            "context": "User wants to permanently remove their account",
            "response": "Go to Settings > Account > Delete Account. Note that this is permanent and all your data will be deleted. Download your data first if needed."
        },
    ]
    
    # Return/Refund Queries
    return_queries = [
        {
            "query": "What's your return policy?",
            "context": "Customer checking return eligibility before purchase",
            "response": "We offer 30-day returns on unopened items for a full refund. For opened items, we offer store credit within 14 days. Shipping costs are non-refundable."
        },
        {
            "query": "I received a damaged product. What should I do?",
            "context": "Customer received defective item, order ORD-2024-45678",
            "response": "We're sorry for the inconvenience. Please reply with photos of the damaged item, and we'll send a replacement immediately at no cost to you."
        },
        {
            "query": "How long does a refund take?",
            "context": "Customer initiated return, wants to know processing timeline",
            "response": "Once we receive your return, processing takes 5-7 business days. The refund will appear in your account 3-5 business days after approval."
        },
    ]
    
    # General FAQ Queries
    faq_queries = [
        {
            "query": "Do you have customer support available 24/7?",
            "context": "Customer checking support availability",
            "response": "Our support team is available Monday-Friday 9AM-6PM EST. For urgent issues, we have an emergency support line available 24/7."
        },
        {
            "query": "Is my data secure with your company?",
            "context": "User concerned about data privacy and security",
            "response": "Yes, we use 256-bit SSL encryption, comply with GDPR/CCPA, and undergo regular security audits. Your data is stored in secure, redundant data centers."
        },
        {
            "query": "Can I export my data?",
            "context": "User wants to download or export their stored data",
            "response": "Yes, you can export your data in CSV, JSON, or PDF format from Settings > Data Export. The export is prepared within 24 hours."
        },
    ]
    
    # Combine all query types
    all_queries = (
        product_queries + order_queries + billing_queries + 
        support_queries + account_queries + return_queries + faq_queries
    )
    
    # Use provided data and repeat/vary if needed to reach num_samples
    while len(data) < num_samples:
        for query_template in all_queries:
            if len(data) < num_samples:
                data.append(query_template.copy())
            else:
                break
    
    # Trim to exact number requested
    data = data[:num_samples]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['query', 'context', 'response'])
    return df

if __name__ == "__main__":
    # Generate synthetic dataset with 50 samples
    dataset = generate_synthetic_data(num_samples=50)
    
    # Save to CSV
    dataset.to_csv('customer_queries.csv', index=False)
    
    print("✅ Synthetic dataset generated successfully!")
    print(f"📊 Total records: {len(dataset)}")
    print(f"📁 File saved: customer_queries.csv")
    print(f"\nDataset Preview:")
    print(dataset.head(10))
