import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset


def load_customer_queries_dataset(csv_path):
    """Load customer queries from CSV and prepare for fine-tuning."""
    df = pd.read_csv(csv_path)
    
    # Combine query, context, and response into a single text field
    texts = []
    for idx, row in df.iterrows():
        text = f"Query: {row['query']}\nContext: {row['context']}\nResponse: {row['response']}"
        texts.append(text)
    
    dataset = Dataset.from_dict({"text": texts})
    return dataset


def tokenize_function(examples, tokenizer):
    """Tokenize the dataset."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )


def fine_tune_model():
    """Fine-tune a pre-trained language model on customer query data."""
    
    # Configuration
    model_name = "distilgpt2"
    csv_path = "customer_queries.csv"
    output_dir = "./models/fine_tuned_model"
    
    print("Loading pre-trained model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
    
    print("Loading customer queries dataset...")
    dataset = load_customer_queries_dataset(csv_path)
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        learning_rate=5e-5,
        gradient_accumulation_steps=2,
        fp16=False,
    )
    
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving fine-tuned model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    fine_tune_model()
