import json
import math
import os
import shutil

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
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

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
    
    epochs = 3
    batch_size = 4
    grad_accum = 2

    # Scale warmup to the run length. This dataset only produces ~21 optimizer
    # steps, so a fixed warmup_steps=100 meant the learning rate never finished
    # ramping and training ended at a fifth of the configured rate.
    total_steps = math.ceil(
        math.ceil(len(tokenized_dataset) / batch_size) / grad_accum
    ) * epochs
    warmup_steps = max(2, int(0.1 * total_steps))
    print(f"Total optimizer steps: {total_steps}, warmup steps: {warmup_steps}")

    # Training arguments
    # Note: overwrite_output_dir was removed from TrainingArguments in
    # transformers 5.x, so the output directory is cleared manually above.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_steps=1,
        report_to="none",
        save_steps=50,
        save_total_limit=2,
        learning_rate=5e-5,
        gradient_accumulation_steps=grad_accum,
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

    # Persist the loss curve so training runs can be compared afterwards.
    with open("fine_tune_log.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    steps = [e for e in trainer.state.log_history if "loss" in e]
    if steps:
        print(f"\nTotal optimizer steps: {trainer.state.max_steps}")
        print(f"warmup_steps setting:  {training_args.warmup_steps}")
        print(f"First loss: {steps[0]['loss']:.4f} (lr={steps[0].get('learning_rate', 0):.2e})")
        print(f"Final loss: {steps[-1]['loss']:.4f} (lr={steps[-1].get('learning_rate', 0):.2e})")

    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    fine_tune_model()
