#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import torch
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    pipeline,
    TrainingArguments, 
    Trainer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
from torch.utils.data import DataLoader

# Add parent directory to path to import from search_engine.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_engine import ElasticSearchEngine, QUERIES_PATH, CORPUS_PATH, ES_HOST, INDEX_NAME

# Constants
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "qa_model")
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "covid_qa_dataset.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize NLTK
def download_nltk_resources():
    """Download required NLTK resources if not already downloaded"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading required NLTK resources...")
        nltk.download('punkt')
    print("NLTK resources are ready.")

# Load and prepare corpus data
def load_corpus_from_jsonl(corpus_path):
    """Load corpus from jsonl file and extract text content"""
    print(f"Loading corpus from {corpus_path}...")
    documents = []
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading documents"):
            try:
                doc = json.loads(line)
                text = ""
                
                # Handle different dataset structures
                if 'text' in doc:
                    if 'title' in doc and doc['title']:
                        text += doc['title'] + " "
                    text += doc['text']
                elif 'paper_text' in doc:
                    if 'title' in doc and doc['title']:
                        text += doc['title'] + " "
                    text += doc['paper_text']
                
                if text.strip():
                    documents.append({
                        'doc_id': doc.get('doc_id', doc.get('_id', '')),
                        'text': text
                    })
            except Exception as e:
                print(f"Error processing document: {e}")
    
    print(f"Loaded {len(documents)} documents.")
    return documents

def create_qa_dataset_from_retrieval(queries_path, corpus, es_engine, top_k=10):
    """
    Create a QA dataset by retrieving relevant documents for each query
    
    This function takes queries and uses the ElasticSearch engine to retrieve
    relevant documents. It then creates pairs of questions and context for QA.
    """
    # Load queries
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = [json.loads(line) for line in f]
    
    # Create a lookup dictionary for corpus
    corpus_dict = {doc['doc_id']: doc['text'] for doc in corpus}
    
    qa_pairs = []
    
    for query in tqdm(queries, desc="Creating QA dataset"):
        query_id = query.get('_id', '')
        question = query.get('text', '')
        
        if not question.strip():
            continue
        
        # Make sure the question ends with a question mark
        if not question.endswith('?'):
            question += '?'
        
        # Search for relevant documents
        search_results = es_engine.search(question, k=top_k)
        
        for hit in search_results:
            doc_id = hit["_source"]["doc_id"]
            if doc_id in corpus_dict:
                context = corpus_dict[doc_id]
                
                # Split into sentences to create smaller contexts
                sentences = sent_tokenize(context)
                
                # Group sentences into chunks that are not too long (max 512 tokens approx)
                max_chunk_length = 2000  # Approx. character limit for ~512 tokens
                chunks = []
                current_chunk = ""
                
                for sent in sentences:
                    if len(current_chunk) + len(sent) <= max_chunk_length:
                        current_chunk += " " + sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Create QA pairs for each chunk
                for chunk in chunks:
                    qa_pairs.append({
                        'question': question,
                        'context': chunk,
                        'query_id': query_id,
                        'doc_id': doc_id
                    })
    
    print(f"Created {len(qa_pairs)} QA pairs from {len(queries)} queries.")
    return qa_pairs

def initialize_qa_pipeline(model_name="distilbert-base-cased-distilled-squad"):
    """Initialize a QA pipeline with a pre-trained model"""
    print(f"Initializing QA pipeline with model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    
    return qa_pipeline, model, tokenizer

def fine_tune_qa_model(qa_dataset, model_name="distilbert-base-cased-distilled-squad", epochs=3, batch_size=8):
    """Fine-tune a pre-trained QA model on our dataset"""
    print(f"Fine-tuning QA model: {model_name}")
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(qa_dataset)
    
    # Split dataset into train and evaluation sets
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Function to preprocess data
    def preprocess_function(examples):
        questions = [q for q in examples["question"]]
        contexts = [c for c in examples["context"]]
        
        # Tokenize inputs
        inputs = tokenizer(
            questions,
            contexts,
            max_length=512,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # Since we don't have answer annotations, we'll create dummy answer spans
        # pointing to the first 5 tokens of each context
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        
        start_positions = []
        end_positions = []
        
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            # Find the start of the context (after the question)
            sequence_ids = inputs.sequence_ids(i)
            context_start = 0
            while sequence_ids[context_start] != 1:
                context_start += 1
            
            # Set start_positions to beginning of context and end_positions a few tokens later
            start_positions.append(context_start)
            end_positions.append(min(context_start + 5, len(sequence_ids) - 1))
        
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    # Convert to datasets format
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    # Preprocess the datasets
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    
    print(f"Model fine-tuning complete. Model saved to {MODEL_PATH}")
    return model, tokenizer

def answer_question(question, qa_pipeline, es_engine, corpus_dict, top_k=5):
    """
    Answer a question using the QA pipeline and ElasticSearch
    
    This function retrieves relevant documents from ElasticSearch
    and then uses the QA pipeline to extract answers.
    """
    if not question.endswith('?'):
        question += '?'
    
    # Get relevant contexts from ElasticSearch
    search_results = es_engine.search(question, k=top_k)
    
    answers = []
    for hit in search_results:
        doc_id = hit["_source"]["doc_id"]
        if doc_id in corpus_dict:
            context = corpus_dict[doc_id]
            
            # Split context into manageable chunks if too long
            sentences = sent_tokenize(context)
            chunks = []
            current_chunk = ""
            max_chunk_length = 2000  # Character limit approximation
            
            for sent in sentences:
                if len(current_chunk) + len(sent) <= max_chunk_length:
                    current_chunk += " " + sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Process each chunk to find answers
            for chunk in chunks:
                try:
                    result = qa_pipeline(question=question, context=chunk)
                    if result and result["score"] > 0.1:  # Filter low-confidence answers
                        answers.append({
                            "answer": result["answer"],
                            "score": result["score"],
                            "doc_id": doc_id,
                            "context": chunk
                        })
                except Exception as e:
                    print(f"Error processing context: {e}")
    
    # Sort answers by confidence score
    answers = sorted(answers, key=lambda x: x["score"], reverse=True)
    return answers

def interactive_qa_mode(qa_pipeline, es_engine, corpus_dict):
    """Interactive mode for asking questions to the QA system"""
    print("\n=== COVID-19 Q&A System ===")
    print("Type your questions about COVID-19 or type 'exit' to quit.\n")
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
        
        print("\nSearching for answers...")
        answers = answer_question(question, qa_pipeline, es_engine, corpus_dict)
        
        if not answers:
            print("I couldn't find a good answer to your question in the corpus.")
        else:
            print("\nTop answers:")
            for i, ans in enumerate(answers[:3], 1):  # Show top 3 answers
                print(f"\n{i}. {ans['answer']} (confidence: {ans['score']:.2f})")
                print(f"   From document: {ans['doc_id']}")
                print(f"   Context: \"{ans['context'][:150]}...\"")

def main():
    """Main function to set up and run the QA system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='COVID-19 Question Answering System')
    parser.add_argument('--create-dataset', action='store_true', help='Create a QA dataset from the corpus')
    parser.add_argument('--fine-tune', action='store_true', help='Fine-tune the QA model on our dataset')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive Q&A mode')
    parser.add_argument('--model', type=str, default="distilbert-base-cased-distilled-squad", 
                        help='Pre-trained model to use (default: distilbert-base-cased-distilled-squad)')
    args = parser.parse_args()
    
    # Make sure NLTK resources are downloaded
    download_nltk_resources()
    
    # Set up ElasticSearch
    es_engine = ElasticSearchEngine(host=ES_HOST)
    if not es_engine.check_connection():
        print("Cannot connect to ElasticSearch server. Make sure it's running.")
        return
    
    # Create output directories
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    
    # Create dataset if requested
    if args.create_dataset:
        print("\n=== Creating QA Dataset ===")
        corpus = load_corpus_from_jsonl(CORPUS_PATH)
        qa_dataset = create_qa_dataset_from_retrieval(QUERIES_PATH, corpus, es_engine)
        
        # Save dataset
        with open(DATASET_PATH, 'w', encoding='utf-8') as f:
            json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
        print(f"QA dataset saved to {DATASET_PATH}")
    
    # Fine-tune model if requested
    if args.fine_tune:
        print("\n=== Fine-tuning QA Model ===")
        if not os.path.exists(DATASET_PATH):
            print(f"Dataset not found at {DATASET_PATH}. Create it first with --create-dataset")
            return
        
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            qa_dataset = json.load(f)
        
        model, tokenizer = fine_tune_qa_model(qa_dataset, model_name=args.model)
    
    # Run interactive mode if requested
    if args.interactive:
        print("\n=== Starting Interactive QA Mode ===")
        # Load model: use fine-tuned if available, otherwise use pre-trained
        model_to_load = MODEL_PATH if os.path.exists(MODEL_PATH) else args.model
        qa_pipeline, _, _ = initialize_qa_pipeline(model_to_load)
        
        # Load corpus for retrieving contexts
        corpus = load_corpus_from_jsonl(CORPUS_PATH)
        corpus_dict = {doc['doc_id']: doc['text'] for doc in corpus}
        
        interactive_qa_mode(qa_pipeline, es_engine, corpus_dict)
    
    # If no specific action was requested, show help
    if not (args.create_dataset or args.fine_tune or args.interactive):
        parser.print_help()

if __name__ == "__main__":
    main()