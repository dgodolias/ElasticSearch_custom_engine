import json
import os
import sys
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import time
from tqdm import tqdm

# Add parent directory to path to import from search_engine.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_engine import ElasticSearchEngine, QUERIES_PATH, ES_HOST, INDEX_NAME

# Download necessary NLTK resources if they're not already downloaded
def download_nltk_resources():
    """Download necessary NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading necessary NLTK resources...")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('stopwords')
    print("NLTK resources are ready.")

# Function to get synonyms from WordNet as provided in the instructions
def get_synonyms(word):
    """Get synonyms for a word from WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Replace underscores with spaces in multi-word synonyms
            synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)

def get_wordnet_pos(tag):
    """
    Map POS tag to first character used by WordNet.
    This helps in getting better results by searching for specific POS.
    """
    tag = tag[0].upper()
    tag_dict = {
        'J': wordnet.ADJ,        # Adjective
        'N': wordnet.NOUN,       # Noun 
        'V': wordnet.VERB,       # Verb
        'R': wordnet.ADV         # Adverb
    }
    return tag_dict.get(tag, None)

def expand_query_with_wordnet(query_text, max_synonyms=3, min_word_length=4):
    """
    Expand a query with synonyms from WordNet.
    
    - Only expands nouns and adjectives (based on POS tagging)
    - Ignores stopwords and words that are too short
    - Limits the number of synonyms per word
    - Filters synonyms by similarity threshold
    """
    # Tokenize and tag parts of speech
    tokens = word_tokenize(query_text.lower())
    pos_tags = pos_tag(tokens)
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    expanded_query = query_text
    
    # Collect all expanded terms to avoid duplicates
    all_synonyms = []
    
    # For each word in the query
    for word, tag in pos_tags:
        # Skip stopwords, short words, and words that are not nouns or adjectives
        wordnet_pos = get_wordnet_pos(tag)
        if (word in stop_words or 
            len(word) < min_word_length or 
            wordnet_pos not in [wordnet.NOUN, wordnet.ADJ]):
            continue
        
        # Get synonyms for this word
        word_synonyms = []
        
        # Try to get synonyms for this specific POS
        if wordnet_pos:
            for syn in wordnet.synsets(word, pos=wordnet_pos):
                for lemma in syn.lemmas():
                    # Skip if the lemma name is the same as the original word
                    if lemma.name().lower() != word:
                        # Replace underscores with spaces in multi-word synonyms
                        word_synonyms.append(lemma.name().replace("_", " "))
        
        # If no synonyms found with specific POS, try without it
        if not word_synonyms:
            word_synonyms = [s for s in get_synonyms(word) if s.lower() != word]
        
        # Limit the number of synonyms
        if word_synonyms:
            # Remove duplicates while preserving order
            unique_synonyms = []
            for syn in word_synonyms:
                if syn not in unique_synonyms:
                    unique_synonyms.append(syn)
            
            # Take up to max_synonyms
            selected_synonyms = unique_synonyms[:max_synonyms]
            
            # Add to all synonyms list
            all_synonyms.extend(selected_synonyms)
    
    # Append all unique synonyms to the original query
    if all_synonyms:
        expanded_query = f"{query_text} {' '.join(all_synonyms)}"
    
    return expanded_query

def expand_and_save_queries(queries_path, output_path):
    """
    Load queries from queries_path, expand them with WordNet synonyms,
    and save the expanded queries to output_path.
    """
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Το αρχείο {queries_path} δε βρέθηκε")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    expanded_queries = []
    original_queries = []
    
    print(f"Επέκταση ερωτημάτων με συνώνυμα από το WordNet...")
    
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = [json.loads(line) for line in f]
    
    for query in tqdm(queries, desc="Επέκταση ερωτημάτων"):
        query_id = query.get("_id", "")
        query_text = query.get("text", "")
        
        # Save original query
        original_queries.append({
            "_id": query_id,
            "text": query_text
        })
        
        # Expand query
        expanded_query_text = expand_query_with_wordnet(query_text)
        
        # Save expanded query
        expanded_queries.append({
            "_id": query_id,
            "text": expanded_query_text,
            "original_text": query_text  # Keep the original for comparison
        })
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for query in expanded_queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    print(f"Αποθηκεύτηκαν {len(expanded_queries)} διευρυμένα ερωτήματα στο αρχείο: {output_path}")
    
    # For demonstration, show a few examples of expansion
    print("\nΠαραδείγματα διεύρυνσης ερωτημάτων:")
    for i in range(min(3, len(expanded_queries))):
        print(f"Original: {original_queries[i]['text']}")
        print(f"Expanded: {expanded_queries[i]['text']}")
        print()
    
    return expanded_queries

def run_search_with_expanded_queries(expanded_queries_path, output_dir, k_values=[20, 30, 50]):
    """
    Run search with expanded queries and save results.
    """
    # Initialize ElasticSearch engine
    es_engine = ElasticSearchEngine(host=ES_HOST)
    
    # Check connection
    if not es_engine.check_connection():
        print("Δεν είναι δυνατή η σύνδεση με τον ElasticSearch server. Βεβαιωθείτε ότι ο server είναι σε λειτουργία.")
        return
    
    # Load expanded queries
    with open(expanded_queries_path, 'r', encoding='utf-8') as f:
        expanded_queries = [json.loads(line) for line in f]
    
    # For each k value
    for k in k_values:
        print(f"\nΕκτέλεση αναζήτησης με διευρυμένα ερωτήματα για τα top-{k} αποτελέσματα...")
        
        results = {}
        
        # Process each query
        for query in tqdm(expanded_queries, desc=f"Εκτέλεση διευρυμένων ερωτημάτων (k={k})"):
            query_id = query.get("_id", "")
            query_text = query.get("text", "")  # This is the expanded text
            
            # Search with the expanded query
            search_results = es_engine.search(query_text, k=k)
            results[query_id] = search_results
        
        # Save results in TREC format
        output_file = os.path.join(output_dir, f"results_wordnet_top{k}.txt")
        run_name = f"covid_search_wordnet_top{k}"
        
        es_engine.save_results_trec_format(results, output_file, run_name=run_name)

def main():
    # Setup directories
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Path for expanded queries
    expanded_queries_path = os.path.join(output_dir, "expanded_queries_wordnet.jsonl")
    
    # Expand and save queries
    expand_and_save_queries(QUERIES_PATH, expanded_queries_path)
    
    # Run search with expanded queries
    run_search_with_expanded_queries(expanded_queries_path, output_dir)
    
    print("\nΗ διαδικασία ολοκληρώθηκε! Τα αποτελέσματα είναι έτοιμα για αξιολόγηση με το trec_eval.")

if __name__ == "__main__":
    main()