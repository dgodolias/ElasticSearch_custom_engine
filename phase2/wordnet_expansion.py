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
import itertools # Import itertools for combinations

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

# NEW function to get expandable words and their synonyms
def get_expandable_words_and_synonyms(query_text, max_synonyms=3, min_word_length=4):
    """
    Identifies expandable words (nouns/adjectives) in a query and their synonyms.

    Returns:
        tuple: (list of original tokens, list of tuples (word_index, original_word, [synonyms]))
    """
    tokens = word_tokenize(query_text.lower())
    pos_tags = pos_tag(tokens)
    stop_words = set(stopwords.words('english'))
    expandable_info = []

    for i, (word, tag) in enumerate(pos_tags):
        wordnet_pos = get_wordnet_pos(tag)
        if (word in stop_words or
            len(word) < min_word_length or
            wordnet_pos not in [wordnet.NOUN, wordnet.ADJ]):
            continue

        # Get synonyms
        word_synonyms = []
        # Try specific POS
        if wordnet_pos:
            for syn in wordnet.synsets(word, pos=wordnet_pos):
                for lemma in syn.lemmas():
                    syn_name = lemma.name().replace("_", " ")
                    if syn_name.lower() != word and syn_name not in word_synonyms:
                         word_synonyms.append(syn_name)
        # Try without specific POS if none found
        if not word_synonyms:
             all_syns = get_synonyms(word)
             for s in all_syns:
                 if s.lower() != word and s not in word_synonyms:
                     word_synonyms.append(s)

        # Limit synonyms and add to list if found
        if word_synonyms:
            selected_synonyms = word_synonyms[:max_synonyms]
            expandable_info.append((i, word, selected_synonyms))

    return tokens, expandable_info

# NEW function to generate query variations (replacing one word at a time)
def generate_query_variations(original_tokens, expandable_info):
    """
    Generates query variations by replacing one expandable word at a time with its synonyms.

    Args:
        original_tokens (list): The tokenized original query.
        expandable_info (list): List of tuples (word_index, original_word, [synonyms]).

    Returns:
        list: A list of query variation strings.
    """
    variations = set()
    # Add the original query first
    variations.add(" ".join(original_tokens))

    for index, original_word, synonyms in expandable_info:
        for synonym in synonyms:
            new_tokens = original_tokens[:] # Create a copy
            new_tokens[index] = synonym    # Replace the word
            variations.add(" ".join(new_tokens))

    return list(variations)

# MODIFIED: run_search_with_expanded_queries
def run_search_with_variations(queries_path, output_dir, k_values=[20, 30, 50]):
    """
    Run search with query variations and save aggregated results.
    """
    # Initialize ElasticSearch engine
    es_engine = ElasticSearchEngine(host=ES_HOST)

    # Check connection
    if not es_engine.check_connection():
        print("Δεν είναι δυνατή η σύνδεση με τον ElasticSearch server. Βεβαιωθείτε ότι ο server είναι σε λειτουργία.")
        return

    # Load original queries
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Το αρχείο {queries_path} δε βρέθηκε")
    with open(queries_path, 'r', encoding='utf-8') as f:
        original_queries = [json.loads(line) for line in f]

    # Define path for the variations file
    variations_output_path = os.path.join(output_dir, "query_variations.jsonl")
    # Clear the file if it exists from a previous run
    if os.path.exists(variations_output_path):
        print(f"Διαγραφή προηγούμενου αρχείου παραλλαγών: {variations_output_path}")
        os.remove(variations_output_path)

    # For each k value
    for k in k_values:
        print(f"\nΕκτέλεση αναζήτησης με παραλλαγές ερωτημάτων για τα top-{k} αποτελέσματα...")

        # This will store the final aggregated results for all queries for this k
        final_results_for_trec = {}

        # Process each original query
        for original_query_data in tqdm(original_queries, desc=f"Επεξεργασία ερωτημάτων (k={k})"):
            query_id = original_query_data.get("_id", "")
            original_text = original_query_data.get("text", "")

            if not query_id or not original_text:
                continue

            # 1. Get expandable words and synonyms
            original_tokens, expandable_info = get_expandable_words_and_synonyms(original_text)

            # 2. Generate variations
            query_variations = generate_query_variations(original_tokens, expandable_info)

            # --- START: Save variations to JSONL ---
            if k == k_values[0]: # Save variations only once (during the first k iteration)
                try:
                    with open(variations_output_path, 'a', encoding='utf-8') as var_f:
                        variation_data = {
                            "_id": query_id,
                            "original_text": original_text,
                            "variations": query_variations
                        }
                        var_f.write(json.dumps(variation_data, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"\nΣφάλμα κατά την αποθήκευση παραλλαγών για το query {query_id}: {e}")
            # --- END: Save variations to JSONL ---


            # Dictionary to store aggregated results for THIS query_id: {doc_id: max_score}
            aggregated_results_for_query = {}

            # 3. Run search for each variation
            for variation_text in query_variations:
                try:
                    search_results = es_engine.search(variation_text, k=k) # Returns list of hits
                    # 4. Aggregate results (keep max score)
                    for hit in search_results:
                        doc_id = hit["_source"]["doc_id"] # Assuming doc_id is stored in _source
                        score = hit["_score"]
                        # Keep the maximum score found for this document across all variations
                        aggregated_results_for_query[doc_id] = max(aggregated_results_for_query.get(doc_id, 0.0), score)
                except Exception as e:
                    print(f"\nΣφάλμα κατά την αναζήτηση για την παραλλαγή '{variation_text}' του query {query_id}: {e}")

            # 5. Convert aggregated results to sorted list for TREC format
            # Sort documents by their maximum score in descending order
            sorted_docs = sorted(aggregated_results_for_query.items(), key=lambda item: item[1], reverse=True)

            # Store the sorted list for this query_id
            final_results_for_trec[query_id] = sorted_docs # Store list of (doc_id, max_score) tuples

        # Save aggregated results in TREC format
        output_file = os.path.join(output_dir, f"results_wordnet_variations_top{k}.txt")
        run_name = f"covid_search_wordnet_variations_top{k}"

        # We need to adapt save_results_trec_format or do the formatting here
        print(f"Αποθήκευση {len(final_results_for_trec)} αποτελεσμάτων στο {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for query_id, sorted_doc_scores in final_results_for_trec.items():
                for rank, (doc_id, score) in enumerate(sorted_doc_scores, start=1):
                    # Format: query_id Q0 doc_id rank score run_name
                    f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n") # Format score

# MODIFIED: main function
def main():
    # Setup directories
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)

    # Download NLTK resources
    download_nltk_resources()

    # Run search with query variations
    run_search_with_variations(QUERIES_PATH, output_dir) # Use original queries path

    print("\nΗ διαδικασία ολοκληρώθηκε! Τα αποτελέσματα είναι έτοιμα για αξιολόγηση με το trec_eval.")

if __name__ == "__main__":
    main()