# word2vec_expansion.py - Για την υλοποίηση της Φάσης 3
import json
import os
import sys
import time
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import gensim
from gensim.models import Word2Vec
import logging
import numpy as np

# Προσθήκη του γονικού καταλόγου στο path για να εισάγουμε από το search_engine.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_engine import ElasticSearchEngine, QUERIES_PATH, CORPUS_PATH, ES_HOST, INDEX_NAME

# Ρύθμιση του logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Κατέβασμα απαραίτητων NLTK πόρων αν δεν έχουν ήδη κατέβει
def download_nltk_resources():
    """Κατέβασμα απαραίτητων NLTK πόρων"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Κατέβασμα απαραίτητων NLTK πόρων...")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')
    print("Οι πόροι NLTK είναι έτοιμοι.")

# Φόρτωση του corpus από το jsonl αρχείο
def load_corpus_from_jsonl(corpus_path):
    """Φόρτωση και προεπεξεργασία του corpus για εκπαίδευση του Word2Vec"""
    print(f"Φόρτωση corpus από {corpus_path}...")
    documents = []
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Φόρτωση εγγράφων"):
            doc = json.loads(line)
            # Για το covid dataset
            if 'text' in doc:
                # Συνδυασμός τίτλου και κειμένου αν υπάρχουν και τα δύο
                text = ""
                if 'title' in doc:
                    text += doc['title'] + " "
                text += doc['text']
                documents.append(text)
            # Για το scidocs dataset (προσαρμόστε ανάλογα με τη δομή του dataset σας)
            elif 'paper_text' in doc:
                text = ""
                if 'title' in doc:
                    text += doc['title'] + " "
                text += doc['paper_text']
                documents.append(text)
    
    print(f"Φορτώθηκαν {len(documents)} έγγραφα.")
    return documents

# Προεπεξεργασία του corpus για εκπαίδευση
def preprocess_corpus(documents):
    """Προεπεξεργασία του corpus για εκπαίδευση του Word2Vec"""
    print("Προεπεξεργασία του corpus...")
    stop_words = set(stopwords.words('english'))
    
    tokenized_corpus = []
    for doc in tqdm(documents, desc="Tokenization & preprocessing"):
        # Tokenization και μετατροπή σε πεζά
        tokens = word_tokenize(doc.lower())
        # Αφαίρεση stopwords και σημείων στίξης
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        tokenized_corpus.append(filtered_tokens)
    
    return tokenized_corpus

# Εκπαίδευση του μοντέλου Word2Vec
def train_word2vec_model(tokenized_corpus, vector_size=100, window=5, min_count=5, 
                          workers=4, sg=1, epochs=5):
    """
    Εκπαίδευση του μοντέλου Word2Vec.
    
    Args:
        tokenized_corpus: Λίστα λιστών με tokens
        vector_size: Μέγεθος των διανυσμάτων λέξεων
        window: Μέγεθος παραθύρου για το context
        min_count: Ελάχιστος αριθμός εμφανίσεων λέξης
        workers: Αριθμός threads για παραλληλοποίηση
        sg: 1 για Skip-gram, 0 για CBOW
        epochs: Αριθμός εποχών εκπαίδευσης
    """
    print(f"Εκπαίδευση μοντέλου Word2Vec (αρχιτεκτονική: {'Skip-gram' if sg==1 else 'CBOW'}, "
          f"vector_size={vector_size}, window={window}, min_count={min_count})...")
    
    model = Word2Vec(sentences=tokenized_corpus, 
                      vector_size=vector_size, 
                      window=window, 
                      min_count=min_count, 
                      workers=workers, 
                      sg=sg,
                      epochs=epochs)
    
    print("Η εκπαίδευση του μοντέλου ολοκληρώθηκε.")
    return model

# Αποθήκευση και φόρτωση του μοντέλου
def save_model(model, model_path):
    """Αποθήκευση του μοντέλου"""
    model.save(model_path)
    print(f"Το μοντέλο αποθηκεύτηκε στο {model_path}")

def load_model(model_path):
    """Φόρτωση του μοντέλου"""
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
        print(f"Το μοντέλο φορτώθηκε από το {model_path}")
        return model
    else:
        print(f"Το μοντέλο δε βρέθηκε στο {model_path}")
        return None

# Εύρεση παρόμοιων λέξεων με το Word2Vec
def get_similar_words(model, word, topn=5, threshold=0.5):
    """
    Εύρεση παρόμοιων λέξεων με το Word2Vec
    
    Args:
        model: Το εκπαιδευμένο μοντέλο Word2Vec
        word: Η λέξη για την οποία αναζητούμε παρόμοιες
        topn: Μέγιστος αριθμός παρόμοιων λέξεων
        threshold: Κατώφλι ομοιότητας (0-1)
    
    Returns:
        Λίστα με παρόμοιες λέξεις που ξεπερνούν το κατώφλι
    """
    similar_words = []
    
    if word in model.wv:
        word_matches = model.wv.most_similar(word, topn=topn)
        # Φιλτράρισμα με βάση το κατώφλι
        similar_words = [(w, score) for w, score in word_matches if score >= threshold]
    
    return similar_words

# Αναγνώριση επεκτάσιμων λέξεων και εύρεση παρόμοιων από Word2Vec
def get_expandable_words_and_similar(query_text, model, topn=3, threshold=0.6, min_word_length=4):
    """
    Αναγνώριση σημαντικών λέξεων στο ερώτημα και εύρεση παρόμοιων λέξεων με Word2Vec
    
    Args:
        query_text: Το κείμενο του ερωτήματος
        model: Το εκπαιδευμένο μοντέλο Word2Vec
        topn: Μέγιστος αριθμός παρόμοιων λέξεων ανά λέξη
        threshold: Κατώφλι ομοιότητας (0-1)
        min_word_length: Ελάχιστο μήκος λέξης για να θεωρηθεί σημαντική
    
    Returns:
        tuple: (λίστα αρχικών tokens, λίστα με πληροφορίες επεκτάσιμων λέξεων)
    """
    tokens = word_tokenize(query_text.lower())
    pos_tags = pos_tag(tokens)
    stop_words = set(stopwords.words('english'))
    expandable_info = []
    
    for i, (word, tag) in enumerate(pos_tags):
        # Φιλτράρισμα λέξεων με βάση τα κριτήρια που θέσαμε
        if (word in stop_words or 
            len(word) < min_word_length or 
            not word.isalpha() or
            tag[0] not in ['N', 'J']):  # Μόνο ουσιαστικά και επίθετα
            continue
        
        # Εύρεση παρόμοιων λέξεων από το Word2Vec
        similar_words = []
        if word in model.wv:
            word_matches = model.wv.most_similar(word, topn=topn*2)  # Ζητάμε περισσότερες για να έχουμε εφεδρείες
            # Φιλτράρισμα με βάση το κατώφλι και αποφυγή διπλοτύπων
            for w, score in word_matches:
                if score >= threshold and w != word and w not in similar_words:
                    similar_words.append((w, score))
            
            # Περιορισμός στις top-n
            similar_words = similar_words[:topn]
            
            if similar_words:
                expandable_info.append((i, word, similar_words))
    
    return tokens, expandable_info

# Δημιουργία παραλλαγών ερωτημάτων
def generate_query_variations(original_tokens, expandable_info):
    """
    Δημιουργία παραλλαγών ερωτημάτων αντικαθιστώντας μία λέξη κάθε φορά με παρόμοιες
    
    Args:
        original_tokens: Λίστα με τα tokens του αρχικού ερωτήματος
        expandable_info: Λίστα με πληροφορίες για επεκτάσιμες λέξεις
    
    Returns:
        list: Λίστα με παραλλαγές ερωτημάτων (συμπεριλαμβανομένου του αρχικού)
    """
    variations = set()
    # Προσθήκη του αρχικού ερωτήματος
    variations.add(" ".join(original_tokens))
    
    for index, original_word, similar_words in expandable_info:
        for sim_word, score in similar_words:
            new_tokens = original_tokens.copy()
            new_tokens[index] = sim_word
            variations.add(" ".join(new_tokens))
    
    return list(variations)

# Εκτέλεση αναζήτησης με παραλλαγές ερωτημάτων
def run_search_with_variations(queries_path, model, output_dir, k_values=[20, 30, 50]):
    """
    Εκτέλεση αναζήτησης με παραλλαγές ερωτημάτων και αποθήκευση αποτελεσμάτων
    """
    # Αρχικοποίηση ElasticSearch
    es_engine = ElasticSearchEngine(host=ES_HOST)
    
    # Έλεγχος σύνδεσης
    if not es_engine.check_connection():
        print("Δεν είναι δυνατή η σύνδεση με τον ElasticSearch server. Βεβαιωθείτε ότι ο server είναι σε λειτουργία.")
        return
    
    # Φόρτωση αρχικών ερωτημάτων
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Το αρχείο {queries_path} δε βρέθηκε")
    
    with open(queries_path, 'r', encoding='utf-8') as f:
        original_queries = [json.loads(line) for line in f]
    
    # Ορισμός διαδρομής για το αρχείο παραλλαγών
    variations_output_path = os.path.join(output_dir, "query_variations_word2vec.jsonl")
    # Καθαρισμός του αρχείου αν υπάρχει από προηγούμενη εκτέλεση
    if os.path.exists(variations_output_path):
        print(f"Διαγραφή προηγούμενου αρχείου παραλλαγών: {variations_output_path}")
        os.remove(variations_output_path)
    
    # Για κάθε τιμή k
    for k in k_values:
        print(f"\nΕκτέλεση αναζήτησης με παραλλαγές ερωτημάτων για τα top-{k} αποτελέσματα...")
        
        # Αποθήκευση τελικών συγκεντρωτικών αποτελεσμάτων για όλα τα ερωτήματα
        final_results_for_trec = {}
        
        # Επεξεργασία κάθε αρχικού ερωτήματος
        for original_query_data in tqdm(original_queries, desc=f"Επεξεργασία ερωτημάτων (k={k})"):
            query_id = original_query_data.get("_id", "")
            original_text = original_query_data.get("text", "")
            
            if not query_id or not original_text:
                continue
            
            # 1. Εύρεση επεκτάσιμων λέξεων και παρόμοιων λέξεων
            original_tokens, expandable_info = get_expandable_words_and_similar(original_text, model)
            
            # 2. Δημιουργία παραλλαγών
            query_variations = generate_query_variations(original_tokens, expandable_info)
            
            # Αποθήκευση παραλλαγών σε JSONL (μόνο την πρώτη φορά)
            if k == k_values[0]:
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
            
            # Λεξικό για αποθήκευση συγκεντρωτικών αποτελεσμάτων για ΑΥΤΟ το query_id
            aggregated_results_for_query = {}
            
            # 3. Εκτέλεση αναζήτησης για κάθε παραλλαγή
            for variation_text in query_variations:
                try:
                    search_results = es_engine.search(variation_text, k=k)
                    # 4. Συγκέντρωση αποτελεσμάτων (κρατάμε max score)
                    for hit in search_results:
                        doc_id = hit["_source"]["doc_id"]
                        score = hit["_score"]
                        # Κρατάμε τη μέγιστη βαθμολογία για αυτό το έγγραφο
                        aggregated_results_for_query[doc_id] = max(aggregated_results_for_query.get(doc_id, 0.0), score)
                except Exception as e:
                    print(f"\nΣφάλμα κατά την αναζήτηση για την παραλλαγή '{variation_text}' του query {query_id}: {e}")
            
            # 5. Μετατροπή αποτελεσμάτων σε ταξινομημένη λίστα για μορφή TREC
            sorted_docs = sorted(aggregated_results_for_query.items(), key=lambda item: item[1], reverse=True)
            
            # Αποθήκευση της ταξινομημένης λίστας για αυτό το query_id
            final_results_for_trec[query_id] = sorted_docs
        
        # Αποθήκευση συγκεντρωτικών αποτελεσμάτων σε μορφή TREC
        output_file = os.path.join(output_dir, f"results_word2vec_variations_top{k}.txt")
        run_name = f"covid_search_word2vec_variations_top{k}"
        
        print(f"Αποθήκευση {len(final_results_for_trec)} αποτελεσμάτων στο {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for query_id, sorted_doc_scores in final_results_for_trec.items():
                for rank, (doc_id, score) in enumerate(sorted_doc_scores, start=1):
                    # Μορφή: query_id Q0 doc_id rank score run_name
                    f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

# Κύρια συνάρτηση
def main():
    # Ρύθμιση καταλόγων
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "word2vec_model.bin")
    
    # Κατέβασμα NLTK πόρων
    download_nltk_resources()
    
    # Έλεγχος αν υπάρχει ήδη εκπαιδευμένο μοντέλο
    model = load_model(model_path)
    
    if model is None:
        print("Δεν βρέθηκε εκπαιδευμένο μοντέλο. Εκκίνηση εκπαίδευσης...")
        
        # Φόρτωση και προεπεξεργασία του corpus
        documents = load_corpus_from_jsonl(CORPUS_PATH)
        tokenized_corpus = preprocess_corpus(documents)
        
        # Εκπαίδευση του μοντέλου Word2Vec
        # Χρησιμοποιείται Skip-gram (sg=1) που συνήθως δίνει καλύτερα αποτελέσματα για συνώνυμα
        model = train_word2vec_model(tokenized_corpus, 
                                     vector_size=100,  # Μέγεθος διανυσμάτων
                                     window=5,         # Μέγεθος παραθύρου context
                                     min_count=5,      # Ελάχιστος αριθμός εμφανίσεων
                                     workers=4,        # Αριθμός threads
                                     sg=1,             # 1 για Skip-gram, 0 για CBOW
                                     epochs=5)         # Αριθμός εποχών εκπαίδευσης
        
        # Αποθήκευση του μοντέλου
        save_model(model, model_path)
    
    # Έλεγχος λειτουργίας του μοντέλου με μερικά παραδείγματα
    print("\nΠαραδείγματα παρόμοιων λέξεων από το μοντέλο:")
    example_words = ["covid", "treatment", "disease", "vaccine", "health"]
    for word in example_words:
        if word in model.wv:
            similar_words = model.wv.most_similar(word, topn=5)
            print(f"\nTop 5 λέξεις παρόμοιες με '{word}':")
            for w, score in similar_words:
                print(f"  {w}: {score:.4f}")
        else:
            print(f"\nΗ λέξη '{word}' δεν βρέθηκε στο λεξικό του μοντέλου.")
    
    # Εκτέλεση αναζήτησης με τα επεκταμένα ερωτήματα
    print("\nΕκτέλεση αναζήτησης με επεκταμένα ερωτήματα από Word2Vec...")
    run_search_with_variations(QUERIES_PATH, model, output_dir)
    
    print("\nΗ διαδικασία ολοκληρώθηκε! Τα αποτελέσματα είναι έτοιμα για αξιολόγηση με το trec_eval.")

if __name__ == "__main__":
    main()