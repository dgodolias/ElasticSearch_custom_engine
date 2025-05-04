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
from nltk.stem import PorterStemmer # <-- Προσθήκη import
import gensim
from gensim.models import Word2Vec
import logging
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

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
def get_expandable_words_and_similar(query_text, model, topn=3, threshold=0.8, min_word_length=4):
    """
    Αναγνώριση σημαντικών λέξεων στο ερώτημα και εύρεση παρόμοιων λέξεων με Word2Vec,
    εξαιρώντας τις ομόρριζες λέξεις.

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
    stemmer = PorterStemmer() # <-- Δημιουργία stemmer
    expandable_info = []

    for i, (word, tag) in enumerate(pos_tags):
        # Φιλτράρισμα λέξεων με βάση τα κριτήρια που θέσαμε
        if (word in stop_words or
            len(word) < min_word_length or
            not word.isalpha() or
            tag[0] not in ['N', 'J']):  # Μόνο ουσιαστικά και επίθετα
            continue

        # Εύρεση παρόμοιων λέξεων από το Word2Vec
        similar_words_filtered = []
        accepted_stems = set() # <-- Set για να παρακολουθούμε τα stems που έχουμε ήδη δεχτεί
        original_word_stem = stemmer.stem(word) # <-- Stem της αρχικής λέξης
        accepted_stems.add(original_word_stem)

        if word in model.wv:
            try:
                # Ζητάμε περισσότερες για να έχουμε περιθώριο μετά το φιλτράρισμα
                word_matches = model.wv.most_similar(word, topn=topn * 3)

                # Φιλτράρισμα με βάση το κατώφλι, αποφυγή διπλοτύπων και ομόρριζων
                for w, score in word_matches:
                    if score < threshold or w == word or not w.isalpha():
                        continue

                    current_stem = stemmer.stem(w)

                    # Έλεγχος αν το stem είναι ίδιο με της αρχικής λέξης ή με κάποιο ήδη αποδεκτό stem
                    if current_stem in accepted_stems:
                        continue

                    # Αν περάσει τους ελέγχους, προσθέτουμε τη λέξη και το stem της
                    similar_words_filtered.append((w, score))
                    accepted_stems.add(current_stem)

                    # Σταματάμε αν έχουμε βρει αρκετές (topn)
                    if len(similar_words_filtered) >= topn:
                        break

            except KeyError:
                 # Η λέξη μπορεί να υπάρχει στο wv αλλά να μην έχει neighbors (σπάνιο)
                 pass

            if similar_words_filtered:
                expandable_info.append((i, word, similar_words_filtered))

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
            for query_id, sorted_doc_scores in sorted(final_results_for_trec.items()):
                for rank, (doc_id, score) in enumerate(sorted_doc_scores, start=1):
                    # Μορφή: query_id Q0 doc_id rank score run_name
                    f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

# Νέα συνάρτηση αξιολόγησης του μοντέλου Word2Vec για P@5
def evaluate_word2vec_model(model, evaluation_words=None):
    """
    Αξιολόγηση της ποιότητας του μοντέλου Word2Vec με βάση το P@5
    
    Χρησιμοποιεί προσομοίωση της μετρικής Precision@5 μετρώντας πόσο καλά το μοντέλο
    εντοπίζει σχετικές αλλά διαφορετικές λέξεις στις πρώτες 5 θέσεις.
    
    Args:
        model: Εκπαιδευμένο μοντέλο Word2Vec
        evaluation_words: Λίστα λέξεων για αξιολόγηση, ή None για προεπιλεγμένες λέξεις

    Returns:
        float: Βαθμολογία αξιολόγησης του μοντέλου ως μέσο P@5 (υψηλότερη είναι καλύτερη)
    """
    if evaluation_words is None:
        # Λέξεις σχετικές με ιατρικά/covid θέματα (προσαρμόστε ανάλογα με το dataset)
        evaluation_words = ["covid", "virus", "disease", "treatment", "patient", 
                           "health", "symptom", "study", "infection", "research"]

    stemmer = PorterStemmer()
    total_p5_score = 0
    evaluated_words = 0

    for word in evaluation_words:
        if word not in model.wv:
            continue

        evaluated_words += 1
        
        # Μέτρηση Precision@5 - πόσες από τις πρώτες 5 λέξεις είναι διαφορετικές αλλά σχετικές
        original_stem = stemmer.stem(word)
        accepted_stems = {original_stem}
        p5_hits = 0
        
        try:
            # Ζητάμε μόνο τις top-5 λέξεις (+ επιπλέον για περιθώριο αν κάποιες φιλτραριστούν)
            similar_words = model.wv.most_similar(word, topn=10)
            examined_count = 0
            
            for sim_word, sim_score in similar_words:
                # Φιλτράρισμα μη αλφαβητικών και παρόμοιων με χαμηλό σκορ
                if not sim_word.isalpha() or sim_score < 0.6:
                    continue
                    
                current_stem = stemmer.stem(sim_word)
                
                # Αν το stem είναι διαφορετικό, θεωρείται επιτυχία για το P@5
                if current_stem not in accepted_stems:
                    p5_hits += 1
                    accepted_stems.add(current_stem)
                
                # Μετράμε μόνο τις πρώτες 5 έγκυρες λέξεις
                examined_count += 1
                if examined_count >= 5:
                    break
            
            # Υπολογισμός του P@5 για αυτή τη λέξη
            p5_score = p5_hits / 5.0
            total_p5_score += p5_score
            
        except Exception as e:
            # Σε περίπτωση σφάλματος, αυτή η λέξη δεν συνεισφέρει στο σκορ
            pass
        
    # Επιστροφή μέσου όρου P@5 ή 0 αν καμία λέξη δεν αξιολογήθηκε
    return total_p5_score / max(1, evaluated_words)

# Συνάρτηση βελτιστοποίησης παραμέτρων του Word2Vec
def optimize_word2vec_parameters(tokenized_corpus, output_dir, test_params=False):
    """
    Βελτιστοποίηση παραμέτρων του Word2Vec με δοκιμή διαφορετικών συνδυασμών
    
    Args:
        tokenized_corpus: Προεπεξεργασμένο corpus για εκπαίδευση
        output_dir: Κατάλογος για αποθήκευση αποτελεσμάτων
        test_params: Αν True, δοκιμάζει μόνο λίγες τιμές για σύντομο τεστ
    
    Returns:
        tuple: (Βέλτιστο μοντέλο, λεξικό με τις βέλτιστες παραμέτρους)
    """
    print("\nΒελτιστοποίηση παραμέτρων Word2Vec...")
    
    # Ορισμός παραμέτρων προς βελτιστοποίηση
    if test_params:
        # Περιορισμένες τιμές για γρήγορο τεστ
        vector_sizes = [50, 100] 
        window_sizes = [3, 5]
        sg_values = [0, 1]  # 0: CBOW, 1: Skip-gram
        epochs_values = [2]
    else:
        # Πλήρες σύνολο τιμών για εύρεση βέλτιστων παραμέτρων
        vector_sizes = [50, 100, 200, 300] 
        window_sizes = [3, 5, 7, 10]
        sg_values = [0, 1]  # 0: CBOW, 1: Skip-gram
        epochs_values = [3, 5]
    
    # Σταθερές παράμετροι
    min_count = 5
    workers = 4
    
    # Αποθήκευση αποτελεσμάτων
    results = []
    best_score = -1
    best_model = None
    best_params = None
    
    total_combinations = len(vector_sizes) * len(window_sizes) * len(sg_values) * len(epochs_values)
    print(f"Δοκιμή {total_combinations} συνδυασμών παραμέτρων...")
    
    # Αντιγραφή του corpus για αποφυγή προβλημάτων
    train_data = [tokens.copy() for tokens in tokenized_corpus]
    
    # Δοκιμή όλων των συνδυασμών παραμέτρων
    for vector_size, window, sg, epochs in tqdm(
        product(vector_sizes, window_sizes, sg_values, epochs_values), 
        total=total_combinations,
        desc="Βελτιστοποίηση παραμέτρων"
    ):
        try:
            # Εκπαίδευση μοντέλου με τις τρέχουσες παραμέτρους
            model = Word2Vec(
                sentences=train_data,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=workers,
                sg=sg,
                epochs=epochs
            )
            
            # Αξιολόγηση μοντέλου
            score = evaluate_word2vec_model(model)
            
            # Αποθήκευση αποτελεσμάτων
            param_results = {
                'vector_size': vector_size,
                'window': window,
                'sg': sg,
                'epochs': epochs,
                'architecture': 'Skip-gram' if sg == 1 else 'CBOW',
                'score': score
            }
            results.append(param_results)
            
            # Ενημέρωση του καλύτερου μοντέλου αν είναι απαραίτητο
            if score > best_score:
                best_score = score
                best_model = model
                best_params = param_results.copy()
                
            print(f"Παράμετροι: vector_size={vector_size}, window={window}, sg={sg} ({param_results['architecture']}), epochs={epochs} → Βαθμολογία: {score:.4f}")
            
        except Exception as e:
            print(f"Σφάλμα κατά την εκπαίδευση του μοντέλου με παραμέτρους vector_size={vector_size}, window={window}, sg={sg}, epochs={epochs}: {e}")
    
    # Αποθήκευση αποτελεσμάτων σε αρχείο
    results_path = os.path.join(output_dir, "word2vec_parameter_optimization_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Δημιουργία διαγράμματος για CBOW vs Skip-gram
    plt.figure(figsize=(12, 6))
    
    # Οργάνωση αποτελεσμάτων ανά αρχιτεκτονική
    cbow_results = [r for r in results if r['sg'] == 0]
    skipgram_results = [r for r in results if r['sg'] == 1]
    
    # Δημιουργία scatter plot για κάθε αρχιτεκτονική
    plt.scatter(
        [r['vector_size'] for r in cbow_results], 
        [r['score'] for r in cbow_results],
        label='CBOW', marker='o', alpha=0.7
    )
    plt.scatter(
        [r['vector_size'] for r in skipgram_results], 
        [r['score'] for r in skipgram_results],
        label='Skip-gram', marker='x', alpha=0.7
    )
    
    plt.title('Επίδοση Μοντέλων Word2Vec με Διαφορετικές Παραμέτρους')
    plt.xlabel('Μέγεθος Διανύσματος (vector_size)')
    plt.ylabel('Βαθμολογία Αξιολόγησης')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Σημείωση του καλύτερου μοντέλου στο διάγραμμα
    if best_params:
        plt.annotate(
            f"Βέλτιστο: {best_params['architecture']}, VS={best_params['vector_size']}, W={best_params['window']}",
            xy=(best_params['vector_size'], best_params['score']),
            xytext=(10, -20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5')
        )
    
    # Αποθήκευση διαγράμματος
    chart_path = os.path.join(output_dir, "word2vec_optimization_chart.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    
    print(f"\nΒέλτιστες παράμετροι:")
    print(f"Αρχιτεκτονική: {best_params['architecture']} (sg={best_params['sg']})")
    print(f"Μέγεθος διανύσματος: {best_params['vector_size']}")
    print(f"Μέγεθος παραθύρου: {best_params['window']}")
    print(f"Εποχές: {best_params['epochs']}")
    print(f"Βαθμολογία: {best_params['score']:.4f}")
    print(f"Αποτελέσματα αποθηκεύτηκαν στο {results_path}")
    print(f"Διάγραμμα αποθηκεύτηκε στο {chart_path}")
    
    return best_model, best_params

# Κύρια συνάρτηση
def main():
    # Ρύθμιση καταλόγων
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "word2vec_model.bin")
    optimized_model_path = os.path.join(model_dir, "word2vec_model_optimized.bin")
    
    # Κατέβασμα NLTK πόρων
    download_nltk_resources()
    
    # Φόρτωση δεδομένων και προεπεξεργασία (αυτό χρειάζεται είτε για εκπαίδευση είτε για βελτιστοποίηση)
    print("Φόρτωση και προεπεξεργασία του corpus...")
    documents = load_corpus_from_jsonl(CORPUS_PATH)
    tokenized_corpus = preprocess_corpus(documents)
    
    # Προσθήκη παραμέτρων γραμμής εντολών για βελτιστοποίηση
    import argparse
    parser = argparse.ArgumentParser(description='Word2Vec Expansion and Optimization')
    parser.add_argument('--optimize', action='store_true', help='Perform parameter optimization')
    parser.add_argument('--test-params', action='store_true', help='Test only a few parameter combinations (faster)')
    parser.add_argument('--use-optimized', action='store_true', help='Use the optimized model if available')
    args = parser.parse_args()
    
    if args.optimize:
        print("\n=== Εκτέλεση βελτιστοποίησης παραμέτρων Word2Vec ===")
        best_model, best_params = optimize_word2vec_parameters(tokenized_corpus, output_dir, args.test_params)
        
        if best_model:
            # Αποθήκευση του βέλτιστου μοντέλου
            save_model(best_model, optimized_model_path)
            
            # Αποθήκευση των βέλτιστων παραμέτρων
            params_path = os.path.join(model_dir, "best_params.json")
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(best_params, f, ensure_ascii=False, indent=2)
            
            model = best_model
        else:
            print("Η βελτιστοποίηση δεν βρήκε έγκυρο μοντέλο. Προσπάθεια φόρτωσης ή εκπαίδευσης βασικού μοντέλου.")
            model = load_model(model_path)
    elif args.use_optimized and os.path.exists(optimized_model_path):
        # Φόρτωση του βελτιστοποιημένου μοντέλου αν υπάρχει και ζητήθηκε
        print("\nΦόρτωση βελτιστοποιημένου μοντέλου...")
        model = load_model(optimized_model_path)
        
        # Ανάκτηση και εμφάνιση των βέλτιστων παραμέτρων
        params_path = os.path.join(model_dir, "best_params.json")
        if os.path.exists(params_path):
            with open(params_path, 'r', encoding='utf-8') as f:
                best_params = json.load(f)
            print(f"Βέλτιστες παράμετροι:")
            print(f"Αρχιτεκτονική: {best_params['architecture']} (sg={best_params['sg']})")
            print(f"Μέγεθος διανύσματος: {best_params['vector_size']}")
            print(f"Μέγεθος παραθύρου: {best_params['window']}")
            print(f"Εποχές: {best_params['epochs']}")
            print(f"Βαθμολογία: {best_params['score']:.4f}")
    else:
        # Κανονική ροή εκτέλεσης: Έλεγχος για υπάρχον μοντέλο ή εκπαίδευση νέου
        model = load_model(model_path)
    
    # Αν δεν έχουμε μοντέλο ακόμα, εκπαίδευση με προεπιλεγμένες παραμέτρους
    if model is None:
        print("Δεν βρέθηκε εκπαιδευμένο μοντέλο. Εκκίνηση εκπαίδευσης με προεπιλεγμένες παραμέτρους...")
        
        # Εκπαίδευση του μοντέλου Word2Vec με προεπιλεγμένες παραμέτρους
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
    print("\nΠαραδείγματα παρόμοιων λέξεων από το μοντέλο (μετά το φιλτράρισμα ομόρριζων):")
    example_words = ["origin", "covid", "treatment", "disease", "vaccine", "health"]
    stemmer = PorterStemmer() # <-- Initialize stemmer here for example printing
    topn_examples = 5

    for word in example_words:
        if word in model.wv:
            original_word_stem = stemmer.stem(word)
            accepted_stems = {original_word_stem}
            filtered_similar_words = []
            try:
                # Get more initially to allow for filtering
                raw_similar_words = model.wv.most_similar(word, topn=topn_examples * 3)

                for sim_word, score in raw_similar_words:
                    if not sim_word.isalpha(): # Skip non-alphabetic words
                        continue

                    current_stem = stemmer.stem(sim_word)

                    if current_stem not in accepted_stems:
                        filtered_similar_words.append((sim_word, score))
                        accepted_stems.add(current_stem)
                        if len(filtered_similar_words) >= topn_examples:
                            break # Stop when we have enough filtered words

                print(f"\nTop {len(filtered_similar_words)} λέξεις παρόμοιες με '{word}' (μετά το φιλτράρισμα):")
                if filtered_similar_words:
                    for w, score in filtered_similar_words:
                        print(f"  {w}: {score:.4f}")
                else:
                    print("  (Δεν βρέθηκαν μη ομόρριζες παρόμοιες λέξεις)")

            except KeyError:
                print(f"\nΗ λέξη '{word}' υπάρχει στο λεξιλόγιο αλλά δεν έχει παρόμοιες λέξεις.")
        else:
            print(f"\nΗ λέξη '{word}' δεν βρέθηκε στο λεξικό του μοντέλου.")

    # Εκτέλεση αναζήτησης με τα επεκταμένα ερωτήματα
    print("\nΕκτέλεση αναζήτησης με επεκταμένα ερωτήματα από Word2Vec...")
    run_search_with_variations(QUERIES_PATH, model, output_dir)
    
    print("\nΗ διαδικασία ολοκληρώθηκε! Τα αποτελέσματα είναι έτοιμα για αξιολόγηση με το trec_eval.")

if __name__ == "__main__":
    main()