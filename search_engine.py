import json
import os
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
import ssl
import urllib3
from dotenv import load_dotenv
import time
import threading

# Φόρτωση μεταβλητών περιβάλλοντος από το .env αρχείο
load_dotenv()

# Απενεργοποίηση προειδοποιήσεων για μη επαληθευμένα SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Διαδρομές για τα αρχεία της συλλογής
CORPUS_PATH = "covid_datasets/corpus.jsonl"
QUERIES_PATH = "covid_datasets/queries.jsonl"
QRELS_PATH = "covid_datasets/qrels/test.tsv"

# Ρυθμίσεις ElasticSearch
INDEX_NAME = "covid_search"
ES_HOST = "https://localhost:9200"  # Χρήση HTTPS αντί για HTTP
ES_USER = os.getenv("ES_USER", "")  # Αφήνουμε κενό για σύνδεση χωρίς auth
ES_PASS = os.getenv("ES_PASS", "")
ES_TIMEOUT = 30  # Timeout σε δευτερόλεπτα

class ElasticSearchEngine:
    def __init__(self, host=ES_HOST):
        """Αρχικοποίηση του ElasticSearch client"""
        # Δημιουργία SSL context για ασφαλή σύνδεση
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Σύνδεση με διαπιστευτήρια (βασική στρατηγική)
        if ES_USER and ES_PASS:
            try:
                self.es = Elasticsearch(
                    host,
                    basic_auth=(ES_USER, ES_PASS),
                    verify_certs=False,
                    ssl_context=ssl_context,
                    request_timeout=ES_TIMEOUT
                )
                print("Επιτυχής σύνδεση με τον ElasticSearch με διαπιστευτήρια")
            except Exception as e:
                print(f"Αποτυχία σύνδεσης με διαπιστευτήρια: {e}")
                raise ConnectionError("Δεν ήταν δυνατή η σύνδεση με τον ElasticSearch server")
        else:
            # Εναλλακτική προσπάθεια σύνδεσης χωρίς διαπιστευτήρια
            try:
                self.es = Elasticsearch(
                    host,
                    verify_certs=False,
                    ssl_context=ssl_context,
                    request_timeout=ES_TIMEOUT
                )
                print("Επιτυχής σύνδεση με τον ElasticSearch με SSL αλλά χωρίς διαπιστευτήρια")
            except Exception as e:
                print(f"Αποτυχία σύνδεσης: {e}")
                raise ConnectionError("Δεν ήταν δυνατή η σύνδεση με τον ElasticSearch server")
        
        self.index_name = INDEX_NAME
    
    def check_connection(self):
        """Έλεγχος σύνδεσης με τον ElasticSearch server"""
        try:
            info = self.es.info()
            print(f"Έκδοση ElasticSearch: {info.get('version', {}).get('number', 'unknown')}")
            return True
        except Exception as e:
            print(f"Σφάλμα σύνδεσης με τον ElasticSearch: {e}")
            return False
    
    def count_corpus_documents(self, corpus_path):
        """Μέτρηση του αριθμού των εγγράφων στο αρχείο corpus"""
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Το αρχείο {corpus_path} δε βρέθηκε")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            # Μέτρηση των γραμμών στο αρχείο
            doc_count = sum(1 for _ in f)
        
        return doc_count
    
    def index_exists_and_complete(self, corpus_path):
        """Έλεγχος αν το ευρετήριο υπάρχει και περιέχει όλα τα έγγραφα του corpus"""
        if not self.es.indices.exists(index=self.index_name):
            return False
        
        # Μέτρηση εγγράφων στο ευρετήριο
        count_result = self.es.count(index=self.index_name)
        indexed_doc_count = count_result.get('count', 0)
        
        # Μέτρηση εγγράφων στο αρχείο corpus
        corpus_doc_count = self.count_corpus_documents(corpus_path)
        
        if indexed_doc_count > 0:
            print(f"Το ευρετήριο {self.index_name} υπάρχει ήδη και περιέχει {indexed_doc_count} έγγραφα από σύνολο {corpus_doc_count}.")
            
            # Έλεγχος αν όλα τα έγγραφα έχουν εισαχθεί
            if indexed_doc_count >= corpus_doc_count:
                print("Το ευρετήριο περιέχει όλα τα έγγραφα του corpus!")
                return True
            else:
                print(f"Το ευρετήριο είναι ελλιπές. Λείπουν {corpus_doc_count - indexed_doc_count} έγγραφα.")
                # Προτείνουμε επανεισαγωγή
                return False
        return False
    
    def create_index(self, delete_if_exists=True):
        """Δημιουργία ευρετηρίου με βελτιστοποιημένες ρυθμίσεις"""
        if delete_if_exists and self.es.indices.exists(index=self.index_name):
            print(f"Διαγραφή υπάρχοντος ευρετηρίου: {self.index_name}")
            self.es.indices.delete(index=self.index_name)
        
        # Βελτιστοποιημένες ρυθμίσεις του ευρετηρίου
        settings = {
            "settings": {
                "analysis": {
                    "filter": {
                        "english_stop": {
                            "type": "stop",
                            "stopwords": "_english_"
                        },
                        "english_stemmer": {
                            "type": "stemmer",
                            "language": "english"
                        },
                        "english_possessive_stemmer": {
                            "type": "stemmer",
                            "language": "possessive_english"
                        },
                        # Φίλτρο για ιατρικούς όρους - θα διατηρήσει αντί να κόψει κοινά ιατρικά προθέματα
                        "medical_prefix_filter": {
                            "type": "pattern_replace",
                            "pattern": "^(covid|sars|corona|virus|pneumonia)",
                            "replacement": "$1"
                        }
                    },
                    "analyzer": {
                        "custom_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "english_possessive_stemmer",
                                "english_stop",
                                "english_stemmer",
                                "medical_prefix_filter"
                            ]
                        }
                    }
                },
                "similarity": {
                    "custom_similarity": {
                        "type": "BM25",
                        # Βελτιστοποιημένες παράμετροι για ιατρικά κείμενα
                        "b": 0.5,         # Μειώνουμε το b για να μειωθεί η επιρροή του μήκους εγγράφου
                        "k1": 1.6         # Αυξάνουμε το k1 για καλύτερο χειρισμό επαναλαμβανόμενων όρων
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "custom_analyzer",
                        "similarity": "custom_similarity",
                        "boost": 3.0      # Δίνουμε μεγαλύτερο βάρος στον τίτλο
                    },
                    "abstract": {
                        "type": "text",
                        "analyzer": "custom_analyzer",
                        "similarity": "custom_similarity",
                        "boost": 2.0      # Δίνουμε μεσαίο βάρος στην περίληψη
                    },
                    "body_text": {
                        "type": "text",
                        "analyzer": "custom_analyzer",
                        "similarity": "custom_similarity",
                        "boost": 1.0      # Κανονικό βάρος στο κυρίως κείμενο
                    },
                    "doc_id": {
                        "type": "keyword"
                    }
                }
            }
        }
        
        print(f"Δημιουργία ευρετηρίου: {self.index_name}")
        self.es.indices.create(index=self.index_name, body=settings)
        print("Το ευρετήριο δημιουργήθηκε επιτυχώς!")
    
    def _generate_bulk_actions(self, corpus_path):
        """Generator function to yield bulk actions for indexing."""
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Το αρχείο {corpus_path} δε βρέθηκε")
            
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                yield {
                    "_index": self.index_name,
                    "_source": {
                        "title": doc.get("title", ""),
                        "abstract": doc.get("abstract", ""),
                        "body_text": self._extract_body_text(doc),
                        "doc_id": doc.get("_id", "")
                    }
                }

    def index_documents(self, corpus_path, chunk_size=500, max_chunk_bytes=100*1024*1024, thread_count=8):
        """Εισαγωγή εγγράφων στο ευρετήριο χρησιμοποιώντας το parallel_bulk helper."""
        print(f"Εισαγωγή εγγράφων από το αρχείο: {corpus_path} χρησιμοποιώντας parallel_bulk")
        
        # Μέτρηση του συνολικού αριθμού εγγράφων για το tqdm
        total_docs = self.count_corpus_documents(corpus_path)
        
        # Χρήση του parallel_bulk για ταχύτερη εισαγωγή με threads
        progress = tqdm(unit="docs", total=total_docs, desc="Εισαγωγή εγγράφων (parallel_bulk)")
        success_count = 0
        fail_count = 0
        
        try:
            # Το parallel_bulk επιστρέφει ένα generator με τα αποτελέσματα
            for success, info in helpers.parallel_bulk(
                client=self.es,
                actions=self._generate_bulk_actions(corpus_path),
                thread_count=thread_count, # Αριθμός threads
                chunk_size=chunk_size, # Αριθμός εγγράφων ανά chunk
                max_chunk_bytes=max_chunk_bytes, # Μέγιστο μέγεθος chunk σε bytes
                raise_on_error=False, # Συνέχεια ακόμα και αν υπάρχουν σφάλματα
                raise_on_exception=False # Συνέχεια ακόμα και αν υπάρχουν exceptions
            ):
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"Αποτυχία εισαγωγής εγγράφου: {info}") # Εκτύπωση σφάλματος
                progress.update(1) # Ενημέρωση του progress bar
                
        except Exception as e:
            print(f"Προέκυψε σφάλμα κατά τη διάρκεια του parallel_bulk: {e}")
        finally:
            progress.close() # Κλείσιμο του progress bar

        # Ανανέωση του ευρετηρίου για να είναι διαθέσιμα τα έγγραφα για αναζήτηση
        print("Ανανέωση ευρετηρίου...")
        self.es.indices.refresh(index=self.index_name)
        
        print(f"Ολοκληρώθηκε η εισαγωγή. Επιτυχίες: {success_count}, Αποτυχίες: {fail_count}")
        if fail_count > 0:
             print("Υπήρξαν αποτυχίες κατά την εισαγωγή. Ελέγξτε τα παραπάνω μηνύματα.")

    def _extract_body_text(self, doc):
        """Εξαγωγή του κειμένου από την ενότητα body_text"""
        body_text = ""
        if "body_text" in doc and isinstance(doc["body_text"], list):
            body_text = " ".join([entry.get("text", "") for entry in doc["body_text"]])
        return body_text
    
    def search(self, query_text, k=20):
        """Αναζήτηση με βάση το κείμενο του ερωτήματος - Βελτιωμένη έκδοση"""
        query = {
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["title^3", "abstract^2", "body_text^1"],
                    "type": "cross_fields",  # Αναζήτηση σε όλα τα πεδία ταυτόχρονα
                    "tie_breaker": 0.3,      # Αύξηση της επιρροής των λιγότερο σημαντικών πεδίων
                    "operator": "or",
                    "minimum_should_match": "70%"  # Τουλάχιστον 70% των όρων πρέπει να ταιριάζουν
                }
            },
            "_source": ["doc_id", "title"],
            "size": k
        }
        
        result = self.es.search(index=self.index_name, body=query)
        return result["hits"]["hits"]
    
    def search_all_queries(self, queries_path, k=20):
        """Αναζήτηση για όλα τα ερωτήματα του dataset"""
        if not os.path.exists(queries_path):
            raise FileNotFoundError(f"Το αρχείο {queries_path} δε βρέθηκε")
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            queries = [json.loads(line) for line in f]
        
        results = {}
        for query in tqdm(queries, desc="Εκτέλεση ερωτημάτων"):
            query_id = query.get("_id", "")
            query_text = query.get("text", "")
            
            search_results = self.search(query_text, k=k)
            results[query_id] = search_results
        
        return results
    
    def save_results_trec_format(self, results, output_file, run_name="covid_search"):
        """Αποθήκευση των αποτελεσμάτων σε μορφή TREC για αξιολόγηση με το trec_eval"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for query_id, hits in results.items():
                for rank, hit in enumerate(hits, start=1):
                    doc_id = hit["_source"]["doc_id"]
                    score = hit["_score"]
                    # Format: query_id Q0 doc_id rank score run_name
                    f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_name}\n")
        
        print(f"Τα αποτελέσματα αποθηκεύτηκαν στο αρχείο: {output_file}")


def main():
    # Δημιουργία φακέλου αποτελεσμάτων αν δεν υπάρχει
    os.makedirs("results", exist_ok=True)
    
    # Αρχικοποίηση του ElasticSearch engine
    es_engine = ElasticSearchEngine()
    
    # Έλεγχος σύνδεσης
    if not es_engine.check_connection():
        print("Δεν είναι δυνατή η σύνδεση με τον ElasticSearch server. Βεβαιωθείτε ότι ο server είναι σε λειτουργία.")
        return
    
    # Έλεγχος αν το ευρετήριο υπάρχει ήδη και περιέχει έγγραφα
    if not es_engine.index_exists_and_complete(CORPUS_PATH):
        # Δημιουργία ευρετηρίου μόνο αν δεν υπάρχει ή είναι ελλιπές
        print("Το ευρετήριο δεν υπάρχει ή είναι ελλιπές. Δημιουργία/Επανεισαγωγή...")
        es_engine.create_index(delete_if_exists=True)
        
        # Εισαγωγή εγγράφων στο ευρετήριο με parallel_bulk
        es_engine.index_documents(CORPUS_PATH, thread_count=8)  # Αύξηση των threads για ταχύτερη εισαγωγή
    else:
        print("Το ευρετήριο υπάρχει και είναι πλήρες. Παράλειψη δημιουργίας/εισαγωγής.")

    # Εκτέλεση αναζήτησης για όλα τα ερωτήματα
    for k in [20, 30, 50]:
        print(f"\nΕκτέλεση αναζήτησης για τα top-{k} αποτελέσματα...")
        results = es_engine.search_all_queries(QUERIES_PATH, k=k)
        output_file = f"results/results_top{k}.txt"
        es_engine.save_results_trec_format(results, output_file, run_name=f"covid_search_top{k}")


if __name__ == "__main__":
    main()