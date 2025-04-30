import json
import os
from elasticsearch import Elasticsearch
from tqdm import tqdm
import ssl
import urllib3
from dotenv import load_dotenv
import time

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
        """Δημιουργία ευρετηρίου"""
        if delete_if_exists and self.es.indices.exists(index=self.index_name):
            print(f"Διαγραφή υπάρχοντος ευρετηρίου: {self.index_name}")
            self.es.indices.delete(index=self.index_name)
        
        # Ορισμός των ρυθμίσεων του ευρετηρίου
        settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "custom_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                },
                "similarity": {
                    "custom_similarity": {
                        "type": "BM25",
                        "b": 0.75,
                        "k1": 1.2
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "custom_analyzer",
                        "similarity": "custom_similarity"
                    },
                    "abstract": {
                        "type": "text",
                        "analyzer": "custom_analyzer",
                        "similarity": "custom_similarity"
                    },
                    "body_text": {
                        "type": "text",
                        "analyzer": "custom_analyzer",
                        "similarity": "custom_similarity"
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
    
    def index_documents(self, corpus_path):
        """Εισαγωγή εγγράφων στο ευρετήριο"""
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Το αρχείο {corpus_path} δε βρέθηκε")
        
        print(f"Εισαγωγή εγγράφων από το αρχείο: {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = [json.loads(line) for line in f]
        
        # Εισαγωγή των εγγράφων στο ευρετήριο
        count = 0
        for doc in tqdm(corpus, desc="Εισαγωγή εγγράφων"):
            document = {
                "title": doc.get("title", ""),
                "abstract": doc.get("abstract", ""),
                "body_text": self._extract_body_text(doc),
                "doc_id": doc.get("_id", "")
            }
            
            self.es.index(index=self.index_name, document=document)
            count += 1
        
        # Ανανέωση του ευρετηρίου για να είναι διαθέσιμα τα έγγραφα για αναζήτηση
        self.es.indices.refresh(index=self.index_name)
        print(f"Εισήχθησαν επιτυχώς {count} έγγραφα στο ευρετήριο!")
    
    def _extract_body_text(self, doc):
        """Εξαγωγή του κειμένου από την ενότητα body_text"""
        body_text = ""
        if "body_text" in doc and isinstance(doc["body_text"], list):
            body_text = " ".join([entry.get("text", "") for entry in doc["body_text"]])
        return body_text
    
    def search(self, query_text, k=20):
        """Αναζήτηση με βάση το κείμενο του ερωτήματος"""
        query = {
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["title^2", "abstract^1.5", "body_text"],
                    "type": "best_fields"
                }
            },
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
        # Δημιουργία ευρετηρίου μόνο αν δεν υπάρχει ή είναι κενό
        es_engine.create_index()
        
        # Εισαγωγή εγγράφων στο ευρετήριο
        es_engine.index_documents(CORPUS_PATH)
    
    # Εκτέλεση αναζήτησης για όλα τα ερωτήματα
    for k in [20, 30, 50]:
        print(f"\nΕκτέλεση αναζήτησης για τα top-{k} αποτελέσματα...")
        results = es_engine.search_all_queries(QUERIES_PATH, k=k)
        output_file = f"results/results_top{k}.txt"
        es_engine.save_results_trec_format(results, output_file, run_name=f"covid_search_top{k}")


if __name__ == "__main__":
    main()