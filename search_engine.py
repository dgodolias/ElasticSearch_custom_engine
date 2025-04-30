import json
import os
from elasticsearch import Elasticsearch
from tqdm import tqdm

# Διαδρομές για τα αρχεία της συλλογής
CORPUS_PATH = "covid_datasets/corpus.jsonl"
QUERIES_PATH = "covid_datasets/queries.jsonl"
QRELS_PATH = "covid_datasets/qrels/test.tsv"

# Ρυθμίσεις ElasticSearch
INDEX_NAME = "covid_search"
ES_HOST = "http://localhost:9200"

class ElasticSearchEngine:
    def __init__(self, host=ES_HOST):
        """Αρχικοποίηση του ElasticSearch client"""
        self.es = Elasticsearch(host)
        self.index_name = INDEX_NAME
        
    def check_connection(self):
        """Έλεγχος σύνδεσης με τον ElasticSearch server"""
        try:
            return self.es.ping()
        except Exception as e:
            print(f"Σφάλμα σύνδεσης με τον ElasticSearch: {e}")
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
    
    # Δημιουργία ευρετηρίου
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