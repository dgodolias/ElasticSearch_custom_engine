import os
import subprocess
import pandas as pd
from tabulate import tabulate

# Διαδρομές για τα αρχεία αξιολόγησης
QRELS_PATH = "covid_datasets/qrels/test.tsv"
RESULTS_DIR = "results"

# Μετρικές αξιολόγησης
EVALUATION_METRICS = [
    "map",  # Mean Average Precision
    "P.5",  # Precision at 5
    "P.10",  # Precision at 10
    "P.15",  # Precision at 15
    "P.20"   # Precision at 20
]

def run_trec_eval(qrels_path, results_path, metrics):
    """Εκτέλεση του trec_eval για αξιολόγηση των αποτελεσμάτων"""
    # Βεβαιωθείτε ότι το trec_eval είναι εγκατεστημένο και διαθέσιμο στο path
    # Αν όχι, αλλάξτε την παρακάτω εντολή με το πλήρες path του εκτελέσιμου
    
    cmd = ["trec_eval", "-m", ",".join(metrics), qrels_path, results_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return parse_trec_eval_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Σφάλμα κατά την εκτέλεση του trec_eval: {e}")
        print(f"Έξοδος σφάλματος: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Το εργαλείο trec_eval δεν βρέθηκε. Βεβαιωθείτε ότι είναι εγκατεστημένο και διαθέσιμο στο PATH.")
        print("Εναλλακτικά, αλλάξτε την εντολή στον κώδικα με το πλήρες path του εκτελέσιμου.")
        return None

def parse_trec_eval_output(output):
    """Ανάλυση της εξόδου του trec_eval και μετατροπή σε dictionary"""
    results = {}
    for line in output.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 3:
            metric = parts[0]
            value = float(parts[2])
            results[metric] = value
    return results

def evaluate_all_results():
    """Αξιολόγηση όλων των αποτελεσμάτων για διαφορετικά k"""
    if not os.path.exists(QRELS_PATH):
        print(f"Το αρχείο {QRELS_PATH} δε βρέθηκε.")
        return
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Ο φάκελος {RESULTS_DIR} δε βρέθηκε.")
        return
    
    # Συλλογή αποτελεσμάτων αξιολόγησης για κάθε k
    evaluation_results = {}
    k_values = [20, 30, 50]
    
    for k in k_values:
        results_path = os.path.join(RESULTS_DIR, f"results_top{k}.txt")
        if not os.path.exists(results_path):
            print(f"Το αρχείο {results_path} δε βρέθηκε.")
            continue
        
        print(f"Αξιολόγηση αποτελεσμάτων για k={k}...")
        metrics = run_trec_eval(QRELS_PATH, results_path, EVALUATION_METRICS)
        if metrics:
            evaluation_results[k] = metrics
    
    # Δημιουργία συγκριτικού πίνακα
    if evaluation_results:
        # Μετατροπή σε DataFrame για εύκολη απεικόνιση
        df = pd.DataFrame.from_dict(evaluation_results, orient='index')
        
        # Μορφοποίηση και εμφάνιση του πίνακα
        print("\nΣύγκριση αποτελεσμάτων για διαφορετικά k:")
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt='.4f'))
        
        # Αποθήκευση των αποτελεσμάτων σε CSV
        csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
        df.to_csv(csv_path)
        print(f"\nΤα αποτελέσματα αξιολόγησης αποθηκεύτηκαν στο αρχείο: {csv_path}")
    else:
        print("Δεν υπάρχουν διαθέσιμα αποτελέσματα αξιολόγησης.")

if __name__ == "__main__":
    evaluate_all_results()