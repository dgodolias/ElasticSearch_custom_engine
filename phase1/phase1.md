# Αναφορά Προγραμματιστικής Εργασίας - Φάση 1: Κλασική Ανάκτηση

**Ονοματεπώνυμα:**
*   [Ονοματεπώνυμο Φοιτητή 1] - [Αριθμός Μητρώου 1]
*   [Ονοματεπώνυμο Φοιτητή 2] - [Αριθμός Μητρώου 2]

**Ημερομηνία:** 30/04/2025

---

## 1. Υλοποίηση

Σε αυτή την ενότητα περιγράφεται η υλοποίηση των βημάτων της Φάσης 1 για τη δημιουργία και αξιολόγηση της βασικής μηχανής αναζήτησης με χρήση Elasticsearch.

### 1.1 Προεπεξεργασία Συλλογής (Βήμα 1)

Η συλλογή κειμένων `trec-covid` παρέχεται ήδη σε μορφή `.jsonl`, η οποία είναι κατάλληλη για επεξεργασία γραμμή προς γραμμή. Κάθε γραμμή αντιστοιχεί σε ένα έγγραφο JSON.

Η προεπεξεργασία περιλάμβανε κυρίως την ανάγνωση κάθε γραμμής JSON και την αντιστοίχιση των πεδίων του JSON (`_id`, `title`, `text`) στα αντίστοιχα πεδία που θα χρησιμοποιούσε το Elasticsearch (`doc_id`, `title`, `body_text`). Δεν απαιτήθηκε περαιτέρω μετατροπή της δομής των αρχείων.

*(Προαιρετικό Screenshot: {δομή_αρχείου_jsonl}.png)*

### 1.2 Δημιουργία Ευρετηρίου (Βήμα 2)

Για τη δημιουργία του ευρετηρίου χρησιμοποιήθηκε η μηχανή αναζήτησης Elasticsearch (έκδοση 9.0.0).

*   **Analyzer:** Επιλέχθηκε ένας custom analyzer (`custom_analyzer`) με τις παρακάτω ρυθμίσεις:
    *   **Tokenizer:** `standard`
    *   **Filters:**
        *   `lowercase`: Μετατροπή όλων των χαρακτήρων σε πεζούς.
        *   `english_possessive_stemmer`: Αφαίρεση κτητικών καταλήξεων ('s).
        *   `english_stop`: Αφαίρεση αγγλικών stop words.
        *   `english_stemmer`: Εφαρμογή stemming για την αγγλική γλώσσα.
        *   `medical_prefix_filter`: Προσαρμοσμένο φίλτρο για τη διατήρηση κοινών ιατρικών προθεμάτων (π.χ., "covid", "sars").
*   **Similarity Function:** Επιλέχθηκε η συνάρτηση ομοιότητας `BM25` με προσαρμοσμένες παραμέτρους (`"b": 0.5`, `"k1": 1.6`) για καλύτερη απόδοση σε ιατρικά κείμενα (`custom_similarity`).
*   **Field Mapping:** Τα πεδία του ευρετηρίου ορίστηκαν ως εξής:
    *   `title`: Τύπου `text`, με χρήση του `custom_analyzer` και `custom_similarity`.
    *   `body_text`: Τύπου `text`, με χρήση του `custom_analyzer` και `custom_similarity`.
    *   `doc_id`: Τύπου `keyword` για ακριβή αντιστοίχιση του αναγνωριστικού.

Η εισαγωγή των εγγράφων έγινε με χρήση της συνάρτησης `helpers.parallel_bulk` της βιβλιοθήκης `elasticsearch-py` για βελτιωμένη ταχύτητα.

*(Προαιρετικό Screenshot: {ρυθμίσεις_analyzer_similarity}.png)*
*(Προαιρετικό Screenshot: {εκτέλεση_indexing}.png)*

### 1.3 Εκτέλεση Ερωτημάτων (Βήμα 3)

Τα ερωτήματα από το αρχείο `queries.jsonl` εκτελέστηκαν πάνω στο ευρετήριο που δημιουργήθηκε. Για κάθε ερώτημα, χρησιμοποιήθηκε ένα `multi_match` query στα πεδία `title` (με βάρος 3) και `body_text` (με βάρος 1).

Συλλέχθηκαν τα `k` πρώτα αποτελέσματα για τις τιμές `k = 20`, `k = 30`, και `k = 50`. Τα αποτελέσματα για κάθε τιμή του `k` αποθηκεύτηκαν σε ξεχωριστά αρχεία (`results_top20.txt`, `results_top30.txt`, `results_top50.txt`) σε μορφή συμβατή με το `trec_eval`.

*(Προαιρετικό Screenshot: {κώδικας_εκτέλεσης_query}.png)*
*(Προαιρετικό Screenshot: {μορφή_αρχείου_αποτελεσμάτων_trec}.png)*

### 1.4 Αξιολόγηση (Βήμα 4)

Η αξιολόγηση των αποτελεσμάτων έγινε με χρήση του εργαλείου `trec_eval`. Χρησιμοποιήθηκε το αρχείο `test_trec_format.tsv` ως ground truth (qrels). Οι μετρικές που υπολογίστηκαν ήταν:

*   **MAP (Mean Average Precision)**
*   **P@k (Precision at k)** για k = 5, 10, 15, 20.

Η αξιολόγηση εκτελέστηκε ξεχωριστά για κάθε αρχείο αποτελεσμάτων (`results_top20.txt`, `results_top30.txt`, `results_top50.txt`).

---

## 2. Αποτελέσματα Αξιολόγησης

Τα αποτελέσματα της αξιολόγησης με το `trec_eval` παρουσιάζονται στις παρακάτω φωτογραφίες:

*(Screenshot: {εκτέλεση_trec_eval_top20}.png)*
*(Screenshot: {εκτέλεση_trec_eval_top30}.png)*
*(Screenshot: {εκτέλεση_trec_eval_top50}.png)*

---

## 3. Συζήτηση Αποτελεσμάτων

*   Σχολιάστε τις τιμές MAP. Είναι ικανοποιητικές; Πώς συγκρίνονται μεταξύ των τριών αρχείων αποτελεσμάτων (top20, top30, top50) και γιατί;
*   Σχολιάστε τις τιμές P@k. Γιατί οι τιμές P@5, P@10, P@15, P@20 είναι ίδιες και στα τρία αρχεία; Τι σημαίνει αυτό για την κατάταξη των πρώτων 20 αποτελεσμάτων;
*   Παρατηρείτε κάποια τάση στις τιμές P@k καθώς το k αυξάνεται (P@5 vs P@10 vs P@15 vs P@20); Τι υποδηλώνει αυτό για την ποιότητα των αποτελεσμάτων στις πρώτες θέσεις;
*   Ποιες πιθανές βελτιώσεις θα μπορούσαν να γίνουν στον Analyzer, τη συνάρτηση ομοιότητας ή τον τρόπο εκτέλεσης των ερωτημάτων για να αυξηθεί η απόδοση;

---

## 4. Πηγές

*   Elasticsearch Documentation: [https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
*   elasticsearch-py Documentation: [https://elasticsearch-py.readthedocs.io/](https://elasticsearch-py.readthedocs.io/)
*   BM25 Similarity: [https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html#bm25](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html#bm25)
*   trec_eval Documentation: [https://trec.nist.gov/trec_eval/](https://trec.nist.gov/trec_eval/)
*   [Προσθέστε τυχόν άλλες πηγές που χρησιμοποιήσατε]

---