#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Μετατροπή αρχείου qrels από τη μορφή του πανεπιστημίου:
    query-id corpus-id score
στη μορφή που χρειάζεται το trec_eval:
    query_id 0 doc_id relevance_level
"""

import os
import csv
import argparse

def convert_qrels(input_file, output_file):
    """
    Μετατρέπει το αρχείο qrels από τη μορφή του πανεπιστημίου στη μορφή για το trec_eval.
    
    Args:
        input_file (str): Το αρχείο εισόδου στη μορφή του πανεπιστημίου.
        output_file (str): Το αρχείο εξόδου στη μορφή για το trec_eval.
    """
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        reader = csv.reader(fin, delimiter='\t')
        
        # Αγνόηση της γραμμής επικεφαλίδας
        header = next(reader, None)
        
        # Εγγραφή των δεδομένων στη νέα μορφή
        for row in reader:
            if len(row) >= 3:
                query_id, doc_id, relevance = row[0], row[1], row[2]
                fout.write(f"{query_id} 0 {doc_id} {relevance}\n")
            else:
                print(f"Προσπέραση ελλιπούς γραμμής: {row}")
    
    print(f"Η μετατροπή ολοκληρώθηκε. Το αρχείο αποθηκεύτηκε ως: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Μετατροπή αρχείου qrels από τη μορφή του πανεπιστημίου στη μορφή για το trec_eval'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Το αρχείο εισόδου στη μορφή του πανεπιστημίου'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Το αρχείο εξόδου στη μορφή για το trec_eval'
    )
    
    args = parser.parse_args()
    
    convert_qrels(args.input_file, args.output_file)

if __name__ == '__main__':
    main()