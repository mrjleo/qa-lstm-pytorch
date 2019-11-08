#! /usr/bin/python3


import argparse
import os
import csv
from collections import defaultdict

import nltk
from tqdm import tqdm

from functions import count_lines, get_word_counter, TrainDataset, TestDataset, save


def read_files(train_file, dev_file, test_file):
    q_ids, doc_ids, train_queries, docs = {}, {}, {}, {}
    cur_q_id, cur_doc_id = 0, 0

    print('processing {}...'.format(train_file), flush=True)
    train_qrels = defaultdict(set)
    total = count_lines(train_file)
    with open(train_file, encoding='utf-8') as fp:
        for q, doc, label in tqdm(csv.reader(fp, delimiter='\t'), total=total):
            q = q.lower()
            doc = doc.lower()
            if q not in q_ids:
                q_ids[q] = cur_q_id
                cur_q_id += 1
            # account for duplicate queries, if any
            train_queries[q_ids[q]] = nltk.word_tokenize(q)
            if doc not in doc_ids:
                doc_ids[doc] = cur_doc_id
                docs[cur_doc_id] = nltk.word_tokenize(doc)
                cur_doc_id += 1
            if label == '1':
                train_qrels[q_ids[q]].add(doc_ids[doc])

    def _process_test_file(f):
        nonlocal q_ids, doc_ids, cur_q_id, cur_doc_id
        print('processing {}...'.format(f), flush=True)
        test_set = defaultdict(list)
        test_queries = {}
        total = count_lines(f)
        with open(f, encoding='utf-8') as fp:
            for q, doc, label in tqdm(csv.reader(fp, delimiter='\t'), total=total):
                q = q.lower()
                doc = doc.lower()
                if q not in q_ids:
                    q_ids[q] = cur_q_id
                    cur_q_id += 1
                # account for duplicate queries, if any
                test_queries[q_ids[q]] = nltk.word_tokenize(q)
                if doc not in doc_ids:
                    doc_ids[doc] = cur_doc_id
                    docs[cur_doc_id] = nltk.word_tokenize(doc)
                    cur_doc_id += 1
                test_set[q_ids[q]].append((doc_ids[doc], int(label)))
        return test_set, test_queries

    dev_set, dev_queries = _process_test_file(dev_file)
    test_set, test_queries =_process_test_file(test_file)
    return docs, train_qrels, train_queries, dev_set, dev_queries, test_set, test_queries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('WIKIQA_DIR', help='Folder with all WikiQA files')
    ap.add_argument('--save', default='result', help='Where to save the results')
    ap.add_argument('-n', '--num_neg_examples', type=int, default=16, help='Number of negative examples to sample')
    args = ap.parse_args()

    os.makedirs(args.save, exist_ok=True)

    train_file = os.path.join(args.WIKIQA_DIR, 'WikiQASent-train.txt')
    dev_file = os.path.join(args.WIKIQA_DIR, 'WikiQASent-dev.txt')
    test_file = os.path.join(args.WIKIQA_DIR, 'WikiQASent-test.txt')
    docs, train_qrels, train_queries, dev_set, dev_queries, test_set, test_queries = read_files(train_file, dev_file, test_file)

    print('counting words...', flush=True)
    sentences = list(train_queries.values()) + list(dev_queries.values()) + list(test_queries.values()) + list(docs.values())
    word_counter = get_word_counter(sentences)

    train_dataset = TrainDataset(train_queries, docs, train_qrels, args.num_neg_examples)
    dev_dataset = TestDataset(dev_queries, docs, dev_set)
    test_dataset = TestDataset(test_queries, docs, test_set)

    save(args.save, word_counter, args.num_neg_examples, docs,
         train_dataset, train_queries,
         dev_dataset, dev_queries,
         test_dataset, test_queries)


if __name__ == '__main__':
    main()
