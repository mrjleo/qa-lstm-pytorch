#! /usr/bin/python3


import argparse
import os
import csv
import pickle
from collections import defaultdict

import nltk
from tqdm import tqdm

from functions import count_lines, get_word_counter, TrainDataset, TestDataset, save


def read_files(question_file, doc_file, question_doc_file):
    print('processing {}...'.format(question_file), flush=True)
    queries = {}
    total = count_lines(question_file) - 1
    with open(question_file, encoding='utf-8') as fp:
        # skip header
        next(fp)
        for _, q_id, question, _ in tqdm(csv.reader(fp, delimiter='\t'), total=total):
            queries[int(q_id)] = nltk.word_tokenize(question.lower())

    print('processing {}...'.format(doc_file), flush=True)
    docs = {}
    total = count_lines(doc_file) - 1
    with open(doc_file, encoding='utf-8') as fp:
        # skip header
        next(fp)
        for _, doc_id, doc, _ in tqdm(csv.reader(fp, delimiter='\t'), total=total):
            docs[int(doc_id)] = nltk.word_tokenize(doc.lower())

    print('processing {}...'.format(question_doc_file), flush=True)
    qrels = defaultdict(set)
    total = count_lines(question_doc_file) - 1
    with open(question_doc_file, encoding='utf-8') as fp:
        # skip header
        next(fp)
        for _, q_id, doc_id in tqdm(csv.reader(fp, delimiter='\t'), total=total):
            qrels[int(q_id)].add(int(doc_id))

    return queries, docs, qrels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('FIQA_DIR', help='Folder with all FiQA files')
    ap.add_argument('SPLIT_FILE', help='FiQA split pickle file')
    ap.add_argument('--save', default='result', help='Where to save the results')
    ap.add_argument('-n', '--num_neg_examples', type=int, default=16, help='Number of negative examples to sample')
    args = ap.parse_args()

    os.makedirs(args.save, exist_ok=True)

    question_file = os.path.join(args.FIQA_DIR, 'FiQA_train_question_final.tsv')
    doc_file = os.path.join(args.FIQA_DIR, 'FiQA_train_doc_final.tsv')
    question_doc_file = os.path.join(args.FIQA_DIR, 'FiQA_train_question_doc_final.tsv')
    queries, docs, qrels = read_files(question_file, doc_file, question_doc_file)
    
    print('counting words...', flush=True)
    sentences = list(queries.values()) + list(docs.values())
    word_counter = get_word_counter(sentences)

    print('reading {}...'.format(args.SPLIT_FILE), flush=True)
    with open(args.SPLIT_FILE, 'rb') as fp:
        train_q_ids, dev_set, test_set = pickle.load(fp)

    train_queries = {q_id: queries[q_id] for q_id in train_q_ids}
    train_dataset = TrainDataset(train_queries, docs, qrels, args.num_neg_examples)
    dev_queries = {q_id: queries[q_id] for q_id in dev_set}
    dev_dataset = TestDataset(dev_queries, docs, dev_set)
    test_queries = {q_id: queries[q_id] for q_id in test_set}
    test_dataset = TestDataset(test_queries, docs, test_set)

    save(args.save, word_counter, args.num_neg_examples, docs,
         train_dataset, train_queries,
         dev_dataset, dev_queries,
         test_dataset, test_queries)


if __name__ == '__main__':
    main()
