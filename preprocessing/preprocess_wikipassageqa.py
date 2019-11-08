#! /usr/bin/python3


import argparse
import os
import csv
import json
import random
from collections import defaultdict

import nltk
from tqdm import tqdm

from functions import count_lines, get_word_counter, TrainDataset, TestDataset, save


def read_docs(file_path):
    print('processing {}...'.format(file_path), flush=True)
    with open(file_path, encoding='utf-8') as fp:
        docs_json = json.load(fp)
    docs = {}
    for art_id in tqdm(docs_json):
        for p_id, passage in docs_json[art_id].items():
            docs[(int(art_id), int(p_id))] = nltk.word_tokenize(passage.lower())
    return docs


def read_train_file(file_path):
    print('processing {}...'.format(file_path), flush=True)
    queries = {}
    qrels = defaultdict(set)
    total = count_lines(file_path) - 1
    with open(file_path, encoding='utf-8') as fp:
        # skip header
        next(fp)
        for q_id, question, d_id, _, p_ids in tqdm(csv.reader(fp, delimiter='\t'), total=total):
            queries[int(q_id)] = nltk.word_tokenize(question.lower())
            for p_id in map(int, p_ids.split(',')):
                qrels[int(q_id)].add((int(d_id), p_id))
    return queries, qrels


def read_test_file(file_path, docs, num_neg_examples_dev):
    print('processing {}...'.format(file_path), flush=True)
    queries = {}
    test_set = defaultdict(set)
    total = count_lines(file_path) - 1
    with open(file_path, encoding='utf-8') as fp:
        # skip header
        next(fp)
        for q_id, question, d_id, _, p_ids in tqdm(csv.reader(fp, delimiter='\t'), total=total):
            queries[int(q_id)] = nltk.word_tokenize(question.lower())
            for p_id in map(int, p_ids.split(',')):
                pos_id = (int(d_id), p_id)
                test_set[int(q_id)].add((pos_id, 1))

            # sample negatives, making sure we dont sample any positives
            population = set(docs) - test_set[int(q_id)]
            n = num_neg_examples_dev - len(test_set[int(q_id)])
            for neg_id in random.sample(population, n):
                test_set[int(q_id)].add((neg_id, 0))
    return queries, test_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('WIKIPASSAGEQA_DIR', help='Folder with all WikiPassageQA files')
    ap.add_argument('--save', default='result', help='Where to save the results')
    ap.add_argument('-n', '--num_neg_examples', type=int, default=16, help='Number of negative examples to sample')
    ap.add_argument('-nd', '--num_neg_examples_dev', type=int, default=1000, help='How many examples per query for dev/test sets')
    args = ap.parse_args()

    os.makedirs(args.save, exist_ok=True)

    doc_file = os.path.join(args.WIKIPASSAGEQA_DIR, 'document_passages.json')
    train_file = os.path.join(args.WIKIPASSAGEQA_DIR, 'train.tsv')
    dev_file = os.path.join(args.WIKIPASSAGEQA_DIR, 'dev.tsv')
    test_file = os.path.join(args.WIKIPASSAGEQA_DIR, 'test.tsv')
    docs = read_docs(doc_file)
    train_queries, train_qrels = read_train_file(train_file)
    dev_queries, dev_set = read_test_file(dev_file, docs, args.num_neg_examples_dev)
    test_queries, test_set = read_test_file(test_file, docs, args.num_neg_examples_dev)
    
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
