#! /usr/bin/python3


import argparse
import os
import csv
import pickle
from collections import defaultdict

import nltk
from tqdm import tqdm

from functions import count_lines, get_word_counter, TrainDataset, TestDataset, save


def read_collection(file_path, cache_file=None):
    print('processing {}...'.format(file_path), flush=True)
    if cache_file is not None and os.path.isfile(cache_file):
        print('loading cached file {}...'.format(cache_file), flush=True)
        with open(cache_file, 'rb') as fp:
            return pickle.load(fp)

    items = {}
    total = count_lines(file_path)
    with open(file_path, encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for item_id, item in tqdm(reader, total=total):
            items[int(item_id)] = nltk.word_tokenize(item.lower())
    
    if cache_file is not None:
        print('caching in {}...'.format(cache_file), flush=True)
        with open(cache_file, 'wb') as fp:
            pickle.dump(items, fp)
    return items


def read_qrels(file_path):
    print('processing {}...'.format(file_path), flush=True)
    qrels = defaultdict(set)
    total = count_lines(file_path)
    with open(file_path, encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for q_id, _, doc_id, _ in tqdm(reader, total=total):
            qrels[int(q_id)].add(int(doc_id))
    return qrels


def read_dev_set(dev_set_file, qrels_file, cache_file=None):
    print('processing {}...'.format(qrels_file), flush=True)
    if cache_file is not None and os.path.isfile(cache_file):
        print('loading cached file {}...'.format(cache_file), flush=True)
        with open(cache_file, 'rb') as fp:
            return pickle.load(fp)

    qrels = defaultdict(set)
    total = count_lines(qrels_file)
    with open(qrels_file, encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for q_id, _, doc_id, _ in tqdm(reader, total=total):
            qrels[int(q_id)].add(int(doc_id))

    print('processing {}...'.format(dev_set_file), flush=True)
    dev_set = defaultdict(list)
    total = count_lines(dev_set_file)
    with open(dev_set_file, encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for q_id, doc_id, _, _ in tqdm(reader, total=total):
            label = 1 if int(doc_id) in qrels[int(q_id)] else 0
            dev_set[int(q_id)].append((int(doc_id), label))

    if cache_file is not None:
        print('caching in {}...'.format(cache_file), flush=True)
        with open(cache_file, 'wb') as fp:
            pickle.dump(dev_set, fp)
    return dev_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('MSM_DIR', help='Folder with all necessary MS MARCO files')
    ap.add_argument('--save', default='result', help='Where to save the results')
    ap.add_argument('-n', '--num_neg_examples', type=int, default=16, help='Number of negative examples to sample')
    ap.add_argument('--cache_dir', default='.preproc_cache', help='Cache directory')
    args = ap.parse_args()

    os.makedirs(args.save, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    collection_file = os.path.join(args.MSM_DIR, 'collection.tsv')
    collection_cache = os.path.join(args.cache_dir, 'msm_collection.pkl')
    docs = read_collection(collection_file, collection_cache)

    train_queries_file = os.path.join(args.MSM_DIR, 'queries.train.tsv')
    train_queries_cache = os.path.join(args.cache_dir, 'msm_train_queries.pkl')
    train_queries = read_collection(train_queries_file, train_queries_cache)
    train_qrels_file = os.path.join(args.MSM_DIR, 'qrels.train.tsv')
    train_qrels = read_qrels(train_qrels_file)
    
    dev_file = os.path.join(args.MSM_DIR, 'top1000.dev.tsv')
    dev_qrels_file = os.path.join(args.MSM_DIR, 'qrels.dev.tsv')
    dev_cache = os.path.join(args.cache_dir, 'msm_dev.pkl')
    dev_set = read_dev_set(dev_file, dev_qrels_file, dev_cache)
    dev_queries_file = os.path.join(args.MSM_DIR, 'queries.dev.tsv')
    dev_queries_cache = os.path.join(args.cache_dir, 'dev_msm_queries.pkl')
    dev_queries = read_collection(dev_queries_file, dev_queries_cache)

    print('counting words...', flush=True)
    sentences = list(docs.values()) + list(train_queries.values()) + list(dev_queries.values())
    word_counter = get_word_counter(sentences)

    train_dataset = TrainDataset(train_queries, docs, train_qrels, args.num_neg_examples)
    dev_dataset = TestDataset(dev_queries, docs, dev_set)

    save(args.save, word_counter, args.num_neg_examples, docs,
         train_dataset, train_queries,
         dev_dataset, dev_queries)


if __name__ == '__main__':
    main()
