#! /usr/bin/python3


import argparse
import os
import gzip
import csv
from collections import defaultdict

from tqdm import tqdm

from functions import count_lines, get_word_counter, TrainDataset, TestDataset, save


def decode(idx_list, vocab):
    return [vocab[idx].lower() for idx in idx_list]


def read_vocab(vocab_file):
    print('processing {}...'.format(vocab_file), flush=True)
    vocab = {}
    total = count_lines(vocab_file)
    with open(vocab_file, encoding='utf-8') as fp:
        for idx, word in tqdm(csv.reader(fp, delimiter='\t', quotechar=None), total=total):
            vocab[idx] = word
    return vocab


def read_docs(l2a_file, vocab):
    print('processing {}...'.format(l2a_file), flush=True)
    docs = {}
    total = count_lines(l2a_file)
    with gzip.open(l2a_file) as fp:
        for line in tqdm(fp, total=total):
            doc_id, doc_idxs = line.decode('utf-8').split('\t')
            docs[int(doc_id)] = decode(doc_idxs.split(), vocab)
    return docs


def read_train_set(train_file, vocab):
    print('processing {}...'.format(train_file), flush=True)
    train_queries = {}
    qrels = defaultdict(set)
    total = count_lines(train_file)
    with gzip.open(train_file) as fp:
        for q_id, line in enumerate(tqdm(fp, total=total)):
            _, q_idxs, gt, _ = line.decode('utf-8').split('\t')
            train_queries[q_id] = decode(q_idxs.split(), vocab)
            qrels[q_id] = set(map(int, gt.split()))
    return train_queries, qrels


def read_test_set(test_file, vocab):
    print('processing {}...'.format(test_file), flush=True)
    test_queries = {}
    test_set = defaultdict(list)
    total = count_lines(test_file)
    with gzip.open(test_file) as fp:
        for q_id, line in enumerate(tqdm(fp, total=total)):
            _, q_idxs, gt, pool = line.decode('utf-8').split('\t')
            test_queries[q_id] = decode(q_idxs.split(), vocab)

            pos_doc_ids = set(map(int, gt.split()))
            # make sure no positive IDs are in the pool
            neg_doc_ids = set(map(int, pool.split())) - pos_doc_ids
            assert len(pos_doc_ids & neg_doc_ids) == 0

            for pos_doc_id in pos_doc_ids:
                test_set[q_id].append((pos_doc_id, 1))
            for neg_doc_id in neg_doc_ids:
                test_set[q_id].append((neg_doc_id, 0)) 

    return test_queries, test_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('INSRQA_V2_DIR', help='Folder with insuranceQA v2 files')
    ap.add_argument('--save', default='result', help='Where to save the results')
    ap.add_argument('-n', '--num_neg_examples', type=int, default=16, help='Number of negative examples to sample')
    args = ap.parse_args()

    os.makedirs(args.save, exist_ok=True)

    vocab_file = os.path.join(args.INSRQA_V2_DIR, 'vocabulary')
    vocab = read_vocab(vocab_file)

    l2a_file = os.path.join(args.INSRQA_V2_DIR, 'InsuranceQA.label2answer.token.encoded.gz')
    docs = read_docs(l2a_file, vocab)

    # use the smallest file here as we do the sampling by ourself
    train_file = os.path.join(args.INSRQA_V2_DIR, 'InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded.gz')
    train_queries, train_qrels = read_train_set(train_file, vocab)
    train_dataset = TrainDataset(train_queries, docs, train_qrels, args.num_neg_examples)

    # use dev- and test-set with 1000 examples per query
    dev_file = os.path.join(args.INSRQA_V2_DIR, 'InsuranceQA.question.anslabel.token.1000.pool.solr.valid.encoded.gz')
    dev_queries, dev_set = read_test_set(dev_file, vocab)
    dev_dataset = TestDataset(dev_queries, docs, dev_set)

    test_file = os.path.join(args.INSRQA_V2_DIR, 'InsuranceQA.question.anslabel.token.1000.pool.solr.test.encoded.gz')
    test_queries, test_set = read_test_set(test_file, vocab)
    test_dataset = TestDataset(test_queries, docs, test_set)

    print('counting words...', flush=True)
    sentences = list(train_queries.values()) + list(dev_queries.values()) + list(test_queries.values()) + list(docs.values())
    word_counter = get_word_counter(sentences)

    save(args.save, word_counter, args.num_neg_examples, docs,
         train_dataset, train_queries,
         dev_dataset, dev_queries,
         test_dataset, test_queries)


if __name__ == '__main__':
    main()
