#! /usr/bin/python3


import os
import json
import argparse
from collections import Counter

import h5py
import nltk
import numpy as np
from tqdm import tqdm

from qa_utils.preprocessing.fiqa import FiQA
from qa_utils.preprocessing.msmarco import MSMARCO
from qa_utils.preprocessing.insrqa import InsuranceQA
from qa_utils.preprocessing.wpqa import WikiPassageQA


def get_word_counter(sentences):
    words = []
    for s in tqdm(sentences):
        words.extend(s)
    return Counter(words)


def get_vocabulary(counter, vocab_size=None):
    index_to_word = {0: '<PAD>', 1: '<UNK>'}
    word_to_index = {'<PAD>': 0, '<UNK>': 1}
    if vocab_size is not None:
        most_common = counter.most_common(vocab_size)
    else:
        most_common = counter.most_common()
    for i, (word, _) in enumerate(most_common, 1):
        index_to_word[i] = word
        word_to_index[word] = i
    return index_to_word, word_to_index


def get_embeddings(words, word_to_index):
    # 1 is the <UNK> token
    return [word_to_index.get(w, 1) for w in words]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('SAVE', help='Where to save the results')
    ap.add_argument('-vs', '--vocab_size', type=int, help='Vocabulary size')
    ap.add_argument('-n', '--num_neg_examples', type=int, default=32,
                    help='Number of negative examples to sample')

    subparsers = ap.add_subparsers(help='Choose a dataset', dest='dataset')
    subparsers.required = True
    FiQA.add_subparser(subparsers, 'fiqa')
    MSMARCO.add_subparser(subparsers, 'msmarco')
    InsuranceQA.add_subparser(subparsers, 'insrqa')
    WikiPassageQA.add_subparser(subparsers, 'wpqa')
    args = ap.parse_args()

    os.makedirs(args.SAVE, exist_ok=True)
    if args.dataset == 'fiqa':
        ds = FiQA(args)
    elif args.dataset == 'insrqa':
        ds = InsuranceQA(args)
    elif args.dataset == 'msmarco':
        ds = MSMARCO(args)
    elif args.dataset == 'wpqa':
        ds = WikiPassageQA(args)

    def _f(x):
        return nltk.word_tokenize(x.lower())
    print('tokenizing queries...')
    ds.transform_queries(_f)
    print('tokenizing documents...')
    ds.transform_docs(_f)

    print('counting words...')
    sentences = list(ds.queries.values()) + list(ds.docs.values())
    word_counter = get_word_counter(sentences)
    index_to_word, word_to_index = get_vocabulary(word_counter, args.vocab_size)

    vocab = {'w2i': json.dumps(word_to_index), 'i2w': json.dumps(index_to_word)}

    # save h5 data
    var_int32 = h5py.special_dtype(vlen=np.dtype('int32'))

    train_file = os.path.join(args.SAVE, 'train.h5')
    print('writing {}...'.format(train_file))
    with h5py.File(train_file, 'w') as fp:
        queries_shape = pos_docs_shape = (len(ds.trainset),)
        # variable length datasets seem to only support 1D data
        neg_docs_shape = (len(ds.trainset) * args.num_neg_examples,)
        queries_ds = fp.create_dataset('queries', queries_shape, dtype=var_int32)
        pos_docs_ds = fp.create_dataset('pos_docs', pos_docs_shape, dtype=var_int32)
        neg_docs_ds = fp.create_dataset('neg_docs', neg_docs_shape, dtype=var_int32)
        queries_ds.attrs.update(vocab)
        neg_docs_ds.attrs['num_neg_examples'] = args.num_neg_examples

        for i, (query, pos_doc, neg_docs) in tqdm(enumerate(ds.trainset), total=len(ds.trainset)):
            queries_ds[i] = get_embeddings(query, word_to_index)
            pos_docs_ds[i] = get_embeddings(pos_doc, word_to_index)
            for j, neg_doc in enumerate(neg_docs):
                neg_docs_ds[i * args.num_neg_examples + j] = get_embeddings(neg_doc, word_to_index)

    dev_file = os.path.join(args.SAVE, 'dev.h5')
    print('writing {}...'.format(dev_file))
    with h5py.File(dev_file, 'w') as fp:
        dev_shape = (len(ds.devset),)
        q_ids_ds = fp.create_dataset('q_ids', dev_shape, dtype='int32')
        queries_ds = fp.create_dataset('queries', dev_shape, dtype=var_int32)
        docs_ds = fp.create_dataset('docs', dev_shape, dtype=var_int32)
        labels_ds = fp.create_dataset('labels', dev_shape, dtype='int32')
        queries_ds.attrs.update(vocab)

        for i, (q_id, query, doc, label) in tqdm(enumerate(ds.devset), total=len(ds.devset)):
            q_ids_ds[i] = q_id
            queries_ds[i] = get_embeddings(query, word_to_index)
            docs_ds[i] = get_embeddings(doc, word_to_index)
            labels_ds[i] = label

    test_file = os.path.join(args.SAVE, 'test.h5')
    print('writing {}...'.format(test_file))
    with h5py.File(test_file, 'w') as fp:
        test_shape = (len(ds.testset),)
        q_ids_ds = fp.create_dataset('q_ids', test_shape, dtype='int32')
        queries_ds = fp.create_dataset('queries', test_shape, dtype=var_int32)
        docs_ds = fp.create_dataset('docs', test_shape, dtype=var_int32)
        labels_ds = fp.create_dataset('labels', test_shape, dtype='int32')
        queries_ds.attrs.update(vocab)

        for i, (q_id, query, doc, label) in tqdm(enumerate(ds.testset), total=len(ds.testset)):
            q_ids_ds[i] = q_id
            queries_ds[i] = get_embeddings(query, word_to_index)
            docs_ds[i] = get_embeddings(doc, word_to_index)
            labels_ds[i] = label


if __name__ == '__main__':
    main()
