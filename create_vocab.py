#! /usr/bin/env python3


import pickle
import argparse
from itertools import chain
from collections import Counter

import h5py
import nltk
import torch
from tqdm import tqdm
from torchtext.vocab import Vocab
from pytorch_lightning import seed_everything


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('DATA_FILE', help='File that holds the queries and documents')
    ap.add_argument('--max_size', type=int, help='Maximum vocabulary size')
    ap.add_argument('--cache', help='Torchtext cache')
    ap.add_argument('--vectors', default='glove.840B.300d', help='Pre-trained vectors')
    ap.add_argument('--out_file', default='vocab.pkl', help='Where to save the vocabulary')
    ap.add_argument('--random_seed', type=int, default=123, help='Random seed')
    args = ap.parse_args()

    seed_everything(args.random_seed)

    print(f'reading {args.DATA_FILE}...')
    with h5py.File(args.DATA_FILE, 'r') as fp:
        num_items = len(fp['queries']) + len(fp['docs'])
        ct = Counter()
        for s in tqdm(chain(fp['queries'], fp['docs']), total=num_items):
            ct.update(nltk.word_tokenize(s))
        vocab = Vocab(ct, args.max_size, vectors=args.vectors, vectors_cache=args.cache, unk_init=torch.normal)

    print(f'writing {args.out_file}...')
    with open(args.out_file, 'wb') as fp:
        pickle.dump(vocab, fp)


if __name__ == '__main__':
    main()
