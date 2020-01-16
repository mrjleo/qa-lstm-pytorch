#! /usr/bin/python3


import os
import argparse
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from qa_lstm import QA_LSTM
from data_source import TestDataset

from qa_utils.misc import Logger
from qa_utils.evaluation import read_args, evaluate_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('DATA_DIR', help='Folder with all preprocessed files')
    ap.add_argument('WORKING_DIR', help='Working directory')
    ap.add_argument('--mrr_k', type=int, default=10, help='Compute MRR@k')
    ap.add_argument('--batch_size', type=int, default=64, help='Batch size')
    ap.add_argument('--glove_cache', help='Word embeddings cache directory')
    args = ap.parse_args()

    train_args = read_args(args.WORKING_DIR)

    dev_file = os.path.join(args.DATA_DIR, 'dev.h5')
    dev_ds = TestDataset(dev_file)
    dev_dl = DataLoader(dev_ds, args.batch_size, collate_fn=dev_ds.collate_fn, pin_memory=True)

    test_file = os.path.join(args.DATA_DIR, 'test.h5')
    test_ds = TestDataset(test_file)
    test_dl = DataLoader(test_ds, args.batch_size, collate_fn=test_ds.collate_fn, pin_memory=True)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print('using {} device(s): "{}"'.format(torch.cuda.device_count(), dev_name))
    else:
        device = torch.device('cpu')
    model = QA_LSTM(int(train_args['hidden_dim']), float(train_args['dropout']),
                    dev_ds.index_to_word, train_args['emb_name'], int(train_args['emb_dim']),
                    False, args.glove_cache, device)
    model.to(device)
    model = torch.nn.DataParallel(model)

    evaluate_all(model, args.WORKING_DIR, dev_dl, test_dl, args.mrr_k, torch.device(device),
                 has_multiple_inputs=True)


if __name__ == '__main__':
    main()
