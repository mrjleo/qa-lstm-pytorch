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
from qa_utils.evaluation import read_args, get_checkpoints, get_metrics


def evaluate(model, dataloader, k, device):
    result = defaultdict(lambda: ([], []))
    for queries, query_lengths, docs, doc_lengths, q_ids, labels in tqdm(dataloader):
        batch = [t.to(device) for t in [queries, query_lengths, docs, doc_lengths]]
        predictions = model(batch).cpu().detach()
        for q_id, prediction, label in zip(q_ids, predictions.numpy(), labels):
            result[q_id][0].append(prediction)
            result[q_id][1].append(label)

    all_scores, all_labels = [], []
    for q_id, (score, label) in result.items():
        all_scores.append(score)
        all_labels.append(label)
    return get_metrics(all_scores, all_labels, k)


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
    dev_dl = DataLoader(dev_ds, args.batch_size, shuffle=True, collate_fn=dev_ds.collate_fn,
                        pin_memory=True)

    test_file = os.path.join(args.DATA_DIR, 'test.h5')
    test_ds = TestDataset(test_file)
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=True, collate_fn=test_ds.collate_fn,
                         pin_memory=True)

    if torch.cuda.is_available():
        # cuda:0 will still use all GPUs
        device = 'cuda:0'
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print('using {} device(s): "{}"'.format(torch.cuda.device_count(), dev_name))
    else:
        device = 'cpu'
    model = QA_LSTM(int(train_args['hidden_dim']), float(train_args['dropout']),
                    dev_ds.index_to_word, train_args['emb_name'], int(train_args['emb_dim']),
                    False, args.glove_cache, device)
    model = torch.nn.DataParallel(model)

    eval_file = os.path.join(args.WORKING_DIR, 'eval.csv')
    logger = Logger(eval_file, ['ckpt', 'dev_map', 'dev_mrr', 'test_map', 'test_mrr'])
    model.eval()
    for ckpt in get_checkpoints(os.path.join(args.WORKING_DIR, 'ckpt'), r'weights_(\d+).pt'):
        print('processing {}...'.format(ckpt))
        state = torch.load(ckpt)
        model.module.load_state_dict(state['state_dict'])
        with torch.no_grad():
            dev_metrics = evaluate(model, dev_dl, args.mrr_k, torch.device(device))
            test_metrics = evaluate(model, test_dl, args.mrr_k, torch.device(device))
        row = [ckpt] + list(dev_metrics) + list(test_metrics)
        logger.log(row)


if __name__ == '__main__':
    main()
