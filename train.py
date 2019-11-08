#! /usr/bin/python3


import argparse
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from qa_lstm import QA_LSTM
from data_source import TrainDataset, TestDataset
from functions import read_pkl, get_vocab, calc_metrics, Logger


def loss(s_pos, s_neg, margin):
    return torch.clamp(margin - s_pos + s_neg, min=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('PREPROC_DIR', help='Directory with preprocessed files')
    ap.add_argument('-vs', '--vocab_size', type=int, help='Limit vocabulary size')
    ap.add_argument('-en', '--emb_name', default='840B', help='GloVe embedding name')
    ap.add_argument('-ed', '--emb_dim', type=int, default=300, help='Word embedding dimension')
    ap.add_argument('-hd', '--hidden_dim', type=int, default=128, help='LSTM hidden dimension')
    ap.add_argument('-d', '--dropout', type=float, default=0.3, help='Dropout rate')
    ap.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
    ap.add_argument('-m', '--margin', type=float, default=1, help='Margin for loss function')
    ap.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs')
    ap.add_argument('-vbs', '--valid_batch_size', type=int, default=32, help='Validation/testing batch size')
    ap.add_argument('-k', '--mrr_k', type=int, default=10, help='Compute MRR@k')
    ap.add_argument('--test', action='store_true', help='Also compute the metrics on the test set')
    ap.add_argument('--ckpt', default='ckpt', help='Where to save checkpoints')
    ap.add_argument('--logfile', default='train.csv', help='Training log file')
    ap.add_argument('--glove_cache', help='Word embeddings cache directory')
    ap.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers')
    ap.add_argument('--random_seed', type=int, default=12345, help='Random seed')
    args = ap.parse_args()

    torch.manual_seed(args.random_seed)
    os.makedirs(args.ckpt, exist_ok=True)

    info_pkl = os.path.join(args.PREPROC_DIR, 'info.pkl')
    word_counts, num_neg_examples, = read_pkl(info_pkl)
    word_to_id, id_to_word = get_vocab(word_counts, args.vocab_size)

    docs_pkl = os.path.join(args.PREPROC_DIR, 'docs.pkl')
    docs = read_pkl(docs_pkl)

    train_pkl = os.path.join(args.PREPROC_DIR, 'train.pkl')
    train_set = read_pkl(train_pkl)
    train_queries_pkl = os.path.join(args.PREPROC_DIR, 'train_queries.pkl')
    train_queries = read_pkl(train_queries_pkl)
    train_dataset = TrainDataset(word_to_id, docs, train_queries, train_set, num_neg_examples)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  collate_fn=train_dataset.collate_fn, pin_memory=True)

    dev_pkl = os.path.join(args.PREPROC_DIR, 'dev.pkl')
    dev_set = read_pkl(dev_pkl)
    dev_queries_pkl = os.path.join(args.PREPROC_DIR, 'dev_queries.pkl')
    dev_queries = read_pkl(dev_queries_pkl)
    dev_dataset = TestDataset(word_to_id, docs, dev_queries, dev_set)
    dev_dataloader = DataLoader(dev_dataset, args.valid_batch_size, num_workers=args.num_workers,
                                collate_fn=dev_dataset.collate_fn, pin_memory=True)
    
    if args.test:
        test_pkl = os.path.join(args.PREPROC_DIR, 'test.pkl')
        test_set = read_pkl(test_pkl)
        test_queries_pkl = os.path.join(args.PREPROC_DIR, 'test_queries.pkl')
        test_queries = read_pkl(test_queries_pkl)
        test_dataset = TestDataset(word_to_id, docs, test_queries, test_set)
        test_dataloader = DataLoader(test_dataset, args.valid_batch_size, num_workers=args.num_workers,
                                    collate_fn=test_dataset.collate_fn, pin_memory=True)

    if torch.cuda.is_available():
        device = 'cuda'
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print('using {} device(s): "{}"'.format(torch.cuda.device_count(), dev_name), flush=True)
    else:
        device = 'cpu'
    model = QA_LSTM(args.hidden_dim, args.dropout, id_to_word, args.emb_name, args.emb_dim, False, args.glove_cache, device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters())

    logger = Logger(args.logfile, ['epoch', 'loss', 'dev_map', 'dev_mrr', 'test_map', 'test_mrr'])
    print('training...', flush=True)
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        for batch in tqdm(train_dataloader, desc='epoch {}'.format(epoch + 1)):
            model.zero_grad()
            batch_losses = []

            pos_sims, neg_sims = model(batch)
            for pos_sim, neg_sims_item in zip(pos_sims, neg_sims):    
                # pos_sim is a single number with shape ()
                # add a dimension and expand the shape to (num_neg_examples,)            
                pos_sim = pos_sim.unsqueeze(0).expand(num_neg_examples)
                losses = loss(pos_sim, neg_sims_item, args.margin)

                # take just the example with maximum loss as per the paper
                max_loss = max(losses)
                batch_losses.append(max_loss)
                epoch_losses.append(max_loss.cpu().detach().numpy())

            batch_loss = torch.mean(torch.stack(batch_losses, 0).squeeze(), 0)
            batch_loss.requires_grad_()
            batch_loss.backward()
            optimizer.step()

        epoch_loss = np.mean(epoch_losses)
        print('epoch: {}\tloss: {}'.format(epoch + 1, epoch_loss), flush=True)

        model.eval()
        dev_map, dev_mrr = calc_metrics(model, dev_dataloader, args.mrr_k)
        print('DEV: MAP: {}\tMRR@{}: {}'.format(dev_map, args.mrr_k, dev_mrr), flush=True)

        if args.test:
            test_map, test_mrr = calc_metrics(model, test_dataloader, args.mrr_k)
            print('TEST: MAP: {}\tMRR@{}: {}'.format(test_map, args.mrr_k, test_mrr), flush=True)
        else:
            test_map, test_mrr = None, None
        logger.log([epoch + 1, epoch_loss, dev_map, dev_mrr, test_map, test_mrr])

        # save the module state dict, since we use DataParallel 
        state = {'epoch': epoch + 1, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
        fname = os.path.join(args.ckpt, 'model_{}.pt'.format(epoch + 1))
        print('saving checkpoint in {}'.format(fname), flush=True)
        torch.save(state, fname)


if __name__ == '__main__':
    main()
