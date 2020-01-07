#! /usr/bin/python3


import os
import argparse

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from qa_lstm import QA_LSTM
from data_source import TrainDataset, TestDataset
from functions import calc_metrics, Logger


def loss(s_pos, s_neg, margin):
    return torch.clamp(margin - s_pos + s_neg, min=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('DATA_DIR', help='Directory with preprocessed files')
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
    ap.add_argument('--random_seed', type=int, default=12345, help='Random seed')
    args = ap.parse_args()

    torch.manual_seed(args.random_seed)
    os.makedirs(args.ckpt, exist_ok=True)

    train_file = os.path.join(args.DATA_DIR, 'train.h5')
    train_ds = TrainDataset(train_file)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=train_ds.collate_fn,
                          pin_memory=True)

    dev_file = os.path.join(args.DATA_DIR, 'dev.h5')
    dev_ds = TestDataset(dev_file)
    dev_dl = DataLoader(dev_ds, args.batch_size, shuffle=True, collate_fn=dev_ds.collate_fn,
                        pin_memory=True)

    test_file = os.path.join(args.DATA_DIR, 'test.h5')
    test_ds = TestDataset(test_file)
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=True, collate_fn=test_ds.collate_fn,
                         pin_memory=True)

    if torch.cuda.is_available():
        device = 'cuda'
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print('using {} device(s): "{}"'.format(torch.cuda.device_count(), dev_name), flush=True)
    else:
        device = 'cpu'
    model = QA_LSTM(args.hidden_dim, args.dropout, train_ds.index_to_word, args.emb_name, args.emb_dim, False, args.glove_cache, device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters())

    logger = Logger(args.logfile, ['epoch', 'loss', 'dev_map', 'dev_mrr', 'test_map', 'test_mrr'])
    print('training...', flush=True)
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        for batch in tqdm(train_dl, desc='epoch {}'.format(epoch + 1)):
            model.zero_grad()
            batch_losses = []

            pos_sims, neg_sims = model(batch)
            for pos_sim, neg_sims_item in zip(pos_sims, neg_sims):
                # pos_sim is a single number with shape ()
                # add a dimension and expand the shape to (num_neg_examples,)
                pos_sim = pos_sim.unsqueeze(0).expand(train_ds.num_neg_examples)
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
        dev_map, dev_mrr = calc_metrics(model, dev_dl, args.mrr_k)
        print('DEV: MAP: {}\tMRR@{}: {}'.format(dev_map, args.mrr_k, dev_mrr), flush=True)

        if args.test:
            test_map, test_mrr = calc_metrics(model, test_dl, args.mrr_k)
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
