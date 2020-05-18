#! /usr/bin/python3


import os
import csv
import argparse

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from qa_lstm import QA_LSTM
from data_source import TrainDataset

from qa_utils.misc import Logger
from qa_utils.io import get_cuda_device
from qa_utils.training import save_args


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
    ap.add_argument('--working_dir', default='train', help='Working directory for checkpoints and logs')
    ap.add_argument('--glove_cache', help='Word embeddings cache directory')
    ap.add_argument('--random_seed', type=int, default=12345, help='Random seed')
    args = ap.parse_args()

    torch.manual_seed(args.random_seed)
    device = get_cuda_device()
    ckpt_dir = os.path.join(args.working_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    # save all args in a file
    args_file = os.path.join(args.working_dir, 'args.csv')
    save_args(args_file, args)

    train_file = os.path.join(args.DATA_DIR, 'train.h5')
    train_ds = TrainDataset(train_file)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=train_ds.collate_fn,
                          pin_memory=True)

    model = QA_LSTM(args.hidden_dim, args.dropout, train_ds.index_to_word, args.emb_name,
                    args.emb_dim, False, args.glove_cache)
    model.to(device)
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters())
    log_file = os.path.join(args.working_dir, 'train.csv')
    logger = Logger(log_file, ['epoch', 'loss'])
    model.train()
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in tqdm(train_dl, desc='epoch {}'.format(epoch)):
            model.zero_grad()
            batch_losses = []
            pos_sims, neg_sims = model(*batch)
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

        logger.log([epoch, np.mean(epoch_losses)])

        # save the module state dict, since we use DataParallel
        state = {'epoch': epoch, 'state_dict': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()}
        fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch))
        torch.save(state, fname)


if __name__ == '__main__':
    main()
