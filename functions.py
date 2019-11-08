import csv
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def read_pkl(file_path):
    print('reading {}...'.format(file_path), flush=True)
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)


def average_precision(pred, gt):
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(pred):
        if p in gt and p not in pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / max(1.0, len(gt))


def reciprocal_rank(pred, gt, position=None):
    score = 0.0
    for rank, item in enumerate(pred[:position]):
        if item in gt:
            score = 1.0 / (rank + 1.0)
            break
    return score


def get_metrics(scores_list, labels_list, rr_pos=None):
    aps, rrs = [], []
    for scores, labels in zip(scores_list, labels_list):
        rank_indices = np.asarray(scores).argsort()[::-1]
        gt_indices = set(list(np.where(np.asarray(labels) > 0)[0]))
        aps.append(average_precision(rank_indices, gt_indices))
        rrs.append(reciprocal_rank(rank_indices, gt_indices, rr_pos))
    return np.mean(aps), np.mean(rrs)


def calc_metrics(model, dev_dataloader, k):
    scores, labels = defaultdict(list), defaultdict(list)
    for queries, query_lengths, docs, doc_lengths, q_ids, batch_labels in tqdm(dev_dataloader, desc='evaluation'):
        batch = [queries, query_lengths, docs, doc_lengths]
        model_output = [t.cpu().detach().numpy() for t in model(batch)]
        for result, q_id, label in zip(model_output, q_ids, batch_labels):
            scores[q_id].append(result)
            labels[q_id].append(label)
    all_scores, all_labels = [], []
    for q_id in scores:
        all_scores.append(scores[q_id])
        all_labels.append(labels[q_id])
    return get_metrics(all_scores, all_labels, k)


class Logger(object):
    def __init__(self, filename, header):
        self._fp = open(filename, 'w', encoding='utf-8')
        self._writer = csv.writer(self._fp)
        self._writer.writerow(header)
        self._fp.flush()
    
    def log(self, item):
        self._writer.writerow(item)
        self._fp.flush()
    
    def __del__(self):
        self._fp.close()


def get_vocab(word_counts, vocab_size=None):
    word_to_id = {'[PAD]': 0, '[UNK]': 1}
    id_to_word = {0: '[PAD]', 1: '[UNK]'}
    # start with the first unused index
    start_index = len(word_to_id)
    if vocab_size is None:
        k = None
    else:
        k = vocab_size - start_index
    for i, (word, _) in enumerate(word_counts.most_common(k), start_index):
        word_to_id[word] = i
        id_to_word[i] = word

    assert vocab_size is None or vocab_size == len(word_to_id) == len(id_to_word)
    return word_to_id, id_to_word
