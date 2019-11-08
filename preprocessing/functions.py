import random
import os
import pickle
import gzip
from collections import Counter

from tqdm import tqdm


def count_lines(file_path):
    if file_path.endswith('.gz'):
        with gzip.open(file_path) as fp:
            for i, _ in enumerate(fp):
                pass
    else:
        with open(file_path, encoding='utf-8') as fp:
            for i, _ in enumerate(fp):
                pass
    return i + 1


def get_word_counter(sentences):
    words = []
    for s in tqdm(sentences):
        words.extend(s)
    return Counter(words)


def save_pkl(obj, pkl_path):
    print('saving {}...'.format(pkl_path), flush=True)
    with open(pkl_path, 'wb') as fp:
        pickle.dump(obj, fp)


def save_pkl_iter(iterator, pkl_path):
    print('saving {}...'.format(pkl_path), flush=True)
    with open(pkl_path, 'wb') as fp:
        pickle.dump(list(tqdm(iterator)), fp)


class TrainDataset(object):
    def __init__(self, train_queries, docs, train_qrels, num_neg_examples):
        self.train_queries = train_queries
        self.docs = docs
        self.train_qrels = train_qrels
        self.num_neg_examples = num_neg_examples

        # enumerate all positive (query, document) pairs
        self.pos_pairs = []
        for q_id in train_queries:
            # empty queries or documents will cause errors
            if len(train_queries.get(q_id, [])) == 0:
                continue
            for doc_id in train_qrels[q_id]:
                if len(docs.get(doc_id, [])) == 0:
                    continue
                self.pos_pairs.append((q_id, doc_id))

        # all positive docs that are used during training
        self.total_items = len(self.pos_pairs)

        # a list of all doc ids to sample negatives from
        self.neg_sample_doc_ids = set()
        for doc_id, doc in docs.items():
            if len(doc) > 0:
                self.neg_sample_doc_ids.add(doc_id)

    def _sample_negatives(self, q_id):
        population = self.neg_sample_doc_ids.copy()
        # the IDs of the docs that are relevant for this query (we can't use these as negatives)
        for doc_id in self.train_qrels[q_id]:
            if doc_id in population:
                population.remove(doc_id)
        return random.sample(population, self.num_neg_examples)

    def _get_train_examples(self):
        for q_id, pos_doc_id in self.pos_pairs:
            neg_ids = self._sample_negatives(q_id)
            yield q_id, pos_doc_id, neg_ids

    def __len__(self):
        return self.total_items

    def __iter__(self):
        yield from self._get_train_examples()


class TestDataset(object):
    def __init__(self, test_queries, docs, test_set):
        self.test_queries = test_queries
        self.docs = docs
        self.test_set = test_set

        self.total_items = sum(map(len, test_set.values()))

    def _get_test_examples(self):
        for q_id, doc_ids in self.test_set.items():
            # empty/nonexistent queries or documents will cause errors
            if len(self.test_queries.get(q_id, [])) == 0:
                continue
            for doc_id, label in doc_ids:
                if len(self.docs[doc_id]) > 0:
                    yield q_id, doc_id, label

    def __len__(self):
        return self.total_items

    def __iter__(self):
        yield from self._get_test_examples()


def save(save_path, word_counter, num_neg_examples, docs,
         train_dataset, train_queries,
         dev_dataset, dev_queries,
         test_dataset=None, test_queries=None):
    info_pkl = os.path.join(save_path, 'info.pkl')
    save_pkl((word_counter, num_neg_examples), info_pkl)

    docs_pkl = os.path.join(save_path, 'docs.pkl')
    save_pkl(docs, docs_pkl)

    train_queries_pkl = os.path.join(save_path, 'train_queries.pkl')
    save_pkl(train_queries, train_queries_pkl)
    train_pkl = os.path.join(save_path, 'train.pkl')
    save_pkl_iter(train_dataset, train_pkl)

    dev_queries_pkl = os.path.join(save_path, 'dev_queries.pkl')
    save_pkl(dev_queries, dev_queries_pkl)
    dev_pkl = os.path.join(save_path, 'dev.pkl')
    save_pkl_iter(dev_dataset, dev_pkl)

    if test_dataset is not None:
        test_queries_pkl = os.path.join(save_path, 'test_queries.pkl')
        save_pkl(test_queries, test_queries_pkl)
        test_pkl = os.path.join(save_path, 'test.pkl')
        save_pkl_iter(test_dataset, test_pkl)
