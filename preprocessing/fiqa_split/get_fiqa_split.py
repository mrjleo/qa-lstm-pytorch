import pickle
from collections import defaultdict


# https://github.com/namkhanhtran/nn4nqa/tree/master/data/fiqa/processed


def read_pickled_file(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def main():
    train_pairs = read_pickled_file('train_sample.pkl')
    dev_candits = read_pickled_file('dev_candidates.pkl')
    test_candits = read_pickled_file('test_candidates.pkl')

    train_set = set(qid for qid, _ in train_pairs)
    dev_set = defaultdict(list)
    for qid, docid, label in dev_candits:
        dev_set[qid].append((docid, label))
    
    test_set = defaultdict(list)
    for qid, docid, label in test_candits:
        test_set[qid].append((docid, label))

    with open('fiqa_split.pkl', 'wb') as fp:
        pickle.dump((train_set, dev_set, test_set), fp)


if __name__ == '__main__':
    main()
