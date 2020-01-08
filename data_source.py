import json

import h5py
import torch


def load_vocab(attrs):
    w2i = json.loads(attrs['w2i'])
    i2w = json.loads(attrs['i2w'])
    word_to_index = {word: int(i) for word, i in w2i.items()}
    index_to_word = {int(i): word for i, word in i2w.items()}
    return word_to_index, index_to_word


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, train_file):
        self.fp = h5py.File(train_file, 'r')
        self.queries = self.fp['queries']
        self.pos_docs = self.fp['pos_docs']
        self.neg_docs = self.fp['neg_docs']
        self.word_to_index, self.index_to_word = load_vocab(self.queries.attrs)
        self.num_neg_examples = self.neg_docs.attrs['num_neg_examples']

    def __getitem__(self, index):
        """
        Return an item as
            * the query tensor
            * the positive document tensor
            * a list of negative document tensors
            * a list of negative document lengths
        """
        query = torch.LongTensor(self.queries[index])
        pos_doc = torch.LongTensor(self.pos_docs[index])
        neg_docs, neg_doc_lengths = [], []
        for i in range(self.num_neg_examples):
            neg_doc = self.neg_docs[index * self.num_neg_examples + i]
            neg_docs.append(torch.LongTensor(neg_doc))
            neg_doc_lengths.append(len(neg_doc))
        return query, pos_doc, neg_docs, neg_doc_lengths

    def __len__(self):
        return len(self.queries)

    def collate_fn(self, batch):
        """
        Assemble a training batch. Return
            * a padded batch of queries
            * the original query lengths
            * a padded batch of positive documents, one for each query
            * the original positive document lengths
            * a batch of negative documents, n for each query, shape (batch_size, n, max_length)
        """
        batch_size = len(batch)
        queries, query_lengths, pos_docs, pos_doc_lengths, neg_docs, neg_doc_lengths = [], [], [], [], [], []
        for b_query, b_pos_doc, b_neg_docs, b_neg_doc_lengths in batch:
            queries.append(b_query)
            query_lengths.append(len(b_query))
            pos_docs.append(b_pos_doc)
            pos_doc_lengths.append(len(b_pos_doc))
            neg_docs.extend(b_neg_docs)
            neg_doc_lengths.extend(b_neg_doc_lengths)

        queries = torch.nn.utils.rnn.pad_sequence(queries, batch_first=True)
        query_lengths = torch.LongTensor(query_lengths)
        pos_docs = torch.nn.utils.rnn.pad_sequence(pos_docs, batch_first=True)
        pos_doc_lengths = torch.LongTensor(pos_doc_lengths)
        neg_docs = torch.nn.utils.rnn.pad_sequence(neg_docs, batch_first=True)
        neg_doc_lengths = torch.LongTensor(neg_doc_lengths)

        # reshape negative documents to group the ones for each query
        neg_docs = neg_docs.view(batch_size, self.num_neg_examples, -1)
        neg_doc_lengths = neg_doc_lengths.view(batch_size, self.num_neg_examples)
        return queries, query_lengths, pos_docs, pos_doc_lengths, neg_docs, neg_doc_lengths

    def __del__(self):
        self.fp.close()


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_file):
        self.fp = h5py.File(test_file, 'r')
        self.q_ids = self.fp['q_ids']
        self.queries = self.fp['queries']
        self.docs = self.fp['docs']
        self.labels = self.fp['labels']
        self.word_to_index, self.index_to_word = load_vocab(self.queries.attrs)

    def __getitem__(self, index):
        """
        Return an item as
            * the query tensor
            * the query length
            * the document tensor
            * the document length
            * the query ID
            * the label
        """
        query = torch.LongTensor(self.queries[index])
        doc = torch.LongTensor(self.docs[index])
        return query, len(query), doc, len(doc), self.q_ids[index], self.labels[index]

    def __len__(self):
        return len(self.queries)

    def collate_fn(self, batch):
        """
        Assemble a training batch. Return
            * a padded batch of queries
            * the original query lengths
            * a padded batch of documents, one for each query
            * the original document lengths
            * the query IDs
            * the labels
        """
        queries, query_lengths, docs, doc_lengths, q_ids, labels = [], [], [], [], [], []
        for b_query, b_query_len, b_doc, b_doc_len, b_q_id, b_label in batch:
            queries.append(b_query)
            query_lengths.append(b_query_len)
            docs.append(b_doc)
            doc_lengths.append(b_doc_len)
            q_ids.append(b_q_id)
            labels.append(b_label)

        queries = torch.nn.utils.rnn.pad_sequence(queries, batch_first=True)
        query_lengths = torch.LongTensor(query_lengths)
        docs = torch.nn.utils.rnn.pad_sequence(docs, batch_first=True)
        doc_lengths = torch.LongTensor(doc_lengths)
        return queries, query_lengths, docs, doc_lengths, q_ids, labels

    def __del__(self):
        self.fp.close()
