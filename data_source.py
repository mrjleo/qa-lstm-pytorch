import torch


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, word_to_id, docs, train_queries, train_set, num_neg_examples):
        self.word_to_id = word_to_id
        self.docs = docs
        self.train_queries = train_queries
        self.train_set = train_set
        self.num_neg_examples = num_neg_examples
    
    def _get_query_tensor(self, query):
        # 1 is the ID for [UNK]
        result = [self.word_to_id.get(w, 1) for w in query]
        return torch.LongTensor(result), len(result)

    def _get_doc_tensor(self, doc):
        # 1 is the ID for [UNK]
        result = [self.word_to_id.get(w, 1) for w in doc]
        return torch.LongTensor(result), len(result)

    def __getitem__(self, index):
        """
        Return an item as
            * the query tensor
            * the positive document tensor
            * a list of negative document tensors
            * a list of negative document lengths
        """
        q_id, pos_doc_id, neg_doc_ids = self.train_set[index]
        query_t, _ = self._get_query_tensor(self.train_queries[q_id])
        pos_doc_t, _ = self._get_doc_tensor(self.docs[pos_doc_id])

        neg_docs_t, neg_doc_lengths = [], []
        for neg_doc_id in neg_doc_ids:
            neg_doc_t, neg_doc_length = self._get_doc_tensor(self.docs[neg_doc_id])
            neg_docs_t.append(neg_doc_t)
            neg_doc_lengths.append(neg_doc_length)

        return query_t, pos_doc_t, neg_docs_t, neg_doc_lengths
        
    def __len__(self):
        return len(self.train_set)

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


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, word_to_id, docs, test_queries, test_set):
        self.word_to_id = word_to_id
        self.docs = docs
        self.test_queries = test_queries
        self.test_set = test_set

    def _get_query_tensor(self, query):
        # 1 is the ID for [UNK]
        result = [self.word_to_id.get(w, 1) for w in query]
        return torch.LongTensor(result), len(result)

    def _get_doc_tensor(self, doc):
        # 1 is the ID for [UNK]
        result = [self.word_to_id.get(w, 1) for w in doc]
        return torch.LongTensor(result), len(result)

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
        q_id, doc_id, label = self.test_set[index]
        query_t, query_len = self._get_query_tensor(self.test_queries[q_id])
        doc_t, doc_len = self._get_doc_tensor(self.docs[doc_id])
        return query_t, query_len, doc_t, doc_len, q_id, label

    def __len__(self):
        return len(self.test_set)

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
