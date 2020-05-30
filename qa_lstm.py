import torch
from torchtext import vocab


class GloveEmbedding(torch.nn.Module):
    def __init__(self, id_to_word, name, dim, freeze=False, cache=None):
        super().__init__()
        self.id_to_word = id_to_word
        self.name = name
        self.dim = dim
        self.glove = vocab.GloVe(name=name, dim=dim, cache=cache)
        weights = self._get_weights()
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze)

    def _get_weights(self):
        weights = []
        for idx in sorted(self.id_to_word):
            word = self.id_to_word[idx]
            if word in self.glove.stoi:
                glove_idx = self.glove.stoi[word]
                weights.append(self.glove.vectors[glove_idx])
            else:
                # initialize randomly
                weights.append(torch.zeros([self.dim]).normal_())
        # this converts a list of tensors to a new tensor
        return torch.stack(weights)

    def forward(self, x):
        return self.embedding(x)


class QA_LSTM(torch.nn.Module):
    def __init__(self, hidden_dim, dropout, id_to_word, emb_name, emb_dim, emb_freeze=False, glove_cache=None):
        super().__init__()
        self.emb = GloveEmbedding(id_to_word, emb_name, emb_dim, emb_freeze, glove_cache)
        self.lstm = torch.nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)

        # attention weights
        self.W_am = torch.nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.W_qm = torch.nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.w_ms = torch.nn.Linear(hidden_dim * 2, 1, bias=False)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

        self.dropout = torch.nn.Dropout(dropout)
        self.cos_sim = torch.nn.CosineSimilarity()

    def _encode(self, inputs, lengths):
        """Encode a batch of padded sequences using the shared LSTM."""
        # this is necessary for multi-GPU support
        # https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        total_length = inputs.size(1)
        input_embed = self.emb(inputs)
        input_seqs = torch.nn.utils.rnn.pack_padded_sequence(input_embed, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(input_seqs)
        out_seqs, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=total_length)
        return out_seqs

    def _max_pool(self, lstm_outputs, lengths):
        """Perform max pooling on the LSTM outputs, masking padding tokens."""
        num_sequences, max_seq_len, num_hidden = lstm_outputs.shape

        # we need to create the mask on the same device
        dev = lstm_outputs.device

        # create mask
        rng = torch.arange(max_seq_len, device=dev).unsqueeze(0).expand(num_sequences, -1)
        rng = rng.unsqueeze(-1).expand(-1, -1, num_hidden)
        lengths = lengths.unsqueeze(1).expand(-1, num_hidden)
        lengths = lengths.unsqueeze(1).expand(-1, max_seq_len, -1)
        mask = rng < lengths

        # set padding outputs to -inf so they dont affect max pooling
        lstm_outputs_clone = lstm_outputs.clone()
        lstm_outputs_clone[~mask] = float('-inf')
        return torch.max(lstm_outputs_clone, 1)[0]

    def _sim(self, a, b):
        """Return the cosine similarity after applying dropout."""
        return self.cos_sim(self.dropout(a), self.dropout(b))

    def _attention(self, query_outputs_pooled, doc_outputs, doc_lengths):
        """Compute attention-weighted outputs for a batch of queries and documents, masking padding tokens."""
        # doc_outputs has shape (num_docs, max_seq_len, 2 * hidden_dim)
        # query_outputs_pooled has shape (num_docs, 2 * hidden_dim)
        # expand its shape so they match
        max_seq_len = doc_outputs.shape[1]
        m = self.tanh(self.W_am(doc_outputs) + self.W_qm(query_outputs_pooled).unsqueeze(1).expand(-1, max_seq_len, -1))
        wm = self.w_ms(m)

        # we need to create the mask on the same device
        dev = wm.device

        # mask the padding tokens before computing the softmax by setting the corresponding values to -inf
        mask = torch.arange(max_seq_len, device=dev)[None, :] < doc_lengths[:, None]
        wm[~mask] = float('-inf')

        s = self.softmax(wm)
        return doc_outputs * s

    def _get_similarities(self, query_outputs_pooled, docs, doc_lengths):
        """For each query, return its similarities to all documents in the corresponding list."""
        sim_list = []
        for query_output_pooled, doc_inputs, doc_lengths in zip(query_outputs_pooled, docs, doc_lengths):
            doc_outputs = self._encode(doc_inputs, doc_lengths)

            # query output has shape (seq_len,)
            # convert to shape (1, seq_len) and expand to (num_docs, seq_len)
            num_docs = doc_outputs.shape[0]
            query_output_expanded = query_output_pooled.unsqueeze(0).expand(num_docs, -1)
            attention = self._attention(query_output_expanded, doc_outputs, doc_lengths)
            attention_pooled = self._max_pool(attention, doc_lengths)

            # finally compute all similarities
            sim_list.append(self._sim(query_output_expanded, attention_pooled))

        # convert to single tensor
        return torch.stack(sim_list)

    def _forward_train(self, queries, query_lengths, pos_docs, pos_doc_lengths, neg_docs,
                       neg_doc_lengths):
        """
        Return the similarities between each query and its positive document and, for each query,
        all similarities to the negative documents.
        """
        query_outputs = self._encode(queries, query_lengths)
        query_outputs_pooled = self._max_pool(query_outputs, query_lengths)
        pos_doc_outputs = self._encode(pos_docs, pos_doc_lengths)

        attention = self._attention(query_outputs_pooled, pos_doc_outputs, pos_doc_lengths)
        attention_pooled = self._max_pool(attention, pos_doc_lengths)

        pos_sims = self._sim(query_outputs_pooled, attention_pooled)
        neg_sims = self._get_similarities(query_outputs_pooled, neg_docs, neg_doc_lengths)
        return pos_sims, neg_sims

    def _forward_test(self, queries, query_lengths, docs, doc_lengths):
        """Return the similarities between all query and document pairs."""
        query_outputs = self._encode(queries, query_lengths)
        query_outputs_pooled = self._max_pool(query_outputs, query_lengths)
        doc_outputs = self._encode(docs, doc_lengths)
        attention = self._attention(query_outputs_pooled, doc_outputs, doc_lengths)
        attention_pooled = self._max_pool(attention, doc_lengths)

        # for testing we need another axis
        return self._sim(query_outputs_pooled, attention_pooled).unsqueeze(1)

    def forward(self, *data):
        """Call _forward_train or _forward_test depending on the model's mode."""
        # for multi-gpu training
        self.lstm.flatten_parameters()
        return self._forward_train(*data) if self.training else self._forward_test(*data)
