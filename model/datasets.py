from pathlib import Path
from typing import Iterable, Tuple

import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab

from qa_utils.lightning.datasets import PointwiseTrainDatasetBase, PairwiseTrainDatasetBase, ValTestDatasetBase


Input = Tuple[torch.LongTensor, torch.IntTensor]
Batch = Tuple[torch.LongTensor, torch.IntTensor, torch.LongTensor, torch.IntTensor]
PointwiseTrainInput = Tuple[Input, int]
PointwiseTrainBatch = Tuple[Batch, torch.FloatTensor]
PairwiseTrainInput = Tuple[Input, Input]
PairwiseTrainBatch = Tuple[Batch, Batch]
ValTestInput = Tuple[int, int, Input, int]
ValTestBatch = Tuple[torch.IntTensor, torch.IntTensor, Batch, torch.IntTensor]


def _get_single_input(query: str, doc: str, vocab: Vocab) -> Input:
    """Tokenize a single (query, document) pair.

    Args:
        query (str): The query
        doc (str): The document
        vocab (Vocab): The vocabulary

    Returns:
        Input: Query and document tokens
    """
    query_tokens = [vocab.stoi[w] for w in nltk.word_tokenize(query.lower())]
    doc_tokens = []
    sentence_lengths = []
    for sentence in nltk.sent_tokenize(doc.lower()):
        sentence_tokens = [vocab.stoi[w] for w in nltk.word_tokenize(sentence)]
        doc_tokens.extend(sentence_tokens)
        sentence_lengths.append(len(sentence_tokens))
    return torch.LongTensor(query_tokens), \
           torch.LongTensor(doc_tokens)


def _collate(inputs: Iterable[Input], pad_id: int) -> Batch:
    """Collate a number of inputs.

    Args:
        inputs (Iterable[Input]): The inputs
        pad_id (int): The padding value

    Returns:
        Batch: Query tokens, query lengths, document tokens, document lengths
    """
    batch_query_tokens, batch_doc_tokens = zip(*inputs)
    query_lengths = [len(x) for x in batch_query_tokens]
    doc_lengths = [len(x) for x in batch_doc_tokens]
    return pad_sequence(batch_query_tokens, batch_first=True, padding_value=pad_id), \
           torch.IntTensor(query_lengths), \
           pad_sequence(batch_doc_tokens, batch_first=True, padding_value=pad_id), \
           torch.IntTensor(doc_lengths)


class PointwiseTrainDataset(PointwiseTrainDatasetBase):
    """Dataset for pointwise training.

    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
        vocab (Vocab): Vocabulary
    """
    def __init__(self, data_file: Path, train_file: Path, vocab: Vocab):
        super().__init__(data_file, train_file)
        self.pad_id = vocab.stoi['<pad>']
        self.vocab = vocab

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc, self.vocab)

    def collate_fn(self, train_inputs: Iterable[PointwiseTrainInput]) -> PointwiseTrainBatch:
        """Collate a number of pointwise inputs.

        Args:
            train_inputs (Iterable[PointwiseTrainInput]): The inputs

        Returns:
            PointwiseTrainBatch: A batch of pointwise inputs
        """
        inputs, labels = zip(*train_inputs)
        return _collate(inputs, self.pad_id), torch.FloatTensor(labels)


class PairwiseTrainDataset(PairwiseTrainDatasetBase):
    """Dataset for pairwise training.

    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
        vocab (Vocab): Vocabulary
    """
    def __init__(self, data_file: Path, train_file: Path, vocab: Vocab):
        super().__init__(data_file, train_file)
        self.pad_id = vocab.stoi['<pad>']
        self.vocab = vocab

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc, self.vocab)

    def collate_fn(self, inputs: Iterable[PairwiseTrainInput]) -> PairwiseTrainBatch:
        """Collate a number of pairwise inputs.

        Args:
            inputs (Iterable[PairwiseTrainInput]): The inputs

        Returns:
            PairwiseTrainBatch: A batch of pairwise inputs
        """
        pos_inputs, neg_inputs = zip(*inputs)
        return _collate(pos_inputs, self.pad_id), _collate(neg_inputs, self.pad_id)


class ValTestDataset(ValTestDatasetBase):
    """Dataset for validation/testing.

    Args:
        data_file (Path): Data file containing queries and documents
        val_test_file (Path): Validationset/testset file
        vocab (Vocab): Vocabulary
    """
    def __init__(self, data_file: Path, val_test_file: Path, vocab: Vocab):
        super().__init__(data_file, val_test_file)
        self.pad_id = vocab.stoi['<pad>']
        self.vocab = vocab

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc, self.vocab)

    def collate_fn(self, val_test_inputs: Iterable[ValTestInput]) -> ValTestBatch:
        """Collate a number of validation/testing inputs.

        Args:
            val_test_inputs (Iterable[ValTestInput]): The inputs

        Returns:
            ValTestBatch: A batch of validation inputs
        """
        q_ids, doc_ids, inputs, labels = zip(*val_test_inputs)
        return torch.IntTensor(q_ids), \
               torch.IntTensor(doc_ids), \
               _collate(inputs, self.pad_id), \
               torch.IntTensor(labels)
