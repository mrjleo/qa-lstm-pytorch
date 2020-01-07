# QA-LSTM-PyTorch

This is an implementation of the (attention based) QA-LSTM model proposed in [Improved Representation Learning for Question Answer Matching](https://www.aclweb.org/anthology/P16-1044/).

## Requirements
This code is tested with Python 3.6.9 and
* torch==1.2.0
* h5py==2.10.0
* numpy==1.17.3
* torchtext==0.4.0
* tqdm==4.36.1
* nltk==3.4.5

## Usage
The following datasets are currently supported:
* [MS MARCO Ranking](http://www.msmarco.org/dataset.aspx)
* [FiQA Task 2](https://sites.google.com/view/fiqa/home)
* [InsuranceQA V2](https://github.com/shuzi/insuranceQA)
* [WikiPassageQA](https://sites.google.com/site/lyangwww/code-data)

### Preprocessing
First, preprocess your dataset:
```
usage: preprocess.py [-h] [-vs VOCAB_SIZE] [-n NUM_NEG_EXAMPLES]
                     SAVE {fiqa,msmarco,insrqa,wpqa} ...

positional arguments:
  SAVE                  Where to save the results
  {fiqa,msmarco,insrqa,wpqa}
                        Choose a dataset

optional arguments:
  -h, --help            show this help message and exit
  -vs VOCAB_SIZE, --vocab_size VOCAB_SIZE
                        Vocabulary size
  -n NUM_NEG_EXAMPLES, --num_neg_examples NUM_NEG_EXAMPLES
                        Number of negative examples to sample
```
For example:
```
python3 preprocess.py ~/fiqa_preprocessed fiqa ~/fiqa_data qa_utils/splits/fiqa_split.pkl
```

### Training
The training script takes care of both training and evaluating on dev- and testset:
```
usage: train.py [-h] [-en EMB_NAME] [-ed EMB_DIM] [-hd HIDDEN_DIM]
                [-d DROPOUT] [-bs BATCH_SIZE] [-m MARGIN] [-e EPOCHS]
                [-vbs VALID_BATCH_SIZE] [-k MRR_K] [--test] [--ckpt CKPT]
                [--logfile LOGFILE] [--glove_cache GLOVE_CACHE]
                [--random_seed RANDOM_SEED]
                DATA_DIR

positional arguments:
  DATA_DIR              Directory with preprocessed files

optional arguments:
  -h, --help            show this help message and exit
  -en EMB_NAME, --emb_name EMB_NAME
                        GloVe embedding name
  -ed EMB_DIM, --emb_dim EMB_DIM
                        Word embedding dimension
  -hd HIDDEN_DIM, --hidden_dim HIDDEN_DIM
                        LSTM hidden dimension
  -d DROPOUT, --dropout DROPOUT
                        Dropout rate
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -m MARGIN, --margin MARGIN
                        Margin for loss function
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -vbs VALID_BATCH_SIZE, --valid_batch_size VALID_BATCH_SIZE
                        Validation/testing batch size
  -k MRR_K, --mrr_k MRR_K
                        Compute MRR@k
  --test                Also compute the metrics on the test set
  --ckpt CKPT           Where to save checkpoints
  --logfile LOGFILE     Training log file
  --glove_cache GLOVE_CACHE
                        Word embeddings cache directory
  --random_seed RANDOM_SEED
                        Random seed
```

For example, training a model on the preprocessed FiQA dataset can be done using
```
python3 train.py ~/fiqa_preprocessed -e 10 -bs 32 -m 0.2 -hd 256 -d 0.5 --ckpt ~/fiqa_ckpt --logfile ~/fiqa.csv -vbs 64 --test --glove_cache ~/torchtext_cache
```

This command trains for 10 epochs and reports metrics on dev- and testset after every epoch.
