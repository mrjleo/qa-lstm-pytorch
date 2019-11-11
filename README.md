# QA-LSTM-PyTorch

This is an implementation of the (attention based) QA-LSTM model proposed in [Improved Representation Learning for Question Answer Matching](https://www.aclweb.org/anthology/P16-1044/).

## Requirements
This code is tested with Python 3.5.2 and
* torch==1.2.0
* numpy==1.17.0
* torchtext==0.3.1
* tqdm==4.32.2
* nltk==3.4.4

## Usage
The following datasets are currently supported:
* [MS MARCO Ranking](http://www.msmarco.org/dataset.aspx)
* [FiQA Task 2](https://sites.google.com/view/fiqa/home)
* [InsuranceQA V2](https://github.com/shuzi/insuranceQA)
* [WikiPassageQA](https://sites.google.com/site/lyangwww/code-data)

Other datasets can be used by simply adding a preprocessing script.

### Preprocessing
First, preprocess your dataset using the corresponding script. For example,
```
python3 preprocessing/preprocess_fiqa.py ~/fiqa_data preprocessing/fiqa_split/fiqa_split.pkl -n 32 --save ~/fiqa_preprocessed
```
preprocesses the FiQA dataset, sampling 32 negative examples for each query.

### Training
The training script takes care of both training and evaluating on dev- and testset:
```
usage: train.py [-h] [-vs VOCAB_SIZE] [-en EMB_NAME] [-ed EMB_DIM]
                [-hd HIDDEN_DIM] [-d DROPOUT] [-bs BATCH_SIZE] [-m MARGIN]
                [-e EPOCHS] [-vbs VALID_BATCH_SIZE] [-k MRR_K] [--test]
                [--ckpt CKPT] [--logfile LOGFILE] [--glove_cache GLOVE_CACHE]
                [--num_workers NUM_WORKERS] [--random_seed RANDOM_SEED]
                PREPROC_DIR

positional arguments:
  PREPROC_DIR           Directory with preprocessed files

optional arguments:
  -h, --help            show this help message and exit
  -vs VOCAB_SIZE, --vocab_size VOCAB_SIZE
                        Limit vocabulary size
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
  --num_workers NUM_WORKERS
                        Number of DataLoader workers
  --random_seed RANDOM_SEED
                        Random seed
```

For example, training a model on the preprocessed FiQA dataset can be done using
```
python3 train.py ~/fiqa_preprocessed -e 10 -bs 32 -m 0.2 -hd 256 -d 0.5 --ckpt ~/fiqa_ckpt --logfile ~/fiqa.csv -vbs 64 --test --glove_cache ~/torchtext_cache
```

This command trains for 10 epochs and reports metrics on dev- and testset after every epoch.
