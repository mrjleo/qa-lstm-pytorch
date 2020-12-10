# QA-LSTM-PyTorch
This is an implementation of the (attention based) QA-LSTM model proposed in [Improved Representation Learning for Question Answer Matching](https://www.aclweb.org/anthology/P16-1044/).

## Requirements
This code is tested with Python 3.8.3 and
* torch==1.7.0
* pytorch-lightning==1.1.0
* h5py==2.10.0
* numpy==1.18.5
* torchtext==0.8.0
* tqdm==4.48.0
* nltk==3.5

## Cloning
Clone this repository using `git clone --recursive` to get the submodule.

## Usage
The following datasets are currently supported:
* [ANTIQUE](https://ciir.cs.umass.edu/downloads/Antique/)
* [FiQA Task 2](https://sites.google.com/view/fiqa/home)
* [MS MARCO Passage Ranking](https://microsoft.github.io/TREC-2019-Deep-Learning/)
* [InsuranceQA V2](https://github.com/shuzi/insuranceQA)

### Preprocessing
First, preprocess your dataset:
```
usage: preprocess.py [-h] [--num_negatives NUM_NEGATIVES] [--pw_num_negatives PW_NUM_NEGATIVES] [--pw_query_limit PW_QUERY_LIMIT] [--random_seed RANDOM_SEED]
                     SAVE {antique,fiqa,insuranceqa,msmarco} ...

positional arguments:
  SAVE                  Where to save the results
  {antique,fiqa,insuranceqa,msmarco}
                        Choose a dataset

optional arguments:
  -h, --help            show this help message and exit
  --num_negatives NUM_NEGATIVES
                        Number of negatives per positive (pointwise training) (default: 1)
  --pw_num_negatives PW_NUM_NEGATIVES
                        Number of negatives per positive (pairwise training) (default: 16)
  --pw_query_limit PW_QUERY_LIMIT
                        Maximum number of training examples per query (pairwise training) (default: 64)
  --random_seed RANDOM_SEED
                        Random seed (default: 123)
```

Next, create a vocabulary:
```
usage: create_vocab.py [-h] [--max_size MAX_SIZE] [--cache CACHE] [--vectors VECTORS] [--out_file OUT_FILE] DATA_FILE

positional arguments:
  DATA_FILE            File that holds the queries and documents

optional arguments:
  -h, --help           show this help message and exit
  --max_size MAX_SIZE  Maximum vocabulary size (default: None)
  --cache CACHE        Torchtext cache (default: None)
  --vectors VECTORS    Pre-trained vectors (default: glove.840B.300d)
  --out_file OUT_FILE  Where to save the vocabulary (default: vocab.pkl)
```

### Training and Evaluation
Use the training script to train a new model and save checkpoints:
```
usage: train.py [-h] [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--max_epochs MAX_EPOCHS] [--gpus GPUS [GPUS ...]] [--val_check_interval VAL_CHECK_INTERVAL]
                [--save_top_k SAVE_TOP_K] [--limit_val_batches LIMIT_VAL_BATCHES] [--limit_train_batches LIMIT_TRAIN_BATCHES] [--limit_test_batches LIMIT_TEST_BATCHES]
                [--precision {16,32}] [--distributed_backend DISTRIBUTED_BACKEND] [--hidden_dim HIDDEN_DIM] [--dropout DROPOUT] [--lr LR] [--loss_margin LOSS_MARGIN]
                [--batch_size BATCH_SIZE] [--training_mode {pointwise,pairwise}] [--val_patience VAL_PATIENCE] [--save_dir SAVE_DIR] [--random_seed RANDOM_SEED]
                [--load_weights LOAD_WEIGHTS] [--test]
                DATA_DIR VOCAB

positional arguments:
  DATA_DIR              Folder with all preprocessed files
  VOCAB                 Vocabulary file

optional arguments:
  -h, --help            show this help message and exit
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Update weights after this many batches (default: 1)
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs (default: 20)
  --gpus GPUS [GPUS ...]
                        GPU IDs to train on (default: None)
  --val_check_interval VAL_CHECK_INTERVAL
                        Validation check interval (default: 1.0)
  --save_top_k SAVE_TOP_K
                        Save top-k checkpoints (default: 1)
  --limit_val_batches LIMIT_VAL_BATCHES
                        Use a subset of validation data (default: 9223372036854775807)
  --limit_train_batches LIMIT_TRAIN_BATCHES
                        Use a subset of training data (default: 9223372036854775807)
  --limit_test_batches LIMIT_TEST_BATCHES
                        Use a subset of test data (default: 9223372036854775807)
  --precision {16,32}   Floating point precision (default: 32)
  --distributed_backend DISTRIBUTED_BACKEND
                        Distributed backend (default: ddp)
  --hidden_dim HIDDEN_DIM
                        The hidden dimensions throughout the model (default: 256)
  --dropout DROPOUT     Dropout percentage (default: 0.5)
  --lr LR               Learning rate (default: 0.001)
  --loss_margin LOSS_MARGIN
                        Hinge loss margin (default: 0.2)
  --batch_size BATCH_SIZE
                        Batch size (default: 32)
  --training_mode {pointwise,pairwise}
                        Training mode (default: pairwise)
  --val_patience VAL_PATIENCE
                        Validation patience (default: 3)
  --save_dir SAVE_DIR   Directory for logs, checkpoints and predictions (default: out)
  --random_seed RANDOM_SEED
                        Random seed (default: 123)
  --load_weights LOAD_WEIGHTS
                        Load pre-trained weights before training (default: None)
  --test                Test the model after training (default: False)
```

Use the `--test` argument to run the model on the testset using the best checkpoint after training.
