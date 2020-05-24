# QA-LSTM-PyTorch

This is an implementation of the (attention based) QA-LSTM model proposed in [Improved Representation Learning for Question Answer Matching](https://www.aclweb.org/anthology/P16-1044/).

## Requirements
This code is tested with Python 3.6.9 and
* torch==1.5.0
* h5py==2.10.0
* numpy==1.18.1
* torchtext==0.6.0
* tqdm==4.46.0
* nltk==3.4.5

## Usage
The following datasets are currently supported:
* [MS MARCO Ranking](http://www.msmarco.org/dataset.aspx)
* [FiQA Task 2](https://sites.google.com/view/fiqa/home)
* [InsuranceQA V2](https://github.com/shuzi/insuranceQA)
* [ANTIQUE](https://ciir.cs.umass.edu/downloads/Antique/)

### Preprocessing
First, preprocess your dataset:
```
usage: preprocess.py [-h] [-vs VOCAB_SIZE] [-n NUM_NEG_EXAMPLES]
                     SAVE {fiqa,antique,msmarco,insrqa} ...

positional arguments:
  SAVE                  Where to save the results
  {fiqa,antique,msmarco,insrqa}
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
Use the training script to train a new model and save checkpoints:
```
usage: train.py [-h] [-en EMB_NAME] [-ed EMB_DIM] [-hd HIDDEN_DIM]
                [-d DROPOUT] [-bs BATCH_SIZE] [-m MARGIN] [-e EPOCHS]
                [--working_dir WORKING_DIR] [--glove_cache GLOVE_CACHE]
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
  --working_dir WORKING_DIR
                        Working directory for checkpoints and logs
  --glove_cache GLOVE_CACHE
                        Word embeddings cache directory
  --random_seed RANDOM_SEED
                        Random seed
```

For example, training a model on the preprocessed FiQA dataset can be done using
```
python3 train.py ~/fiqa_preprocessed --working_dir ~/fiqa_train --glove_cache ~/torchtext_cache
```

### Evaluation
The evaluation script evaluates all saved checkpoints with the preprocessed dev- and testset. The results will be reported in a log file.
```
usage: evaluate.py [-h] [--mrr_k MRR_K] [--batch_size BATCH_SIZE]
                   [--glove_cache GLOVE_CACHE]
                   DATA_DIR WORKING_DIR

positional arguments:
  DATA_DIR              Folder with all preprocessed files
  WORKING_DIR           Working directory

optional arguments:
  -h, --help            show this help message and exit
  --mrr_k MRR_K         Compute MRR@k
  --batch_size BATCH_SIZE
                        Batch size
  --glove_cache GLOVE_CACHE
                        Word embeddings cache directory
```

We can evaluate our trained example model using
```
python evaluate.py ~/fiqa_preprocessed ~/fiqa_train --glove_cache ~/torchtext_cache
```
