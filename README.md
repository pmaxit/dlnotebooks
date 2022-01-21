# Data Processing



```python
%load_ext autoreload
%autoreload 2
```

```python
import sys
sys.path.insert(0,'../')
```

```python
from mllib.nlp.datasets.cmudict import CMUDict
from datasets import load_dataset
from mllib.nlp.seq2seq import Seq2Seq
```

```python
ds = load_dataset('/notebooks/dlnotebooks/mllib/nlp/datasets/cmudict.py')
```

    Downloading and preparing dataset cmu_dict/cmu2 to /root/.cache/huggingface/datasets/cmu_dict/cmu2/1.0.0/3b3904aac9acadebed008a26558832f94749da39e2cd1ecee825720fd34a1da1...
    Dataset cmu_dict downloaded and prepared to /root/.cache/huggingface/datasets/cmu_dict/cmu2/1.0.0/3b3904aac9acadebed008a26558832f94749da39e2cd1ecee825720fd34a1da1. Subsequent calls will reuse this data.


```python
train_test = ds['train'].train_test_split(test_size=0.2)
```

```python
train_test['train'][0]
```




    {'word': 'preempted',
     'word_length': 9,
     'phoneme': ['P', 'R', 'IY0', 'EH1', 'M', 'P', 'T', 'IH0', 'D']}



Processing data with map inspired by `tf.dataset.map` map method 

```python
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


```

```python
train_test['train']['word'][0], train_test['train']['phoneme'][0] 
```




    ('preempted', ['P', 'R', 'IY0', 'EH1', 'M', 'P', 'T', 'IH0', 'D'])



```python
phoneme_vocab = build_vocab_from_iterator(train_test['train']['phoneme'])
word_vocab = build_vocab_from_iterator(train_test['train']['word'])
```

    108124lines [00:00, 531019.42lines/s]
    108124lines [00:00, 547361.56lines/s]


```python
word_vocab.lookup_indices(['a','b','c'])
```




    [3, 18, 11]



# Data Collator

```python
BATCH_SIZE = 32
```

```python
import numpy as np

def process_single_example(word_tokens, phoneme, word_length):
    # Heree you can add variety of operations, Not only is it tokenize
    # The object that this function handles, Namely dataset this data type, adopt featuer
    
    src = word_vocab.lookup_indices(word_tokens)
    trg = phoneme_vocab.lookup_indices(phoneme)

    return src, trg, word_length

def collate_batch(batch):
    
    batch_size = len(batch['word'])
    out = [process_single_example(*tokens) for tokens in zip(batch['word'], batch['phoneme'], batch['word_length'])]
    

    return {
        'src': [b[0] for b in out],
        'trg': [b[1] for b in out],
        'src_len': [b[2] for b in out],
    }
```

```python
ds_processed = train_test.map(collate_batch, remove_columns=['word','word_length','phoneme'], 
                        batch_size= BATCH_SIZE,
                           batched=True).with_format('pytorch', output_all_columns=True)
```

```python
ds_processed['train'][2]
```




    {'src': tensor([18,  8,  9,  9,  5,  7, 16]),
     'trg': tensor([15, 19,  6, 10, 25]),
     'src_len': tensor(7)}



```python
from torch.utils.data import DataLoader
```

```python
def pad_collate(batch):
    
    def pad(xs):
        return torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    
    src = [b['src'] for b in batch]
    trg = [b['trg'] for b in batch]
    src_len = [b['src_len'] for b in batch]
    
    return {
        'src': pad(src), 
        'trg' : pad(trg), 
        'src_len' : src_len
    }

```

```python
dls = DataLoader(ds_processed['train'], shuffle=True, collate_fn=pad_collate, batch_size=32)
```

```python
#next(iter(dls))
```

```python
from transformers import DataCollatorWithPadding, default_data_collator
```

```python
import random
# checking
def decode_word(lst):
    return ''.join([word_vocab.itos[l] for l in lst])

def decode_phoneme(lst):
    return ','.join([phoneme_vocab.itos[l] for l in lst])

indices = random.sample(range(10,1000), 5 )

for l in indices:
    src = decode_word(ds_processed['train']['src'][l])
    trg = decode_phoneme(ds_processed['train']['trg'][l])
    src_len = ds_processed['train']['src_len'][l]
    print(src, trg, src_len)
```

    rids R,IH1,D,Z tensor(4)
    dershem D,ER1,SH,IH0,M tensor(7)
    botello B,OW0,T,EH1,L,OW0 tensor(7)
    brimmed B,R,IH1,M,D tensor(7)
    blinds B,L,AY1,N,D,Z tensor(6)


# Model building

```python
import pytorch_lightning as pl
```

```python
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

```

```python
lr_monitor = LearningRateMonitor(logging_interval='step')
logger = TensorBoardLogger('tb_logs', name='my_model')
trainer = pl.Trainer(callbacks=[lr_monitor],max_epochs=1, gpus=1, logger=[logger])
```

    GPU available: True, used: True
    TPU available: None, using: 0 TPU cores

