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

    Downloading and preparing dataset cmu_dict/cmu3 to /root/.cache/huggingface/datasets/cmu_dict/cmu3/1.0.0/a0e598136ef9603a0d6d97059f5e1d2cac789cfe3c0998cb2b4b7fd4198da504...
    Dataset cmu_dict downloaded and prepared to /root/.cache/huggingface/datasets/cmu_dict/cmu3/1.0.0/a0e598136ef9603a0d6d97059f5e1d2cac789cfe3c0998cb2b4b7fd4198da504. Subsequent calls will reuse this data.


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
build_vocab_from_iterator??
```


    [0;31mSignature:[0m
    [0mbuild_vocab_from_iterator[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0miterator[0m[0;34m:[0m [0mIterable[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmin_freq[0m[0;34m:[0m [0mint[0m [0;34m=[0m [0;36m1[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mspecials[0m[0;34m:[0m [0mUnion[0m[0;34m[[0m[0mList[0m[0;34m[[0m[0mstr[0m[0;34m][0m[0;34m,[0m [0mNoneType[0m[0;34m][0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mspecial_first[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0mtorchtext[0m[0;34m.[0m[0mvocab[0m[0;34m.[0m[0mvocab[0m[0;34m.[0m[0mVocab[0m[0;34m[0m[0;34m[0m[0m
    [0;31mSource:[0m   
    [0;32mdef[0m [0mbuild_vocab_from_iterator[0m[0;34m([0m[0miterator[0m[0;34m:[0m [0mIterable[0m[0;34m,[0m [0mmin_freq[0m[0;34m:[0m [0mint[0m [0;34m=[0m [0;36m1[0m[0;34m,[0m [0mspecials[0m[0;34m:[0m [0mOptional[0m[0;34m[[0m[0mList[0m[0;34m[[0m[0mstr[0m[0;34m][0m[0;34m][0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m [0mspecial_first[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mTrue[0m[0;34m)[0m [0;34m->[0m [0mVocab[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m    [0;34m"""[0m
    [0;34m    Build a Vocab from an iterator.[0m
    [0;34m[0m
    [0;34m    Args:[0m
    [0;34m        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.[0m
    [0;34m        min_freq: The minimum frequency needed to include a token in the vocabulary.[0m
    [0;34m        specials: Special symbols to add. The order of supplied tokens will be preserved.[0m
    [0;34m        special_first: Indicates whether to insert symbols at the beginning or at the end.[0m
    [0;34m[0m
    [0;34m[0m
    [0;34m    Returns:[0m
    [0;34m        torchtext.vocab.Vocab: A `Vocab` object[0m
    [0;34m[0m
    [0;34m    Examples:[0m
    [0;34m        >>> #generating vocab from text file[0m
    [0;34m        >>> import io[0m
    [0;34m        >>> from torchtext.vocab import build_vocab_from_iterator[0m
    [0;34m        >>> def yield_tokens(file_path):[0m
    [0;34m        >>>     with io.open(file_path, encoding = 'utf-8') as f:[0m
    [0;34m        >>>         for line in f:[0m
    [0;34m        >>>             yield line.strip().split()[0m
    [0;34m        >>> vocab = build_vocab_from_iterator(yield_tokens_batch(file_path), specials=["<unk>"])[0m
    [0;34m    """[0m[0;34m[0m
    [0;34m[0m[0;34m[0m
    [0;34m[0m    [0mcounter[0m [0;34m=[0m [0mCounter[0m[0;34m([0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0;32mfor[0m [0mtokens[0m [0;32min[0m [0miterator[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m        [0mcounter[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0mtokens[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m[0;34m[0m
    [0;34m[0m    [0;32mif[0m [0mspecials[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m        [0;32mfor[0m [0mtok[0m [0;32min[0m [0mspecials[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m            [0;32mdel[0m [0mcounter[0m[0;34m[[0m[0mtok[0m[0;34m][0m[0;34m[0m
    [0;34m[0m[0;34m[0m
    [0;34m[0m    [0msorted_by_freq_tuples[0m [0;34m=[0m [0msorted[0m[0;34m([0m[0mcounter[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mkey[0m[0;34m=[0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mx[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0msorted_by_freq_tuples[0m[0;34m.[0m[0msort[0m[0;34m([0m[0mkey[0m[0;34m=[0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mx[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m,[0m [0mreverse[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0mordered_dict[0m [0;34m=[0m [0mOrderedDict[0m[0;34m([0m[0msorted_by_freq_tuples[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m[0;34m[0m
    [0;34m[0m    [0;32mif[0m [0mspecials[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m        [0;32mif[0m [0mspecial_first[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m            [0mspecials[0m [0;34m=[0m [0mspecials[0m[0;34m[[0m[0;34m:[0m[0;34m:[0m[0;34m-[0m[0;36m1[0m[0;34m][0m[0;34m[0m
    [0;34m[0m        [0;32mfor[0m [0msymbol[0m [0;32min[0m [0mspecials[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m            [0mordered_dict[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0;34m{[0m[0msymbol[0m[0;34m:[0m [0mmin_freq[0m[0;34m}[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m            [0mordered_dict[0m[0;34m.[0m[0mmove_to_end[0m[0;34m([0m[0msymbol[0m[0;34m,[0m [0mlast[0m[0;34m=[0m[0;32mnot[0m [0mspecial_first[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m[0;34m[0m
    [0;34m[0m    [0mword_vocab[0m [0;34m=[0m [0mvocab[0m[0;34m([0m[0mordered_dict[0m[0;34m,[0m [0mmin_freq[0m[0;34m=[0m[0mmin_freq[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0;32mreturn[0m [0mword_vocab[0m[0;34m[0m[0;34m[0m[0m
    [0;31mFile:[0m      /opt/conda/lib/python3.8/site-packages/torchtext/vocab/vocab_factory.py
    [0;31mType:[0m      function



```python
phoneme_vocab = build_vocab_from_iterator(train_test['train']['phoneme'],specials=['<unk>','<sos>'])
word_vocab = build_vocab_from_iterator(train_test['train']['word'],specials=['<unk>'])
```

```python
phoneme_vocab(['<unk>','<sos>'])
```




    [0, 1]



```python
phoneme_vocab.set_default_index(0)
word_vocab.set_default_index(0)
```

```python
word_vocab.lookup_indices(['a','b','c'])
```




    [2, 17, 10]



# Data Collator

```python
BATCH_SIZE = 8
```

```python
import numpy as np

def process_single_example(word_tokens, phoneme, word_length):
    # Heree you can add variety of operations, Not only is it tokenize
    # The object that this function handles, Namely dataset this data type, adopt featuer
    
    src = word_vocab.lookup_indices(list(word_tokens))
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




    {'src': tensor([17,  7,  8,  8,  4,  6, 15]),
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
    return ''.join(word_vocab.lookup_tokens(lst.numpy()))

def decode_phoneme(lst):
    return ','.join(phoneme_vocab.lookup_tokens(lst.numpy()))

indices = random.sample(range(10,1000), 5 )

for l in indices:
    src = decode_word(ds_processed['train']['src'][l])
    trg = decode_phoneme(ds_processed['train']['trg'][l])
    src_len = ds_processed['train']['src_len'][l]
    print(src, trg, src_len)
```

    benigno B,EH2,N,IY1,N,Y,OW0 tensor(7)
    canilles K,AH0,N,IH1,L,IY0,Z tensor(8)
    hals HH,AE1,L,Z tensor(4)
    pennella P,EH2,N,EH1,L,AH0 tensor(8)
    kardos K,AA1,R,D,OW0,Z tensor(6)


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


```python
input_vocab_size = len(word_vocab)
output_vocab_size  = len(phoneme_vocab)

model = Seq2Seq(input_vocab_size, output_vocab_size,p=0.1)
```

```python
batch = next(iter(dls))
```

```python
trainer.fit(model, train_dataloader=dls)
```

    
      | Name    | Type             | Params
    ---------------------------------------------
    0 | _loss   | CrossEntropyLoss | 0     
    1 | encoder | Encoder          | 14.0 K
    2 | decoder | Decoder          | 17.3 K
    ---------------------------------------------
    31.3 K    Trainable params
    0         Non-trainable params
    31.3 K    Total params
    0.125     Total estimated model params size (MB)
    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/utilities/distributed.py:51: UserWarning: Detected KeyboardInterrupt, attempting graceful shutdown...
      warnings.warn(*args, **kwargs)





    1


