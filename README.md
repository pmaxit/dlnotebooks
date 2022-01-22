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

    Reusing dataset cmu_dict (/root/.cache/huggingface/datasets/cmu_dict/cmu3/1.0.0/a0e598136ef9603a0d6d97059f5e1d2cac789cfe3c0998cb2b4b7fd4198da504)


```python
train_test = ds['train'].train_test_split(test_size=0.2)
```

    Loading cached split indices for dataset at /root/.cache/huggingface/datasets/cmu_dict/cmu3/1.0.0/a0e598136ef9603a0d6d97059f5e1d2cac789cfe3c0998cb2b4b7fd4198da504/cache-b2d95f15faaf4500.arrow and /root/.cache/huggingface/datasets/cmu_dict/cmu3/1.0.0/a0e598136ef9603a0d6d97059f5e1d2cac789cfe3c0998cb2b4b7fd4198da504/cache-5ed2c4bc219f1934.arrow


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

    108124lines [00:00, 533153.22lines/s]
    108124lines [00:00, 556501.02lines/s]


```python
word_vocab.lookup_indices(['a','b','c'])
```




    [3, 18, 11]



# Data Collator

```python
BATCH_SIZE = 8
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

    Loading cached processed dataset at /root/.cache/huggingface/datasets/cmu_dict/cmu3/1.0.0/a0e598136ef9603a0d6d97059f5e1d2cac789cfe3c0998cb2b4b7fd4198da504/cache-9251eda5a2ed1528.arrow
    Loading cached processed dataset at /root/.cache/huggingface/datasets/cmu_dict/cmu3/1.0.0/a0e598136ef9603a0d6d97059f5e1d2cac789cfe3c0998cb2b4b7fd4198da504/cache-2af5bbd04ed3259a.arrow


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

    spano S,P,AA1,N,OW0 tensor(5)
    crampton K,R,AE1,M,P,T,AH0,N tensor(8)
    dimples D,IH1,M,P,AH0,L,Z tensor(7)
    modality M,AH0,D,AE1,L,AH0,T,IY0 tensor(8)
    receptionists R,IY0,S,EH1,P,SH,AH0,N,IH0,S,T,S tensor(13)


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
    31.4 K    Trainable params
    0         Non-trainable params
    31.4 K    Total params
    0.125     Total estimated model params size (MB)


    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(222)[0;36mtraining_step[0;34m()[0m
    [0;32m    220 [0;31m        [0;32mimport[0m [0mpdb[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    221 [0;31m        [0mpdb[0m[0;34m.[0m[0mset_trace[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m--> 222 [0;31m        [0msrc_seq[0m[0;34m,[0m [0mtrg_seq[0m[0;34m,[0m [0msrc_lengths[0m [0;34m=[0m [0mbatch[0m[0;34m[[0m[0;34m'src'[0m[0;34m][0m[0;34m,[0m[0mbatch[0m[0;34m[[0m[0;34m'trg'[0m[0;34m][0m[0;34m,[0m [0mbatch[0m[0;34m[[0m[0;34m'src_len'[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    223 [0;31m        [0msrc_seq[0m [0;34m=[0m [0msrc_seq[0m[0;34m.[0m[0mtranspose[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    224 [0;31m        [0mtrg_seq[0m [0;34m=[0m [0mtrg_seq[0m[0;34m.[0m[0mtranspose[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m


    ipdb>  n


    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(223)[0;36mtraining_step[0;34m()[0m
    [0;32m    221 [0;31m        [0mpdb[0m[0;34m.[0m[0mset_trace[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    222 [0;31m        [0msrc_seq[0m[0;34m,[0m [0mtrg_seq[0m[0;34m,[0m [0msrc_lengths[0m [0;34m=[0m [0mbatch[0m[0;34m[[0m[0;34m'src'[0m[0;34m][0m[0;34m,[0m[0mbatch[0m[0;34m[[0m[0;34m'trg'[0m[0;34m][0m[0;34m,[0m [0mbatch[0m[0;34m[[0m[0;34m'src_len'[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m--> 223 [0;31m        [0msrc_seq[0m [0;34m=[0m [0msrc_seq[0m[0;34m.[0m[0mtranspose[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    224 [0;31m        [0mtrg_seq[0m [0;34m=[0m [0mtrg_seq[0m[0;34m.[0m[0mtranspose[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    225 [0;31m[0;34m[0m[0m
    [0m


    ipdb>  


    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(224)[0;36mtraining_step[0;34m()[0m
    [0;32m    222 [0;31m        [0msrc_seq[0m[0;34m,[0m [0mtrg_seq[0m[0;34m,[0m [0msrc_lengths[0m [0;34m=[0m [0mbatch[0m[0;34m[[0m[0;34m'src'[0m[0;34m][0m[0;34m,[0m[0mbatch[0m[0;34m[[0m[0;34m'trg'[0m[0;34m][0m[0;34m,[0m [0mbatch[0m[0;34m[[0m[0;34m'src_len'[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    223 [0;31m        [0msrc_seq[0m [0;34m=[0m [0msrc_seq[0m[0;34m.[0m[0mtranspose[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m--> 224 [0;31m        [0mtrg_seq[0m [0;34m=[0m [0mtrg_seq[0m[0;34m.[0m[0mtranspose[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    225 [0;31m[0;34m[0m[0m
    [0m[0;32m    226 [0;31m        [0moutput[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mforward[0m[0;34m([0m[0msrc_seq[0m[0;34m,[0m [0msrc_lengths[0m[0;34m,[0m [0mtrg_seq[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m


    ipdb>  


    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(226)[0;36mtraining_step[0;34m()[0m
    [0;32m    224 [0;31m        [0mtrg_seq[0m [0;34m=[0m [0mtrg_seq[0m[0;34m.[0m[0mtranspose[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    225 [0;31m[0;34m[0m[0m
    [0m[0;32m--> 226 [0;31m        [0moutput[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mforward[0m[0;34m([0m[0msrc_seq[0m[0;34m,[0m [0msrc_lengths[0m[0;34m,[0m [0mtrg_seq[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    227 [0;31m[0;34m[0m[0m
    [0m[0;32m    228 [0;31m        [0;31m# do not know if this is a problem, loss will be computed with sos token[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
    [0m


    ipdb>  b self.forward


    Breakpoint 9 at /notebooks/dlnotebooks/mllib/nlp/seq2seq.py:160


    ipdb>  c


    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(167)[0;36mforward[0;34m()[0m
    [0;32m    165 [0;31m[0;34m[0m[0m
    [0m[0;32m    166 [0;31m[0;34m[0m[0m
    [0m[0;32m--> 167 [0;31m        [0mbatch_size[0m [0;34m=[0m [0msource[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    168 [0;31m        [0mtarget_len[0m [0;34m=[0m [0mtarget[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    169 [0;31m[0;34m[0m[0m
    [0m


    ipdb>  n


    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(168)[0;36mforward[0;34m()[0m
    [0;32m    166 [0;31m[0;34m[0m[0m
    [0m[0;32m    167 [0;31m        [0mbatch_size[0m [0;34m=[0m [0msource[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m--> 168 [0;31m        [0mtarget_len[0m [0;34m=[0m [0mtarget[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    169 [0;31m[0;34m[0m[0m
    [0m[0;32m    170 [0;31m        [0mtarget_vocab_size[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0moutput_dim[0m[0;34m[0m[0;34m[0m[0m
    [0m


    ipdb>  


    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(170)[0;36mforward[0;34m()[0m
    [0;32m    168 [0;31m        [0mtarget_len[0m [0;34m=[0m [0mtarget[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    169 [0;31m[0;34m[0m[0m
    [0m[0;32m--> 170 [0;31m        [0mtarget_vocab_size[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0moutput_dim[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    171 [0;31m[0;34m[0m[0m
    [0m[0;32m    172 [0;31m        [0moutputs[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mzeros[0m[0;34m([0m[0mtarget_len[0m[0;34m,[0m [0mbatch_size[0m[0;34m,[0m [0mtarget_vocab_size[0m[0;34m)[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m


    ipdb>  


    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(172)[0;36mforward[0;34m()[0m
    [0;32m    170 [0;31m        [0mtarget_vocab_size[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0moutput_dim[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    171 [0;31m[0;34m[0m[0m
    [0m[0;32m--> 172 [0;31m        [0moutputs[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mzeros[0m[0;34m([0m[0mtarget_len[0m[0;34m,[0m [0mbatch_size[0m[0;34m,[0m [0mtarget_vocab_size[0m[0;34m)[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    173 [0;31m[0;34m[0m[0m
    [0m[0;32m    174 [0;31m[0;34m[0m[0m
    [0m


    ipdb>  


    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(176)[0;36mforward[0;34m()[0m
    [0;32m    174 [0;31m[0;34m[0m[0m
    [0m[0;32m    175 [0;31m[0;34m[0m[0m
    [0m[0;32m--> 176 [0;31m        [0mhidden[0m[0;34m,[0m [0mcell[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mencoder[0m[0;34m([0m[0msource[0m[0;34m,[0m [0msource_len[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    177 [0;31m[0;34m[0m[0m
    [0m[0;32m    178 [0;31m        [0;31m# mask = [batch_size, src len][0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
    [0m


    ipdb>  


    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(181)[0;36mforward[0;34m()[0m
    [0;32m    179 [0;31m        [0;31m# without sos token at the beginning and eos token at the end[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    180 [0;31m[0;34m[0m[0m
    [0m[0;32m--> 181 [0;31m        [0mx[0m [0;34m=[0m [0mtarget[0m[0;34m[[0m[0;36m0[0m[0;34m,[0m[0;34m:[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    182 [0;31m[0;34m[0m[0m
    [0m[0;32m    183 [0;31m        [0;32mfor[0m [0mt[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mtarget_len[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
    [0m


    ipdb>  


    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(183)[0;36mforward[0;34m()[0m
    [0;32m    181 [0;31m        [0mx[0m [0;34m=[0m [0mtarget[0m[0;34m[[0m[0;36m0[0m[0;34m,[0m[0;34m:[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    182 [0;31m[0;34m[0m[0m
    [0m[0;32m--> 183 [0;31m        [0;32mfor[0m [0mt[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mtarget_len[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    184 [0;31m            [0moutput[0m[0;34m,[0m [0mhidden[0m[0;34m,[0m [0mcell[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdecoder[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mhidden[0m[0;34m,[0m [0mcell[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    185 [0;31m[0;34m[0m[0m
    [0m


    ipdb>  n


    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(184)[0;36mforward[0;34m()[0m
    [0;32m    182 [0;31m[0;34m[0m[0m
    [0m[0;32m    183 [0;31m        [0;32mfor[0m [0mt[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mtarget_len[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m--> 184 [0;31m            [0moutput[0m[0;34m,[0m [0mhidden[0m[0;34m,[0m [0mcell[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdecoder[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mhidden[0m[0;34m,[0m [0mcell[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    185 [0;31m[0;34m[0m[0m
    [0m[0;32m    186 [0;31m            [0moutputs[0m[0;34m[[0m[0mt[0m[0;34m][0m [0;34m=[0m [0moutput[0m[0;34m[0m[0;34m[0m[0m
    [0m


    ipdb>  n


    TypeError: linear(): argument 'input' (position 1) must be Tensor, not tuple
    > [0;32m/notebooks/dlnotebooks/mllib/nlp/seq2seq.py[0m(184)[0;36mforward[0;34m()[0m
    [0;32m    182 [0;31m[0;34m[0m[0m
    [0m[0;32m    183 [0;31m        [0;32mfor[0m [0mt[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mtarget_len[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m--> 184 [0;31m            [0moutput[0m[0;34m,[0m [0mhidden[0m[0;34m,[0m [0mcell[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdecoder[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mhidden[0m[0;34m,[0m [0mcell[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    185 [0;31m[0;34m[0m[0m
    [0m[0;32m    186 [0;31m            [0moutputs[0m[0;34m[[0m[0mt[0m[0;34m][0m [0;34m=[0m [0moutput[0m[0;34m[0m[0;34m[0m[0m
    [0m


    ipdb>  exit()



    ---------------------------------------------------------------------------

    BdbQuit                                   Traceback (most recent call last)

    <ipython-input-55-9697e6b32906> in <module>
    ----> 1 trainer.fit(model, train_dataloader=dls)
    

    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py in fit(self, model, train_dataloader, val_dataloaders, datamodule)
        496 
        497         # dispath `start_training` or `start_testing` or `start_predicting`
    --> 498         self.dispatch()
        499 
        500         # plugin will finalized fitting (e.g. ddp_spawn will load trained model)


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py in dispatch(self)
        543 
        544         else:
    --> 545             self.accelerator.start_training(self)
        546 
        547     def train_or_test_or_predict(self):


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/accelerators/accelerator.py in start_training(self, trainer)
         71 
         72     def start_training(self, trainer):
    ---> 73         self.training_type_plugin.start_training(trainer)
         74 
         75     def start_testing(self, trainer):


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py in start_training(self, trainer)
        112     def start_training(self, trainer: 'Trainer') -> None:
        113         # double dispatch to initiate the training loop
    --> 114         self._results = trainer.run_train()
        115 
        116     def start_testing(self, trainer: 'Trainer') -> None:


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py in run_train(self)
        634                 with self.profiler.profile("run_training_epoch"):
        635                     # run train epoch
    --> 636                     self.train_loop.run_training_epoch()
        637 
        638                 if self.max_steps and self.max_steps <= self.global_step:


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/training_loop.py in run_training_epoch(self)
        491             # ------------------------------------
        492             with self.trainer.profiler.profile("run_training_batch"):
    --> 493                 batch_output = self.run_training_batch(batch, batch_idx, dataloader_idx)
        494 
        495             # when returning -1 from train_step, we end epoch early


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/training_loop.py in run_training_batch(self, batch, batch_idx, dataloader_idx)
        653 
        654                         # optimizer step
    --> 655                         self.optimizer_step(optimizer, opt_idx, batch_idx, train_step_and_backward_closure)
        656 
        657                     else:


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/training_loop.py in optimizer_step(self, optimizer, opt_idx, batch_idx, train_step_and_backward_closure)
        424 
        425         # model hook
    --> 426         model_ref.optimizer_step(
        427             self.trainer.current_epoch,
        428             batch_idx,


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/core/lightning.py in optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs)
       1383             # wraps into LightingOptimizer only for running step
       1384             optimizer = LightningOptimizer._to_lightning_optimizer(optimizer, self.trainer, optimizer_idx)
    -> 1385         optimizer.step(closure=optimizer_closure)
       1386 
       1387     def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/core/optimizer.py in step(self, closure, *args, **kwargs)
        212             profiler_name = f"optimizer_step_and_closure_{self._optimizer_idx}"
        213 
    --> 214         self.__optimizer_step(*args, closure=closure, profiler_name=profiler_name, **kwargs)
        215         self._total_optimizer_step_calls += 1
        216 


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/core/optimizer.py in __optimizer_step(self, closure, profiler_name, **kwargs)
        132 
        133         with trainer.profiler.profile(profiler_name):
    --> 134             trainer.accelerator.optimizer_step(optimizer, self._optimizer_idx, lambda_closure=closure, **kwargs)
        135 
        136     def step(self, *args, closure: Optional[Callable] = None, **kwargs):


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/accelerators/accelerator.py in optimizer_step(self, optimizer, opt_idx, lambda_closure, **kwargs)
        275         )
        276         if make_optimizer_step:
    --> 277             self.run_optimizer_step(optimizer, opt_idx, lambda_closure, **kwargs)
        278         self.precision_plugin.post_optimizer_step(optimizer, opt_idx)
        279         self.training_type_plugin.post_optimizer_step(optimizer, opt_idx, **kwargs)


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/accelerators/accelerator.py in run_optimizer_step(self, optimizer, optimizer_idx, lambda_closure, **kwargs)
        280 
        281     def run_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int, lambda_closure: Callable, **kwargs):
    --> 282         self.training_type_plugin.optimizer_step(optimizer, lambda_closure=lambda_closure, **kwargs)
        283 
        284     def optimizer_zero_grad(self, current_epoch: int, batch_idx: int, optimizer: Optimizer, opt_idx: int) -> None:


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py in optimizer_step(self, optimizer, lambda_closure, **kwargs)
        161 
        162     def optimizer_step(self, optimizer: torch.optim.Optimizer, lambda_closure: Callable, **kwargs):
    --> 163         optimizer.step(closure=lambda_closure, **kwargs)
    

    /opt/conda/lib/python3.8/site-packages/torch/optim/lr_scheduler.py in wrapper(*args, **kwargs)
         63                 instance._step_count += 1
         64                 wrapped = func.__get__(instance, cls)
    ---> 65                 return wrapped(*args, **kwargs)
         66 
         67             # Note that the returned function here is no longer a bound method,


    /opt/conda/lib/python3.8/site-packages/torch/optim/optimizer.py in wrapper(*args, **kwargs)
         87                 profile_name = "Optimizer.step#{}.step".format(obj.__class__.__name__)
         88                 with torch.autograd.profiler.record_function(profile_name):
    ---> 89                     return func(*args, **kwargs)
         90             return wrapper
         91 


    /opt/conda/lib/python3.8/site-packages/torch/autograd/grad_mode.py in decorate_context(*args, **kwargs)
         25         def decorate_context(*args, **kwargs):
         26             with self.__class__():
    ---> 27                 return func(*args, **kwargs)
         28         return cast(F, decorate_context)
         29 


    /opt/conda/lib/python3.8/site-packages/torch/optim/adamw.py in step(self, closure)
         63         if closure is not None:
         64             with torch.enable_grad():
    ---> 65                 loss = closure()
         66 
         67         for group in self.param_groups:


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/training_loop.py in train_step_and_backward_closure()
        647 
        648                         def train_step_and_backward_closure():
    --> 649                             result = self.training_step_and_backward(
        650                                 split_batch, batch_idx, opt_idx, optimizer, self.trainer.hiddens
        651                             )


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/training_loop.py in training_step_and_backward(self, split_batch, batch_idx, opt_idx, optimizer, hiddens)
        741         with self.trainer.profiler.profile("training_step_and_backward"):
        742             # lightning module hook
    --> 743             result = self.training_step(split_batch, batch_idx, opt_idx, hiddens)
        744             self._curr_step_result = result
        745 


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/training_loop.py in training_step(self, split_batch, batch_idx, opt_idx, hiddens)
        291             model_ref._results = Result()
        292             with self.trainer.profiler.profile("training_step"):
    --> 293                 training_step_output = self.trainer.accelerator.training_step(args)
        294                 self.trainer.accelerator.post_training_step()
        295 


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/accelerators/accelerator.py in training_step(self, args)
        154 
        155         with self.precision_plugin.train_step_context(), self.training_type_plugin.train_step_context():
    --> 156             return self.training_type_plugin.training_step(*args)
        157 
        158     def post_training_step(self):


    /opt/conda/lib/python3.8/site-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py in training_step(self, *args, **kwargs)
        123 
        124     def training_step(self, *args, **kwargs):
    --> 125         return self.lightning_module.training_step(*args, **kwargs)
        126 
        127     def post_training_step(self):


    /notebooks/dlnotebooks/mllib/nlp/seq2seq.py in training_step(self, batch, batch_idx)
        224         trg_seq = trg_seq.transpose(0, 1)
        225 
    --> 226         output = self.forward(src_seq, src_lengths, trg_seq)
        227 
        228         # do not know if this is a problem, loss will be computed with sos token


    /notebooks/dlnotebooks/mllib/nlp/seq2seq.py in forward(self, source, source_len, target, teacher_force_ratio)
        182 
        183         for t in range(1, target_len):
    --> 184             output, hidden, cell = self.decoder(x, hidden, cell)
        185 
        186             outputs[t] = output


    /opt/conda/lib/python3.8/bdb.py in trace_dispatch(self, frame, event, arg)
         92             return self.dispatch_return(frame, arg)
         93         if event == 'exception':
    ---> 94             return self.dispatch_exception(frame, arg)
         95         if event == 'c_call':
         96             return self.trace_dispatch


    /opt/conda/lib/python3.8/bdb.py in dispatch_exception(self, frame, arg)
        172                     and arg[0] is StopIteration and arg[2] is None):
        173                 self.user_exception(frame, arg)
    --> 174                 if self.quitting: raise BdbQuit
        175         # Stop at the StopIteration or GeneratorExit exception when the user
        176         # has set stopframe in a generator by issuing a return command, or a


    BdbQuit: 


```python
!nvidia-smi
```

    Fri Jan 21 23:27:59 2022       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Quadro RTX 4000     Off  | 00000000:00:05.0 Off |                  N/A |
    | 30%   46C    P0    29W / 125W |   1044MiB /  7982MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    +-----------------------------------------------------------------------------+

