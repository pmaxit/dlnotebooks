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
```

```python
ds = CMUDict()
config = DownloadConfig(cache_dir=os.path.join(str(Path.home()), '.mozhi'))

ds.download_and_prepare(download_config=config)
```

    Using custom data configuration cmu2


    Downloading and preparing dataset cmu_dict/cmu2 to /root/.cache/huggingface/datasets/cmu_dict/cmu2/1.0.0...
    Dataset cmu_dict downloaded and prepared to /root/.cache/huggingface/datasets/cmu_dict/cmu2/1.0.0. Subsequent calls will reuse this data.


```python
train_test = ds.as_dataset(split='train').train_test_split(test_size=0.2)
```

```python
train_test['train'][0]
```




    {'words': 'multiplying',
     'phoneme': ['M', 'AH1', 'L', 'T', 'AH0', 'P', 'L', 'AY2', 'IH0', 'NG']}



```python
# from datasets import DatasetDict
# trainvalid_test = ds.as_dataset(split='train').train_test_split(test_size=0.2)
# train_test = trainvalid_test['train'].train_test_split(test_size=0.1)

# datasets = DatasetDict({
#     "train": train_test["train"],
#     "valid": train_test["test"],
#     "test": trainvalid_test["test"]})
```

Processing data with map inspired by `tf.dataset.map` map method 

```python
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


```

```python
def get_tokens(word_list):
    return (list(w) for w in word_list)
```

```python
next(iter(get_tokens(train_test['train']['words'])))
```




    ['m', 'u', 'l', 't', 'i', 'p', 'l', 'y', 'i', 'n', 'g']



```python
phoneme_vocab = build_vocab_from_iterator(train_test['train']['phoneme'], specials=["<unk>"])
word_vocab = build_vocab_from_iterator(train_test['train']['words'], specials=["<unk>"])
```

```python
word_vocab(['a','b','c'])
```




    [2, 17, 10]



# Model building
