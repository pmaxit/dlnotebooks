# Title



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

    Using custom data configuration cmu1


    Downloading and preparing dataset cmu_dict/cmu1 to /root/.cache/huggingface/datasets/cmu_dict/cmu1/1.0.0...
    Dataset cmu_dict downloaded and prepared to /root/.cache/huggingface/datasets/cmu_dict/cmu1/1.0.0. Subsequent calls will reuse this data.


```python
from datasets import DatasetDict
```

```python
train_test = ds.as_dataset(split='train').train_test_split(test_size=0.2)
```

    Loading cached split indices for dataset at /root/.cache/huggingface/datasets/cmu_dict/cmu1/1.0.0/cache-852d25264fde6bec.arrow and /root/.cache/huggingface/datasets/cmu_dict/cmu1/1.0.0/cache-3b5edbe869b4e4b0.arrow


```python
train_test
```




    DatasetDict({
        train: Dataset({
            features: ['words', 'phoneme'],
            num_rows: 108124
        })
        test: Dataset({
            features: ['words', 'phoneme'],
            num_rows: 27031
        })
    })



```python
# trainvalid_test = ds.as_dataset(split='train').train_test_split(test_size=0.2)
# train_test = trainvalid_test['train'].train_test_split(test_size=0.1)

# datasets = DatasetDict({
#     "train": train_test["train"],
#     "valid": train_test["test"],
#     "test": trainvalid_test["test"]})
```
