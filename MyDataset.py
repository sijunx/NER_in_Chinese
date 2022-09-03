import torch
from datasets import load_from_disk

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        #names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

        #在线加载数据集
        #dataset = load_dataset(path='peoples_daily_ner', split=split)

        #离线加载数据集
        dataset = load_from_disk(dataset_path='data')[split]

        #过滤掉太长的句子
        def f(data):
            return len(data['tokens']) <= 512 - 2

        dataset = dataset.filter(f)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        tokens = self.dataset[i]['tokens']
        labels = self.dataset[i]['ner_tags']

        return tokens, labels

