import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

#加载分词器
tokenizer = AutoTokenizer.from_pretrained('hfl/rbt6')

print(tokenizer)

#分词测试
tokenizer.batch_encode_plus(
    [[
        '海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间',
        '的', '海', '域', '。'
    ],
     [
         '这', '座', '依', '山', '傍', '水', '的', '博', '物', '馆', '由', '国', '内', '一',
         '流', '的', '设', '计', '师', '主', '持', '设', '计', '，', '整', '个', '建', '筑',
         '群', '精', '美', '而', '恢', '宏', '。'
     ]],
    truncation=True,
    padding=True,
    return_tensors='pt',
    is_split_into_words=True)

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


