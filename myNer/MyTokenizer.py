import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

#加载分词器
tokenizer = AutoTokenizer.from_pretrained('hfl/rbt6')

print(tokenizer)
