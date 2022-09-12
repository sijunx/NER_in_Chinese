#数据整理函数
import torch
from MyTokenizer import tokenizer

def collate_fn(data):
    tokens = [i[0] for i in data]
    labels = [i[1] for i in data]

    inputs = tokenizer.batch_encode_plus(tokens,
                                         truncation=True,
                                         padding=True,
                                         return_tensors='pt',
                                         is_split_into_words=True)

    lens = inputs['input_ids'].shape[1]

    # for i in range(len(labels)):
    #     labels[i] = [8] + labels[i]
    #     labels[i] += [8] * lens
    #     labels[i] = labels[i][:lens]

    for i in range(len(labels)):
        labels[i] = [7] + labels[i]
        labels[i] += [7]*lens
        labels[i] = labels[i][:lens]
    #     print("labels[i]:", labels[i])
    #     print("tokens[i]:", tokens[i])
    #
    # print("len(labels):", len(labels))

    return inputs, torch.LongTensor(labels)
