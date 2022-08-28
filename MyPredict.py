import torch


#测试
from MyDataset import Dataset, tokenizer

#数据整理函数
def collate_fn(data):
    tokens = [i[0] for i in data]
    labels = [i[1] for i in data]

    inputs = tokenizer.batch_encode_plus(tokens,
                                         truncation=True,
                                         padding=True,
                                         return_tensors='pt',
                                         is_split_into_words=True)

    lens = inputs['input_ids'].shape[1]

    for i in range(len(labels)):
        labels[i] = [7] + labels[i]
        labels[i] += [7] * lens
        labels[i] = labels[i][:lens]

    return inputs, torch.LongTensor(labels)


def predict():
    # model_load = torch.load('model/命名实体识别_中文.model')
    model_load = torch.load('/Users/zard/Documents/GitHub/NER_in_Chinese/model/命名实体识别_中文.model')
    model_load.eval()
    print("模型加载----------")

    loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (inputs, labels) in enumerate(loader_test):
        break

    with torch.no_grad():
        #[b, lens] -> [b, lens, 8] -> [b, lens]
        outs = model_load(inputs).argmax(dim=2)

    for i in range(32):
        #移除pad
        select = inputs['attention_mask'][i] == 1
        input_id = inputs['input_ids'][i, select]
        out = outs[i, select]
        label = labels[i, select]

        #输出原句子
        print(tokenizer.decode(input_id).replace(' ', ''))

        #输出tag
        for tag in [label, out]:
            s = ''
            for j in range(len(tag)):
                if tag[j] == 0:
                    s += '·'
                    continue
                s += tokenizer.decode(input_id[j])
                s += str(tag[j].item())

            print(s)
        print('==========================')


predict()
