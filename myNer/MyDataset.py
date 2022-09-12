import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        #names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

        #离线加载数据集
        ori_data = []
        if split == 'train':
            ori_data = load_data('/Users/zard/Documents/GitHub/NER_in_Chinese/data/china-people-daily-ner-corpus/example.train')
        if split == 'test':
            ori_data = load_data('/Users/zard/Documents/GitHub/NER_in_Chinese/data/china-people-daily-ner-corpus/example.test')
        if split == 'dev':
            ori_data = load_data('/Users/zard/Documents/GitHub/NER_in_Chinese/data/china-people-daily-ner-corpus/example.dev')
        self.my_data = []
        for i in range(len(ori_data)):
            data = ori_data[i][0]
            if len(data) > 510:
                print('数值长度超过510了，', ori_data[i])
                continue
            self.my_data.append(ori_data[i])

    def __len__(self):
        return len(self.my_data)

    def __getitem__(self, i):
        tokens = self.my_data[i][0]
        labels = self.my_data[i][1]
        return tokens, labels

def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    # names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    out = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            word_tag = []
            ner_tags = []
            for i, c in enumerate(l.split('\n')):
                word, ner_tag = c.split(' ')
                word_tag.append(word)
                if ner_tag == 'O':
                    ner_tags.append(0)
                elif ner_tag == 'B-PER':
                    ner_tags.append(1)
                elif ner_tag == 'I-PER':
                    ner_tags.append(2)
                elif ner_tag == 'B-ORG':
                    ner_tags.append(3)
                elif ner_tag == 'I-ORG':
                    ner_tags.append(4)
                elif ner_tag == 'B-LOC':
                    ner_tags.append(5)
                elif ner_tag == 'I-LOC':
                    ner_tags.append(6)

                # d[0] += char
                # if flag[0] == 'B':
                #     d.append([i, i, flag[2:]])
                # elif flag[0] == 'I':
                #     d[-1][1] = i
            out.append((word_tag, ner_tags))
    return out



dataset = Dataset('train')

tokens, labels = dataset[0]
print("tokens:", tokens, " labels:", labels)

tokens, labels = dataset[1]
print("tokens:", tokens, " labels:", labels)

# ['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间', '的', '海', '域', '。']  labels: [0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0]
# teacher:
# ['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间', '的', '海', '域', '。']  labels: [0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0]
