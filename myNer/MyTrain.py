import torch

from MyCollateFn import collate_fn
from MyDataset import Dataset
from transformers import AdamW

from MyTokenizer import tokenizer
from myNer.MyModel import Model

dataset = Dataset('train')

tokens, labels = dataset[0]

print("len(dataset):", len(dataset))
print("tokens:", tokens)
print("labels:", labels)

# 数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,
                                     # batch_size=1,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

# 查看数据样例
for i, (inputs, labels) in enumerate(loader):
    break

print(len(loader))
print(tokenizer.decode(inputs['input_ids'][0]))
print(labels[0])

for k, v in inputs.items():
    print(k, v.shape)


# 对计算结果和label变形,并且移除pad
def reshape_and_remove_pad(outs, labels, attention_mask):
    # 变形,便于计算loss
    # todo:这里的8是什么含义？没懂！！！
    # [b, lens, 8] -> [b*lens, 8]
    # print("outs.shape:", outs.shape)
    outs = outs.reshape(-1, 8)
    # print("outs[0]:", outs[0])

    # [b, lens] -> [b*lens]
    # print("labels.shape:", labels.shape)
    labels = labels.reshape(-1)
    # print("labels:", labels)

    # 忽略对pad的计算结果
    # [b, lens] -> [b*lens - pad]
    select = attention_mask.reshape(-1) == 1
    outs = outs[select]
    labels = labels[select]
    # print("outs.shape:", outs.shape)
    # print("labels.shape:", labels.shape)
    return outs, labels


# reshape_and_remove_pad(torch.randn(2, 3, 7), torch.ones(2, 3), torch.ones(2, 3))

# 获取正确数量和总数
def get_correct_and_total_count(labels, outs):
    # [b*lens, 8] -> [b*lens]
    # print("outs[0]:", outs[0])
    outs = outs.argmax(dim=1)
    correct = (outs == labels).sum().item()
    total = len(labels)

    # 计算除了0以外元素的正确率,因为0太多了,包括的话,正确率很容易虚高
    select = labels != 0
    outs = outs[select]
    labels = labels[select]
    correct_content = (outs == labels).sum().item()
    total_content = len(labels)

    return correct, total, correct_content, total_content


# get_correct_and_total_count(torch.ones(16), torch.randn(16, 7))
model = Model()
# model = torch.load('/Users/zard/Documents/GitHub/NER_in_Chinese/model/命名实体识别_中文_my001.model')

# 训练
def train(epochs):
    # lr = 2e-5 if model.tuneing else 5e-4
    lr = 1e-5 if model.tuneing else 1e-4

    # 训练
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for step, (inputs, labels) in enumerate(loader):
            # print("step:", step)
            # print("inputs.len:", len(inputs))
            # print(inputs['input_ids'][0])
            # print("tokenizer.decode(inputs['input_ids'][0]):", tokenizer.decode(inputs['input_ids'][0]))
            # print("inputs['input_ids'][0].len:", len(inputs['input_ids'][0]))
            # print("inputs['input_ids'].shape:", inputs['input_ids'].shape)
            # print("labels[0].len:", len(labels[0]))

            # 模型计算
            # [b, lens] -> [b, lens, 8]
            outs = model(inputs)

            # 对outs和label变形,并且移除pad
            # outs -> [b, lens, 8] -> [c, 8]
            # labels -> [b, lens] -> [c]
            outs, labels = reshape_and_remove_pad(outs, labels,
                                                  inputs['attention_mask'])

            # 梯度下降
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 50 == 0:
                counts = get_correct_and_total_count(labels, outs)

                accuracy = 0
                if counts[1] != 0:
                    accuracy = counts[0] / counts[1]

                accuracy_content = 0
                if counts[3] != 0:
                    accuracy_content = counts[2] / counts[3]

                print(epoch, step, loss.item(), accuracy, accuracy_content)

        torch.save(model, '../model/命名实体识别_中文_my_001.model')


# model.fine_tuneing(False)
# print(sum(p.numel() for p in model.parameters()) / 10000)
# train(10)

model.fine_tuneing(True)
print(sum(p.numel() for p in model.parameters()) / 10000)
train(20)
