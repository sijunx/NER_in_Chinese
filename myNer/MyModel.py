import torch
from transformers import AutoModel

#加载预训练模型
pretrained = AutoModel.from_pretrained('hfl/rbt6')

#统计参数量
print(sum(i.numel() for i in pretrained.parameters()) / 10000)

#模型试算
#[b, lens] -> [b, lens, 768]
# pretrained(**inputs).last_hidden_state.shape

#定义下游模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tuneing = False
        self.pretrained = None

        # self.rnn = torch.nn.GRU(768, 768,batch_first=True)
        self.fc = torch.nn.Linear(768, 8)

    def forward(self, inputs):
        if self.tuneing:
            out = self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrained(**inputs).last_hidden_state

        # out, _ = self.rnn(out)

        # out = self.fc(out).softmax(dim=2)
        out = self.fc(out)
        out = out.softmax(dim=2)
        return out

    def fine_tuneing(self, tuneing):
        self.tuneing = tuneing
        if tuneing:
            for i in pretrained.parameters():
                i.requires_grad = True

            pretrained.train()
            self.pretrained = pretrained
        else:
            for i in pretrained.parameters():
                i.requires_grad_(False)

            pretrained.eval()
            self.pretrained = None




# model(inputs).shape


