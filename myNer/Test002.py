import torch
from MyTokenizer import tokenizer

def predict():
    # model_load = torch.load('model/命名实体识别_中文.model')
    model_load = torch.load('/Users/zard/Documents/GitHub/NER_in_Chinese/model/命名实体识别_中文02.model')
    # model_load.eval()
    print("模型加载----------")
    # tokens = [['测', '试', '数', '据'], ['测', '试', '苹', '果', '价', '格', '，', '虫', '子']]

    s1 = '问题是商家选择的必选品是雪天路滑加的一分钱，那现在也不下雪了。平台对商家也没有要求吗请问未点必选品这个授权怎么清除又没有完全减去的必选品的价格人家店里的必选品都是0元的，选择辣还是不辣，选口味的。这个必选品竟然是强制性的那个必选品算强制消费吗你好这家很特别必须要选择99的一个东西叫做必选品才可以送餐按饿了吗这么多年的经验就是下单五星好评，送南瓜饼，结果收到并没有。也就是说必选品五星好评必选才能下单？我确实有一张20-12的券，但是我完全没有必要白白给2元去买必选品啊，我购买其他商品也可以凑够20呀他都没有点必选品，他选一个不能下单的自己没有把餐盒加到必选品里面不是吗？'
    s2 = '拉特纳亚克议长还向江泽民主席介绍了斯里兰卡议会和国内的情况，并转达了库马拉通加总统对他的亲切问候'
    s3 = '他说，中国共产党领导的多党合作和政治协商制度，是中国的基本政治制度，中国人民政治协商会议则是实现这个制度的组织形式。'
    s4 = '饿了么外卖平台和美团外卖平台相比，怎么样？其实，我更喜欢京东'

    sents = [s1, s2, s3, s4]
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                         truncation=True,
                                         padding=True,
                                         return_tensors='pt',
                                         is_split_into_words=False)

    # inputs = tokenizer.batch_encode_plus(tokens,
    #                                      truncation=True,
    #                                      padding=True,
    #                                      return_tensors='pt',
    #                                      is_split_into_words=True)

    # tokenizer.decode(inputs['input_ids'][0])

    with torch.no_grad():
        # [b, lens] -> [b, lens, 8] -> [b, lens]
        temp_result = model_load(inputs)
        outs = temp_result.argmax(dim=2)

    print(outs)
    print("outs.shape[0]:", outs.shape[0])
    print("type(outs.shape[0]):", type(outs.shape[0]))

    for i in range(outs.shape[0]):
        index = outs[i]
        print("index:", index)
        input_id = inputs['input_ids'][i]

        s = ''
        for j in range(index.shape[0]):
            if index[j] == 0:
                s += '.'
                continue
            s += tokenizer.decode(input_id[j])
        print("s:", s)

        # #输出tag
        # for tag in [label, out]:
        #     s = ''
        #     for j in range(len(tag)):
        #         if tag[j] == 0:
        #             s += '·'
        #             continue
        #         s += tokenizer.decode(input_id[j])
        #         s += str(tag[j].item())
        #
        #     print(s)

    # for i in range(32):
    #     #移除pad
    #     select = inputs['attention_mask'][i] == 1
    #     input_id = inputs['input_ids'][i, select]
    #     out = outs[i, select]
    #     label = labels[i, select]
    #
    #     #输出原句子
    #     print(tokenizer.decode(input_id).replace(' ', ''))
    #
    #     #输出tag
    #     for tag in [label, out]:
    #         s = ''
    #         for j in range(len(tag)):
    #             if tag[j] == 0:
    #                 s += '·'
    #                 continue
    #             s += tokenizer.decode(input_id[j])
    #             s += str(tag[j].item())
    #
    #         print(s)
    #     print('==========================')


predict()
