

import tqdm
from utils.functions import init_transformer_model
import torch
from models.asr.transformer import Transformer
import Levenshtein as Lev

def load_model(load_path):
    """
    Loading model
    args:
        load_path: string
    """
    checkpoint = torch.load(load_path)
    if 'args' in checkpoint:
        args = checkpoint['args']

    label2id = checkpoint['label2id']
    id2label = checkpoint['id2label']

    model = init_transformer_model(args, label2id, id2label)
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.cuda:
        model = model.cuda()
    device = torch.device("cuda:4")
    torch.cuda.set_device(device)
    model.to(device)
    return model

def cal_cer(s1, s2):
    s1, s2, = s1.replace(" ", ""), s2.replace(" ", "")
    return Lev.distance(s1, s2)
if __name__ == "__main__":


    model = Transformer.load("/home/zmw/big_space/zhangmeiwei_space/asr_res_model/multitask/thch30/transformer_model/best_model.th")
    device = torch.device("cuda:2")
    torch.cuda.set_device(device)
    # batch testing
    PAD_CHAR = "¶"
    SOS_CHAR = "§"
    EOS_CHAR = "¤"
    with open('data_thch30/dev.csv','r',encoding='utf-8') as f:
        test = f.readlines()
    cer = 0.0
    count = 0
    for line in test:
        line = line.strip().split(',')
        p = line[0]
        label = line[1]
        pred = model.predict(p)
        if pred == '':
            continue
        pred = pred.replace(PAD_CHAR,'')
        pred = pred.replace(SOS_CHAR, '')
        pred = pred.replace(EOS_CHAR, '')
        if pred == '':
            continue
        print('*'*10)
        print('path: ',p)
        print('实际值：',label)
        print('预测值：',pred)
        v1 = float(cal_cer(pred, label))
        cer += v1 / float(len(label))
        count +=1
    cer = cer / count
    print("测试集上的CER : ", cer)



    # path = './data_aishell/test/S0913/BAC009S0913W0123.wav'
    # model.predict(path)




















