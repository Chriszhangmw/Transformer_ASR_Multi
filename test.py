import json
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm
# from models.asr.transformer import Transformer, Encoder, Decoder
from utils import constant
from utils.data_loader_test import SpectrogramDataset, AudioDataLoader, BucketingSampler
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer, calculate_cer_en_zh,calc_f1,bleu,distinct
from utils.functions_test import  load_model
from utils.lstm_utils import LM
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

def evaluate(model, test_loader):
    """
    Evaluation
    args:
        model: Model object
        test_loader: DataLoader object
    """
    model.eval()
    F1data = []
    predictions = []
    total_word, total_char, total_cer, total_wer = 0, 0, 0, 0
    total_en_cer, total_zh_cer, total_en_char, total_zh_char = 0, 0, 0, 0
    F1data2 = []
    predictions2 = []
    total_word2, total_char2, total_cer2, total_wer2 = 0, 0, 0, 0
    total_en_cer2, total_zh_cer2, total_en_char2, total_zh_char2 = 0, 0, 0, 0
    # device = torch.device("cpu")
    with torch.no_grad():
        test_pbar = tqdm(iter(test_loader), leave=True, total=len(test_loader))
        for i, (data) in enumerate(test_pbar):
            src, tgt, src_percentages, src_lengths, tgt_lengths,targets2 = data

            # batch_ids_nbest_hyps, batch_strs_nbest_hyps
            batch_ids_hyps,batch_strs_hyps, batch_strs_gold = model.evaluate_greedy( src,
                                                                               src_lengths,
                                                                               tgt)
            # batch_ids_hyps, batch_strs_hyps2, batch_strs_gold = model.evaluate_beamsearch(src,
            #                                                                          src_lengths,
            #                                                                          tgt)


            for x in range(len(batch_strs_gold)):
                # print(batch_strs_hyps[x])
                hyp = batch_strs_hyps[x].replace(constant.EOS_CHAR, "").replace(constant.SOS_CHAR, "").replace(constant.PAD_CHAR, "")
                # hyp2 = batch_strs_hyps2[x].replace(constant.EOS_CHAR, "").replace(constant.SOS_CHAR, "").replace(
                #     constant.PAD_CHAR, "")
                gold = batch_strs_gold[x].replace(constant.EOS_CHAR, "").replace(constant.SOS_CHAR, "").replace(constant.PAD_CHAR, "")

                wer = calculate_wer(hyp, gold)
                # wer2 = calculate_wer(hyp2, gold)
                cer = calculate_cer(hyp.strip(), gold.strip())
                # cer2 = calculate_cer(hyp2.strip(), gold.strip())
                F1data.append((hyp.strip(), gold.strip()))
                predictions.append(hyp.strip())
                en_cer, zh_cer, num_en_char, num_zh_char = calculate_cer_en_zh(hyp, gold)
                # F1data2.append((hyp2.strip(), gold.strip()))
                # predictions2.append(hyp2.strip())
                # en_cer2, zh_cer2, num_en_char2, num_zh_char2 = calculate_cer_en_zh(hyp2, gold)
                total_en_cer += en_cer
                total_zh_cer += zh_cer
                total_en_char += num_en_char
                total_zh_char += num_zh_char

                # total_en_cer2 += en_cer2
                # total_zh_cer2 += zh_cer2
                # total_en_char2 += num_en_char2
                # total_zh_char2 += num_zh_char2

                total_wer += wer
                total_cer += cer
                total_word += len(gold.split(" "))
                total_char += len(gold)

                # total_wer2 += wer2
                # total_cer2 += cer2
                # total_word2 += len(gold.split(" "))
                # total_char2 += len(gold)

            test_pbar.set_description("Greedy TEST CER:{:.2f}% WER:{:.2f}% CER_EN:{:.2f}% CER_ZH:{:.2f}%".format(
                total_cer*100/total_char, total_wer*100/total_word, total_en_cer*100/max(1, total_en_char),
                total_zh_cer*100/max(1, total_zh_char)))
        f1 = calc_f1(F1data)
        bleu_1, bleu_2 = bleu(F1data)
        unigrams_distinct, bigrams_distinct = distinct(predictions)
        # f12 = calc_f1(F1data2)
        # bleu_12, bleu_22 = bleu(F1data2)
        # unigrams_distinct2, bigrams_distinct2 = distinct(predictions2)
        print("Greedy TEST CER:{:.2f}% WER:{:.2f}% f1:{:.2f}% bleu_1:{:.2f}% bleu_2:{:.2f}% unigrams_distinct:{:.2f}% bigrams_distinct:{:.2f}%".format(
                total_cer * 100 / total_char, total_wer * 100 / total_word, f1, bleu_1, bleu_2, unigrams_distinct,
                bigrams_distinct))
        return total_cer * 100 / total_char
        # print("Beam TEST CER:{:.2f}% WER:{:.2f}% f1:{:.2f}% bleu_1:{:.2f}% bleu_2:{:.2f}% unigrams_distinct:{:.2f}% bigrams_distinct:{:.2f}%".format(
        #         total_cer2 * 100 / total_char2, total_wer2 * 100 / total_word2, f12, bleu_12, bleu_22, unigrams_distinct2,
        #         bigrams_distinct2))

def jj():
    print('JJ')
def run_test_during_train():
    args = constant.args
    p = "/home/zmw/big_space/zhangmeiwei_space/asr_res_model/multitask/aishell/transformer_model/6_8/best_model.th"
    model, opt, epoch, metrics, loaded_args, label2id, id2label = load_model(p)
    audio_conf = dict(sample_rate=loaded_args.sample_rate,
                      window_size=loaded_args.window_size,
                      window_stride=loaded_args.window_stride,
                      window=loaded_args.window,
                      noise_dir=loaded_args.noise_dir,
                      noise_prob=loaded_args.noise_prob,
                      noise_levels=(loaded_args.noise_min, loaded_args.noise_max))
    test_data = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath_list=constant.args.test_manifest_list,
                                   label2id=label2id,
                                   normalize=True, augment=False)
    test_sampler = BucketingSampler(test_data, batch_size=constant.args.batch_size)
    test_loader = AudioDataLoader(test_data, num_workers=args.num_workers, batch_sampler=test_sampler)
    cer = evaluate(model, test_loader)
    return cer




if __name__ == '__main__':
    args = constant.args
    start_iter = 0
    # Load the model
    load_path = constant.args.continue_from
    p = "/home/zmw/big_space/zhangmeiwei_space/asr_res_model/multitask/aishell/transformer_model/6_8/best_model.th"
    model, opt, epoch, metrics, loaded_args, label2id, id2label = load_model(p)

    audio_conf = dict(sample_rate=loaded_args.sample_rate,
                      window_size=loaded_args.window_size,
                      window_stride=loaded_args.window_stride,
                      window=loaded_args.window,
                      noise_dir=loaded_args.noise_dir,
                      noise_prob=loaded_args.noise_prob,
                      noise_levels=(loaded_args.noise_min, loaded_args.noise_max))

    test_data = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath_list=constant.args.test_manifest_list, label2id=label2id,
                                   normalize=True, augment=False)
    test_sampler = BucketingSampler(test_data, batch_size=constant.args.batch_size)
    test_loader = AudioDataLoader(test_data, num_workers=args.num_workers, batch_sampler=test_sampler)

    lm = None
    if constant.args.lm_rescoring:
        lm = LM(constant.args.lm_path)

    # print(model)

    evaluate(model, test_loader, lm=lm)
