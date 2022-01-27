import torch
import torch.nn.functional as F
import Levenshtein as Lev

from utils import constant

from data.helper import get_word_segments_per_language, is_contain_chinese_word
import torch.nn.functional as F


from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
from collections import Counter


def calc_f1(data):
    golden_char_total = 0.0+1e-5
    pred_char_total = 0.0+1e-5
    hit_char_total = 0.0
    for response, golden_response in data:
        # golden_response = "".join(golden_response).decode("utf8")
        # response = "".join(response).decode("utf8")
        common = Counter(response) & Counter(golden_response)
        hit_char_total += sum(common.values())
        golden_char_total += len(golden_response)
        pred_char_total += len(response)
    p = hit_char_total / pred_char_total
    r = hit_char_total / golden_char_total
    f1 = 2 * p * r / (p + r+1e-5)
    return f1

def bleu(data):
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in data:
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2


def distinct(seqs):
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    # intra_dist1 = np.average(intra_dist1)
    # intra_dist2 = np.average(intra_dist2)
    return  inter_dist1, inter_dist2

device_gpu = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")


def calculate_cer_en_zh(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence (hyp)
        s2 (string): space-separated sentence (gold)
    """
    s1_segments = get_word_segments_per_language(s1)
    s2_segments = get_word_segments_per_language(s2)

    en_s1_seq, en_s2_seq = "", ""
    zh_s1_seq, zh_s2_seq = "", ""

    for segment in s1_segments:
        if is_contain_chinese_word(segment):
            if zh_s1_seq != "":
                zh_s1_seq += " "
            zh_s1_seq += segment
        else:
            if en_s1_seq != "":
                en_s1_seq += " "
            en_s1_seq += segment
    
    for segment in s2_segments:
        if is_contain_chinese_word(segment):
            if zh_s2_seq != "":
                zh_s2_seq += " "
            zh_s2_seq += segment
        else:
            if en_s2_seq != "":
                en_s2_seq += " "
            en_s2_seq += segment

    # print(">", en_s1_seq, "||", en_s2_seq, len(en_s2_seq), "||", calculate_cer(en_s1_seq, en_s2_seq) / max(1, len(en_s2_seq.replace(' ', ''))))
    # print(">>", zh_s1_seq, "||", zh_s2_seq, len(zh_s2_seq), "||", calculate_cer(zh_s1_seq, zh_s2_seq) /  max(1, len(zh_s2_seq.replace(' ', ''))))

    return calculate_cer(en_s1_seq, en_s2_seq), calculate_cer(zh_s1_seq, zh_s2_seq), len(en_s2_seq), len(zh_s2_seq)

def calculate_cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence (hyp)
        s2 (string): space-separated sentence (gold)
    """
    return Lev.distance(s1, s2)

def calculate_wer(s1, s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))
    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]
    return Lev.distance(''.join(w1), ''.join(w2))

def calculate_metrics(pred,gold,targets2,tgt_lengths,en1,en2):
    loss = calculate_loss(pred, gold)
    # print('origanl loss  is :', loss)
    #add kl loss
    kl_loss1 = F.kl_div(F.log_softmax(en1, dim=-1), F.softmax(en2, dim=-1), reduction='none')
    kl_loss1 = kl_loss1.sum()
    kl_loss2 = F.kl_div(F.log_softmax(en2, dim=-1), F.softmax(en1, dim=-1), reduction='none')
    kl_loss2 = kl_loss2.sum()
    kl_loss = (kl_loss1 + kl_loss2) /200 #for regulazation
    # print('kl_loss is :',kl_loss)

    # log_probs = out.transpose(0, 1)  # T x B x C
    # print(gold.size())
    """
    log_probs: torch.Size([209, 8, 3793])
    targets: torch.Size([8, 46])
    input_lengths: torch.Size([8])
    target_lengths: torch.Size([8])
    """
    # log_probs = F.log_softmax(out, dim=2)
    # log_probs = log_probs.to(device_cpu)
    # targets2 = targets2.to(device_cpu)
    # out_lens = out_lens.to(device_cpu)
    # tgt_lengths = tgt_lengths.to(device_cpu)
    # loss_ctc = F.ctc_loss(log_probs, targets2, out_lens, tgt_lengths, reduction="mean")
    # loss_ctc = loss_ctc.to(device_gpu)
    # loss_ctc = loss_ctc/3 #for regulazation
    # print('loss_ctc is :', loss_ctc)



    # pred = pred.view(-1, pred.size(2)) # (B*T) x C
    # gold = gold.contiguous().view(-1) # (B*T)
    # pred = pred.max(1)[1]
    # non_pad_mask = gold.ne(constant.PAD_TOKEN)
    # num_correct = pred.eq(gold)
    # num_correct = num_correct.masked_select(non_pad_mask).sum().item()

    loss = loss + kl_loss
    return loss

def pad_loss(pred,out):

    pred = pred.transpose(0, 1)
    out = out.transpose(0, 1)
    out = [y[y != 0] for y in out]
    n_batch = pred.size(0)
    max_len = pred.size(1)
    pad = out[0].new(n_batch, max_len, *out[0].size()[1:]).fill_(0)
    # print('pad shape 0: ',pad.shape)
    for i in range(n_batch):
        pad[i, :out[i].size(0)] = out[i]
    p1 = pred.transpose(0, 1)
    p2 = pad.transpose(0, 1)
    p3 = (1 - 0.5) * p1 + 0.5 * p2
    # print('p3.shape :', p3.shape)
    return p3



def calculate_loss(pred, gold):
    # pred = pred.view(-1, pred.size(2)) # (B*T) x C
    gold = gold.contiguous().view(-1) # (B*T)
    # print('pred shape loss:',pred.shape)
    # print('out shape loss:', out.shape)
    # print('gold shape loss:', gold.shape)

    # pred = pad_loss(pred,gold)

    loss = F.cross_entropy(pred, gold, ignore_index=constant.PAD_TOKEN, reduction="mean")
    return loss