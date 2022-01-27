import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import librosa
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from utils.data_loader import SpectrogramParser
import os
import numpy as np
import math
from utils.audio import load_audio
from utils import constant
from models.common_layers import MultiHeadAttention, PositionalEncoding, \
    PositionwiseFeedForward, PositionwiseFeedForwardWithConv, get_subsequent_mask, \
    get_non_pad_mask, get_attn_key_pad_mask, get_attn_pad_mask, pad_list
from utils.lstm_utils import calculate_lm_score
from time import *

def initial_model(args, label2id, id2label):
    if args.feat_extractor == 'emb_cnn':
        hidden_size = int(math.floor(
            (args.sample_rate * args.window_size) / 2) + 1)
        hidden_size = int(math.floor(hidden_size - 41) / 2 + 1)
        hidden_size = int(math.floor(hidden_size - 21) / 2 + 1)
        hidden_size *= 32
        args.dim_input = hidden_size
    elif args.feat_extractor == 'vgg_cnn':
        hidden_size = int(math.floor((args.sample_rate * args.window_size) / 2) + 1) # 161
        hidden_size = int(math.floor(int(math.floor(hidden_size)/2)/2)) * 128 # divide by 2 for maxpooling
        args.dim_input = hidden_size
    else:
        print("the model is initialized without feature extractor")

    num_layers = args.num_layers
    num_heads = args.num_heads
    dim_model = args.dim_model
    dim_key = args.dim_key
    dim_value = args.dim_value
    dim_input = args.dim_input
    dim_inner = args.dim_inner
    dim_emb = args.dim_emb
    src_max_len = args.src_max_len
    tgt_max_len = args.tgt_max_len
    dropout = args.dropout
    dropout1 = args.dropout1
    emb_trg_sharing = args.emb_trg_sharing
    feat_extractor = args.feat_extractor
    encoder = Encoder(num_layers, num_heads=num_heads, dim_model=dim_model, dim_key=dim_key,
                      dim_value=dim_value, dim_input=dim_input, dim_inner=dim_inner, src_max_length=src_max_len, dropout=dropout1)

    decoder = Decoder(id2label, num_src_vocab=len(label2id), num_trg_vocab=len(label2id), num_layers=num_layers, num_heads=num_heads,
                      dim_emb=dim_emb, dim_model=dim_model, dim_inner=dim_inner, dim_key=dim_key, dim_value=dim_value, trg_max_length=tgt_max_len, dropout=dropout, emb_trg_sharing=emb_trg_sharing)
    model = Transformer(encoder,decoder, feat_extractor=feat_extractor)
    return model

class ConvBlock(nn.Module):
    def __init__(self, conv, p):
        super().__init__()
        self.conv = conv
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv = weight_norm(self.conv)
        self.act = nn.GLU(1)
        self.dropout = nn.Dropout(p, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class MASRModel(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
    @classmethod
    def load(cls, path):
        m = torch.load(path, map_location=torch.device('cpu'))
        return m
    def predict(self, *args):
        raise NotImplementedError()
    def _default_decode(self, yp, yp_lens):
        idxs = yp.argmax(1)
        texts = []
        for idx, out_len in zip(idxs, yp_lens):
            idx = idx[:out_len]
            text = ""
            last = None
            for i in idx:
                if i.item() not in (last, self.blank):
                    text += self.vocabulary[i.item()]
                last = i
            texts.append(text)
        return texts
    def decode(self, *outputs):  # texts -> list of size B
        return self._default_decode(*outputs)
class GatedConv(MASRModel):
    def __init__(self, vocabulary, blank=0, name="masr"):
        super().__init__(vocabulary=vocabulary, name=name, blank=blank)
        self.blank = blank
        self.vocabulary = vocabulary
        self.name = name
        output_units = len(vocabulary)
        modules = []
        modules.append(ConvBlock(nn.Conv1d(512, 500, 48, 2, 97), 0.2))
        for i in range(7):
            modules.append(ConvBlock(nn.Conv1d(250, 500, 7, 1), 0.3))
        modules.append(ConvBlock(nn.Conv1d(250, 2000, 32, 1), 0.5))
        modules.append(ConvBlock(nn.Conv1d(1000, 2000, 1, 1), 0.5))
        modules.append(weight_norm(nn.Conv1d(1000, output_units, 1, 1)))
        self.cnn = nn.Sequential(*modules)
    def forward(self, x, lens):  # -> B * V * T
        x = self.cnn(x)
        return x, lens


class Transformer(nn.Module):

    def __init__(self, encoder,decoder, feat_extractor='vgg_cnn'):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.id2label = decoder.id2label
        self.feat_extractor = feat_extractor
        self.pointer = GatedConv(self.id2label)
        # self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.device = [3, 4]
        self.device_cpu = torch.device("cpu")
        # feature embedding
        if feat_extractor == 'emb_cnn':
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), ),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True)
            )
        elif feat_extractor == 'vgg_cnn':
            self.conv = nn.Sequential(
                nn.Conv2d(1, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def pad_loss2(self,pred, out):
        # pred = pred.to(self.device_cpu)
        # out = out.to(self.device_cpu)
        out = out.transpose(1, 2)
        new_o = []
        for o in out:
            targets = np.zeros((1000, 4364))
            for index, o1 in enumerate(o):
                targets[index] = np.array(o1.tolist())
            new_o.append(targets)
        new_o = np.array(new_o)
        d = self.device[0]
        new_o = torch.from_numpy(new_o).to(d)
        # new_o = torch.from_numpy(new_o)
        p1 = pred.view(-1, pred.size(2)).to(d)
        p2 = new_o.view(-1, pred.size(2))
        # print('p1: ',p1)
        # print('p2: ',p2)
        p3 = (1 - 0.5) * p1 + 0.5 * p2
        return p3

    def forward(self, padded_input, input_lengths, padded_target, verbose=False):

        if self.feat_extractor == 'emb_cnn' or self.feat_extractor == 'vgg_cnn':
            padded_input = self.conv(padded_input)
        sizes = padded_input.size() # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH

        encoder_padded_outputs1, _ = self.encoder(padded_input, input_lengths)
        encoder_padded_outputs2, _ = self.encoder(padded_input, input_lengths)
        pointer_input = encoder_padded_outputs2.transpose(1, 2).contiguous()
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs1, input_lengths)
        hyp_best_scores, hyp_best_ids = torch.topk(pred, 1, dim=2)
        hyp_seq = hyp_best_ids.squeeze(2)
        gold_seq = gold
        out,lens_out = self.pointer(pointer_input,input_lengths)
        new_pred = self.pad_loss2(pred,out)
        return new_pred,gold, hyp_seq, gold_seq,encoder_padded_outputs1,encoder_padded_outputs2

    def cal_loss(self,pred,gold, en1, en2):
        loss = self.calculate_metrics(pred, gold, en1, en2)
        return loss

    def calculate_loss(self,pred, gold):
        gold = gold.contiguous().view(-1)  # (B*T)
        loss = F.cross_entropy(pred, gold, ignore_index=constant.PAD_TOKEN, reduction="mean")
        return loss

    def calculate_metrics(self,pred, gold, en1, en2):
        loss = self.calculate_loss(pred, gold)
        kl_loss1 = F.kl_div(F.log_softmax(en1, dim=-1), F.softmax(en2, dim=-1), reduction='mean')
        kl_loss1 = kl_loss1.sum()
        kl_loss2 = F.kl_div(F.log_softmax(en2, dim=-1), F.softmax(en1, dim=-1), reduction='mean')
        kl_loss2 = kl_loss2.sum()
        kl_loss = (kl_loss1 + kl_loss2) / 200  # for regulazation
        loss = loss + kl_loss
        return loss

    def evaluate_beamsearch(self, padded_input, input_lengths, padded_target):
        padded_input = self.conv(padded_input)
        # Reshaping features
        sizes = padded_input.size()  # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH
        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        encoder_padded_outputs2, _ = self.encoder(padded_input, input_lengths)
        pointer_input = encoder_padded_outputs2.transpose(1, 2).contiguous()
        out, lens_out = self.pointer(pointer_input, input_lengths)

        hyp, gold, *_ = self.decoder(padded_target, encoder_padded_outputs, input_lengths)

        strs_gold = ["".join([self.id2label[int(x)] for x in gold_seq]) for gold_seq in gold]
        res_str = self.decoder.beam_search(encoder_padded_outputs)

        return _, res_str, strs_gold

    def evaluate_greedy(self, padded_input, input_lengths, padded_target):
        padded_input = self.conv(padded_input)
        # Reshaping features
        sizes = padded_input.size()  # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH
        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        encoder_padded_outputs2, _ = self.encoder(padded_input, input_lengths)
        pointer_input = encoder_padded_outputs2.transpose(1, 2).contiguous()
        out, lens_out = self.pointer(pointer_input, input_lengths)

        hyp, gold, *_ = self.decoder(padded_target, encoder_padded_outputs, input_lengths)

        strs_gold = ["".join([self.id2label[int(x)] for x in gold_seq]) for gold_seq in gold]
        strs_hyps = self.decoder.greedy_search(encoder_padded_outputs,out)

        return _, strs_hyps, strs_gold


class Encoder(nn.Module):
    """ 
    Encoder Transformer class
    """

    def __init__(self, num_layers, num_heads, dim_model, dim_key, dim_value, dim_input, dim_inner, dropout=0.1, src_max_length=2500):
        super(Encoder, self).__init__()

        self.dim_input = dim_input
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_inner = dim_inner

        self.src_max_length = src_max_length

        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout
        # print('dim_input  :',dim_input)
        # print('dim_model  :', dim_model)

        self.input_linear = nn.Linear(dim_input, dim_model)
        self.layer_norm_input = nn.LayerNorm(dim_model)
        self.positional_encoding = PositionalEncoding(
            dim_model, src_max_length)

        self.layers = nn.ModuleList([
            EncoderLayer(num_heads, dim_model, dim_inner, dim_key, dim_value, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, padded_input, input_lengths):

        encoder_self_attn_list = []


        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)  # B x T x D
        seq_len = padded_input.size(1)
        self_attn_mask = get_attn_pad_mask(padded_input, input_lengths, seq_len)  # B x T x T

        encoder_output = self.layer_norm_input(self.input_linear(
            padded_input)) + self.positional_encoding(padded_input)

        for layer in self.layers:
            encoder_output, self_attn = layer(
                encoder_output, non_pad_mask=non_pad_mask, self_attn_mask=self_attn_mask)
            encoder_self_attn_list += [self_attn]

        return encoder_output, encoder_self_attn_list


class EncoderLayer(nn.Module):
    """
    Encoder Layer Transformer class
    """

    def __init__(self, num_heads, dim_model, dim_inner, dim_key, dim_value, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            num_heads, dim_model, dim_key, dim_value, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardWithConv(
            dim_model, dim_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, self_attn_mask=None):
        enc_output, self_attn = self.self_attn(
            enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, self_attn


class Decoder(nn.Module):
    """
    Decoder Layer Transformer class
    """

    def __init__(self, id2label, num_src_vocab, num_trg_vocab, num_layers, num_heads,
                 dim_emb, dim_model, dim_inner, dim_key, dim_value, dropout=0.1,
                 trg_max_length=1000, emb_trg_sharing=False):
        super(Decoder, self).__init__()
        self.sos_id = constant.SOS_TOKEN
        self.eos_id = constant.EOS_TOKEN

        self.id2label = id2label

        self.num_src_vocab = num_src_vocab
        self.num_trg_vocab = num_trg_vocab
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.dim_emb = dim_emb
        self.dim_model = dim_model
        self.dim_inner = dim_inner
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.dropout_rate = dropout
        self.emb_trg_sharing = emb_trg_sharing

        self.trg_max_length = trg_max_length

        self.trg_embedding = nn.Embedding(num_trg_vocab, dim_emb, padding_idx=constant.PAD_TOKEN)
        self.positional_encoding = PositionalEncoding(
            dim_model, max_length=trg_max_length)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(dim_model, dim_inner, num_heads,
                         dim_key, dim_value, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.output_linear = nn.Linear(dim_model, num_trg_vocab, bias=False)
        nn.init.xavier_normal_(self.output_linear.weight)

        if emb_trg_sharing:
            self.output_linear.weight = self.trg_embedding.weight
            self.x_logit_scale = (dim_model ** -0.5)
        else:
            self.x_logit_scale = 1.0

    def preprocess(self, padded_input):
        """
        Add SOS TOKEN and EOS TOKEN into padded_input
        """
        seq = [y[y != constant.PAD_TOKEN] for y in padded_input]
        eos = seq[0].new([self.eos_id])
        sos = seq[0].new([self.sos_id])
        seq_in = [torch.cat([sos, y], dim=0) for y in seq]
        seq_out = [torch.cat([y, eos], dim=0) for y in seq]
        seq_in_pad = pad_list(seq_in, self.eos_id)
        seq_out_pad = pad_list(seq_out, constant.PAD_TOKEN)
        assert seq_in_pad.size() == seq_out_pad.size()
        return seq_in_pad, seq_out_pad

    def forward(self, padded_input, encoder_padded_outputs, encoder_input_lengths):

        decoder_self_attn_list, decoder_encoder_attn_list = [], []
        # print('decoder input before process:',padded_input.shape)
        seq_in_pad, seq_out_pad = self.preprocess(padded_input)
        # print('decoder input after process:', seq_in_pad.shape)

        # Prepare masks
        non_pad_mask = get_non_pad_mask(seq_in_pad, pad_idx=constant.EOS_TOKEN)
        self_attn_mask_subseq = get_subsequent_mask(seq_in_pad)
        self_attn_mask_keypad = get_attn_key_pad_mask(
            seq_k=seq_in_pad, seq_q=seq_in_pad, pad_idx=constant.EOS_TOKEN)
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0)

        output_length = seq_in_pad.size(1)
        dec_enc_attn_mask = get_attn_pad_mask(
            encoder_padded_outputs, encoder_input_lengths, output_length)

        decoder_output = self.dropout(self.trg_embedding(
            seq_in_pad) * self.x_logit_scale + self.positional_encoding(seq_in_pad))
        # print('seq_in_pad:',seq_in_pad.shape)
        # print('trg_embedding shape:',self.trg_embedding(seq_in_pad).shape)
        # print('self.positional_encoding(seq_in_pad) shape:', self.positional_encoding(seq_in_pad).shape)
        # print('decoder_output shape 0:',decoder_output.shape)

        for layer in self.layers:
            decoder_output, decoder_self_attn, decoder_enc_attn = layer(
                decoder_output, encoder_padded_outputs, non_pad_mask=non_pad_mask,
                self_attn_mask=self_attn_mask, dec_enc_attn_mask=dec_enc_attn_mask)

            decoder_self_attn_list += [decoder_self_attn]
            decoder_encoder_attn_list += [decoder_enc_attn]

        # print('decoder_output shape1',decoder_output.shape)
        seq_logit = self.output_linear(decoder_output)
        # print('seq_logit shape1', seq_logit.shape)
        #考虑在这里加入
        pred, gold = seq_logit, seq_out_pad

        return pred, gold, decoder_self_attn_list, decoder_encoder_attn_list

    def post_process_hyp(self, hyp):

        return "".join([self.id2label[int(x)] for x in hyp['yseq'][1:]])

    def greedy_search(self, encoder_padded_outputs,out):
        ys = torch.ones(encoder_padded_outputs.size(0),1).fill_(constant.SOS_TOKEN).long() # batch_size x 1
        # if constant.args.cuda:
        #     ys = ys.cuda()
        device_ids = [3, 4]
        ys = ys.cuda(device=device_ids[0])
        decoded_words = []
        for t in range(300):
            non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # batch_size x t x 1
            self_attn_mask = get_subsequent_mask(ys) # batch_size x t x t

            decoder_output = self.dropout(self.trg_embedding(ys) * self.x_logit_scale 
                                        + self.positional_encoding(ys))


            for layer in self.layers:
                decoder_output, _, _ = layer(
                    decoder_output, encoder_padded_outputs,
                    non_pad_mask=non_pad_mask,
                    self_attn_mask=self_attn_mask,
                    dec_enc_attn_mask=None
                )

            prob = self.output_linear(decoder_output) # batch_size x t x label_size
            # print('prob shape :',prob.shape)


            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append([constant.EOS_CHAR if ni.item() == constant.EOS_TOKEN else self.id2label[ni.item()] for ni in next_word.view(-1)])
            next_word = next_word.unsqueeze(-1)

            if constant.args.cuda:
                ys = torch.cat([ys, next_word.cuda()], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, next_word], dim=1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == constant.EOS_CHAR: 
                    break
                else: 
                    st += e
            sent.append(st)
        return sent

    def beam_search(self, encoder_padded_outputs, beam_width=2,
                    nbest=5,c_weight=1):

        batch_size = encoder_padded_outputs.size(0)
        max_len = encoder_padded_outputs.size(1)

        batch_ids_nbest_hyps = []
        batch_strs_nbest_hyps = []
        for x in range(batch_size):
            encoder_output = encoder_padded_outputs[x].unsqueeze(0) # 1 x T x H
            # add SOS_TOKEN
            ys = torch.ones(1, 1).fill_(constant.SOS_TOKEN).type_as(encoder_output).long()
            device_ids = [3, 4]
            ys = ys.cuda(device=device_ids[0])
            
            hyp = {'score': 0.0, 'yseq':ys}
            hyps = [hyp]
            ended_hyps = []
            for i in range(100):
                hyps_best_kept = []
                for hyp in hyps:
                    ys = hyp['yseq'] # 1 x i
                    # Prepare masks
                    non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # 1xix1
                    self_attn_mask = get_subsequent_mask(ys)
                    decoder_output = self.dropout(self.trg_embedding(ys) * self.x_logit_scale 
                                                + self.positional_encoding(ys))
                    for layer in self.layers:
                        # print(decoder_output.size(), encoder_output.size())
                        decoder_output, _, _ = layer(
                            decoder_output, encoder_output,
                            non_pad_mask=non_pad_mask,
                            self_attn_mask=self_attn_mask,
                            dec_enc_attn_mask=None
                        )
                    seq_logit = self.output_linear(decoder_output[:, -1])
                    local_scores = F.log_softmax(seq_logit, dim=1)
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam_width, dim=1)
                    # calculate beam scores
                    for j in range(beam_width):
                        new_hyp = {}
                        new_hyp["score"] = hyp["score"] + local_best_scores[0, j]

                        new_hyp["yseq"] = torch.ones(1, (1+ys.size(1))).type_as(encoder_output).long()
                        new_hyp["yseq"][:, :ys.size(1)] = hyp["yseq"].cpu()
                        new_hyp["yseq"][:, ys.size(1)] = int(local_best_ids[0, j]) # adding new word
                        
                        hyps_best_kept.append(new_hyp)

                    hyps_best_kept = sorted(hyps_best_kept, key=lambda x:x["score"], reverse=True)[:beam_width]
                
                hyps = hyps_best_kept

                # add EOS_TOKEN
                if i == max_len - 1:
                    for hyp in hyps:
                        hyp["yseq"] = torch.cat([hyp["yseq"], torch.ones(1,1).fill_(constant.EOS_TOKEN).type_as(encoder_output).long()], dim=1)

                # add hypothesis that have EOS_TOKEN to ended_hyps list
                unended_hyps = []
                for hyp in hyps:
                    if hyp["yseq"][0, -1] == constant.EOS_TOKEN:
                        seq_str = "".join(self.id2label[char.item()] for char in hyp["yseq"][0]).replace(
                            constant.PAD_CHAR, "").replace(constant.SOS_CHAR, "").replace(constant.EOS_CHAR, "")
                        seq_str = seq_str.replace("  ", " ")
                        num_words = len(seq_str.split())
                        hyp["final_score"] = hyp["score"] + math.sqrt(num_words) * c_weight
                        ended_hyps.append(hyp)
                    else:
                        unended_hyps.append(hyp)
                hyps = unended_hyps

                if len(hyps) == 0:
                    # decoding process is finished
                    break
                
            num_nbest = min(len(ended_hyps), nbest)
            nbest_hyps = sorted(ended_hyps, key=lambda x:x["final_score"], reverse=True)[:num_nbest]
            a_nbest_hyps = sorted(ended_hyps, key=lambda x:x["final_score"], reverse=True)[:beam_width]
            for hyp in nbest_hyps:                
                hyp["yseq"] = hyp["yseq"][0].cpu().numpy().tolist()
                hyp_strs = self.post_process_hyp(hyp)
                batch_ids_nbest_hyps.append(hyp["yseq"])
                batch_strs_nbest_hyps.append(hyp_strs)
                # print(hyp["yseq"], hyp_strs)
            res_str = []
            for i in range(len(batch_strs_nbest_hyps)//2):
                res_str.append(batch_strs_nbest_hyps[2*i])

        return res_str #batch_ids_nbest_hyps, batch_strs_nbest_hyps

class DecoderLayer(nn.Module):
    """
    Decoder Transformer class
    """

    def __init__(self, dim_model, dim_inner, num_heads, dim_key, dim_value, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            num_heads, dim_model, dim_key, dim_value, dropout=dropout)
        self.encoder_attn = MultiHeadAttention(
            num_heads, dim_model, dim_key, dim_value, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardWithConv(
            dim_model, dim_inner, dropout=dropout)

    def forward(self, decoder_input, encoder_output, non_pad_mask=None, self_attn_mask=None, dec_enc_attn_mask=None):
        decoder_output, decoder_self_attn = self.self_attn(
            decoder_input, decoder_input, decoder_input, mask=self_attn_mask)
        decoder_output *= non_pad_mask

        decoder_output, decoder_encoder_attn = self.encoder_attn(
            decoder_output, encoder_output, encoder_output, mask=dec_enc_attn_mask)
        decoder_output *= non_pad_mask

        decoder_output = self.pos_ffn(decoder_output)
        decoder_output *= non_pad_mask

        return decoder_output, decoder_self_attn, decoder_encoder_attn        