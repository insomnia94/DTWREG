from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from layers.lan_enc import RNNEncoder
from layers.lan_dec import RNNDncoder
from layers.vis_enc import LocationEncoder, SubjectEncoder, RelationEncoder

class SimAttention(nn.Module):
    def __init__(self, vis_dim, jemb_dim):
        super(SimAttention, self).__init__()
        self.embed_dim = 300
        self.feat_fuse = nn.Sequential(nn.Linear(self.embed_dim+vis_dim, jemb_dim),
                                       nn.ReLU(),
                                       nn.Linear(jemb_dim, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, word_emb, vis_feats):
        sent_num, ann_num  = vis_feats.size(0), vis_feats.size(1)
        word_emb = word_emb.unsqueeze(1).expand(sent_num, ann_num, self.embed_dim)
        sim_attn = self.feat_fuse(torch.cat([word_emb, vis_feats], 2))
        sim_attn = sim_attn.squeeze(2)
        return sim_attn


class AttributeReconstructLoss(nn.Module):
    def __init__(self, opt):
        super(AttributeReconstructLoss, self).__init__()
        self.att_dropout = nn.Dropout(opt['visual_drop_out'])
        self.att_fc = nn.Linear(opt['fc7_dim']+opt['pool5_dim'], opt['num_atts'])

    def forward(self, attribute_feats, total_ann_score, att_labels, select_ixs, att_weights):
        """attribute_feats.shape = (sent_num, ann_num, 512), total_ann_score.shape = (sent_num, ann_num)"""
        total_ann_score = total_ann_score.unsqueeze(1)
        att_feats_fuse = torch.bmm(total_ann_score, attribute_feats)
        att_feats_fuse = att_feats_fuse.squeeze(1)
        att_feats_fuse = self.att_dropout(att_feats_fuse)
        att_scores = self.att_fc(att_feats_fuse)
        if len(select_ixs) == 0:
            att_loss = 0
        else:
            att_loss = nn.BCEWithLogitsLoss(att_weights.cuda())(att_scores.index_select(0, select_ixs),
                                                     att_labels.index_select(0, select_ixs))
        return att_scores, att_loss


class KPRN(nn.Module):
    def __init__(self, opt):
        super(KPRN, self).__init__()
        self.num_layers = opt['rnn_num_layers']
        self.hidden_size = opt['rnn_hidden_size']
        self.num_dirs = 2 if opt['bidirectional'] > 0 else 1
        self.jemb_dim = opt['jemb_dim']
        self.word_vec_size = opt['word_vec_size']
        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']
        self.sub_filter_type = opt['sub_filter_type']
        self.filter_thr = opt['sub_filter_thr']

        # language rnn encoder
        self.rnn_encoder = RNNEncoder(vocab_size=opt['vocab_size'],
                                      word_embedding_size=opt['word_embedding_size'],
                                      word_vec_size=opt['word_vec_size'],
                                      hidden_size=opt['rnn_hidden_size'],
                                      bidirectional=opt['bidirectional'] > 0,
                                      input_dropout_p=opt['word_drop_out'],
                                      dropout_p=opt['rnn_drop_out'],
                                      n_layers=opt['rnn_num_layers'],
                                      rnn_type=opt['rnn_type'],
                                      variable_lengths=opt['variable_lengths'] > 0)

        self.mlp = nn.Sequential(nn.Linear((self.hidden_size * self.num_dirs + self.jemb_dim * 2+self.fc7_dim+self.pool5_dim),
                                           self.jemb_dim),
                                 nn.ReLU())

        self.rnn_decoder = RNNDncoder(opt)

        self.sub_encoder = SubjectEncoder(opt)
        self.loc_encoder = LocationEncoder(opt)
        self.rel_encoder = RelationEncoder(opt)

        self.sub_sim_attn = SimAttention(self.pool5_dim + self.fc7_dim, self.jemb_dim)
        self.obj_sim_attn = SimAttention(self.fc7_dim, self.jemb_dim)

        self.attn = nn.Sequential(nn.Linear(self.jemb_dim, 1))

        self.fc = nn.Linear(self.jemb_dim * 2+self.fc7_dim+self.pool5_dim, self.word_vec_size)

        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.softmax = torch.nn.Softmax(dim=1)

        self.att_res_weight = opt['att_res_weight']
        self.att_res_loss = AttributeReconstructLoss(opt)

        self.mse_loss = nn.MSELoss()


    def forward(self, pool5, fc7, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist, labels, enc_labels, dec_labels,
                sub_sim, obj_sim, sub_emb, obj_emb, att_labels, select_ixs, att_weights):

        sent_num = pool5.size(0)
        ann_num =  pool5.size(1)
        label_mask = (dec_labels != 0).float()

        # language feature encoding
        _, hidden, _ = self.rnn_encoder(labels)

        hidden = nn.functional.normalize(hidden, p=2, dim=1)
        hidden = hidden.unsqueeze(1).expand(sent_num, ann_num,
                                            self.hidden_size * self.num_dirs)
        # visual feature encoding
        sub_feats = self.sub_encoder(pool5, fc7)
        loc_feats = self.loc_encoder(lfeats, dif_lfeats)

        sub_loss = 0
        sub_idx = torch.ones([1, ann_num], dtype = torch.int64)

        cxt_fc7_att = cxt_fc7.unsqueeze(0).expand(sent_num, ann_num, self.fc7_dim)
        cxt_fc7_att = nn.functional.normalize(cxt_fc7_att, p=2, dim=2)

        obj_attn = self.obj_sim_attn(obj_emb, cxt_fc7_att)
        wo_obj_idx = obj_sim.sum(1).eq(0)
        obj_attn[wo_obj_idx] = 0

        obj_loss = self.mse_loss(obj_attn, obj_sim)

        rel_feats, dist, max_rel_id = self.rel_encoder(cxt_fc7, cxt_lfeats, obj_attn, wo_obj_idx, dist)
        dist = 100 / (dist + 100)

        vis_feats = torch.cat([sub_feats, loc_feats, rel_feats], 2)

        feat_fuse = torch.cat([hidden, vis_feats], 2)
        feat_fuse = feat_fuse.view(sent_num * ann_num, -1)
        feat_fuse = self.mlp(feat_fuse)

        if self.sub_filter_type == 'thr':
            sub_attn = self.sub_sim_attn(sub_emb, sub_feats)
            sub_loss = self.mse_loss(sub_attn, sub_sim)

            sub_idx = sub_sim.gt(self.filter_thr)

            all_filterd_idx = (sub_idx.sum(1).eq(0))

            sub_idx[all_filterd_idx] = 1
            feat_fuse = feat_fuse.view(sent_num, ann_num, -1)
            sub_filtered_idx = sub_idx.eq(0)
            feat_fuse[sub_filtered_idx] = 0
            feat_fuse = feat_fuse.view(sent_num*ann_num, -1)

            att_score = self.attn(feat_fuse).squeeze(1).view(sent_num, ann_num)
            att_score = att_score

        elif self.sub_filter_type == 'soft':
            sub_attn = self.sub_sim_attn(sub_emb, sub_feats)
            sub_loss = self.mse_loss(sub_attn, sub_sim)

            sub_idx = sub_sim.gt(self.filter_thr)

            all_filterd_idx = (sub_idx.sum(1).eq(0))

            sub_idx[all_filterd_idx] = 1
            att_score = self.attn(feat_fuse).squeeze(1).view(sent_num, ann_num)
            att_score = att_score * sub_attn
        else:
            att_score = self.attn(feat_fuse).squeeze(1).view(sent_num, ann_num)


        att_score = att_score * dist

        att_score = self.softmax(att_score)

        att_score_prob = att_score.unsqueeze(1)
        vis_att_fuse = torch.bmm(att_score_prob, vis_feats)
        vis_att_fuse = vis_att_fuse.squeeze(1)
        vis_att_fuse = self.fc(vis_att_fuse)


        dec_outs = self.rnn_decoder(vis_att_fuse, enc_labels)

        dec_labels = dec_labels.view(-1)
        label_mask = label_mask.view(-1)
        rec_loss = self.cross_entropy(dec_outs, dec_labels)
        rec_loss = torch.sum(rec_loss * label_mask) / torch.sum(label_mask)
        rec_loss = rec_loss + sub_loss + obj_loss

        if self.att_res_weight > 0:
            att_scores, att_res_loss = self.att_res_loss(sub_feats, att_score, att_labels, select_ixs, att_weights)
            rec_loss += self.att_res_weight * att_res_loss

        return att_score, rec_loss, sub_idx, max_rel_id
