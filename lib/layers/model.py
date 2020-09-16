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
from layers.vis_enc import LocationEncoder, SubjectEncoder, RelationEncoder, PairEncoder

class SimAttention(nn.Module):
    def __init__(self, vis_dim, words_dim, jemb_dim):
        super(SimAttention, self).__init__()
        self.embed_dim = 300
        self.words_dim = words_dim
        self.feat_fuse = nn.Sequential(nn.Linear(words_dim + vis_dim, jemb_dim),
                                       nn.ReLU(),
                                       nn.Linear(jemb_dim, jemb_dim),
                                       nn.ReLU(),
                                       nn.Linear(jemb_dim, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, word_emb, vis_feats):
        sent_num, ann_num  = vis_feats.size(0), vis_feats.size(1)
        word_emb = word_emb.unsqueeze(1).expand(sent_num, ann_num, self.words_dim)
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
        self.word_emb_size = opt['word_emb_size']

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
        ''''
        self.mlp = nn.Sequential(
            nn.Linear((self.hidden_size * self.num_dirs + self.jemb_dim * 2+self.fc7_dim+self.pool5_dim), self.jemb_dim),
            nn.ReLU())
        '''

        #self.rnn_decoder = RNNDncoder(opt)
        #self.sub_encoder = SubjectEncoder(opt)
        #self.loc_encoder = LocationEncoder(opt)
        #self.rel_encoder = RelationEncoder(opt)
        #self.sub_sim_attn = SimAttention(self.pool5_dim + self.fc7_dim, self.word_emb_size, self.jemb_dim)
        #self.obj_sim_attn = SimAttention(self.fc7_dim, self.word_emb_size, self.jemb_dim)
        #self.attn = nn.Sequential(nn.Linear(self.jemb_dim, 1))
        #self.fc = nn.Linear(self.jemb_dim * 2+self.fc7_dim+self.pool5_dim, self.word_vec_size)
        #self.att_res_weight = opt['att_res_weight']
        #self.att_res_loss = AttributeReconstructLoss(opt)

        # new objects

        self.visual_noun = nn.Sequential(
            nn.Linear(self.fc7_dim, 1024),
            #nn.Linear(self.pool5_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            #nn.Linear(512, opt['noun_candidate_size']),
            nn.Linear(512, opt['class_size']),
            #nn.ReLU(),
        )

        self.visual_emb = nn.Sequential(
            nn.Linear(self.fc7_dim, 1024),
            #nn.Linear(self.pool5_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, opt['word_emb_size']),
            #nn.ReLU(),
        )

        self.pair_prep = nn.Sequential(
            nn.Linear(opt['pair_feat_size'], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, opt['prep_candidate_size']),
            #nn.ReLU(),
        )

        self.pair_emb = nn.Sequential(
            nn.Linear(opt['pair_feat_size'], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, opt['word_emb_size']),
            #nn.ReLU(),
        )

        self.pair_encoder = PairEncoder(opt)

        #self.pair_attn = SimAttention(opt['pair_feat_size'], self.word_emb_size*4, self.jemb_dim)
        self.pair_attn = SimAttention(opt['pair_feat_size'], self.word_emb_size*3, self.jemb_dim)

        #self.sub_attn = SimAttention(self.fc7_dim, self.word_emb_size*2, self.jemb_dim)
        self.sub_attn = SimAttention(self.fc7_dim, self.word_emb_size, self.jemb_dim)

        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.mse_loss = nn.MSELoss()


    def forward(self, pool5, fc7, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist, labels, enc_labels, dec_labels, att_labels, select_ixs, att_weights,
                sub_wordembs, sub_classembs, obj_wordembs, rel_wordembs,
                ann_pool5, ann_fc7, ann_fleats):

        sent_num = pool5.size(0)
        ann_num =  pool5.size(1)


        sub_fuseembs = 0.1*sub_wordembs + 0.9*sub_classembs

        #################################################################
        ### attn generate ###############################################
        #################################################################

        # pair attention
        ################
        #pair_wordembs = torch.cat([sub_classembs, sub_wordembs, obj_wordembs, rel_wordembs], 1)
        #pair_wordembs = torch.cat([sub_classembs, obj_wordembs, rel_wordembs], 1)
        #pair_wordembs = torch.cat([sub_wordembs, obj_wordembs, rel_wordembs], 1)
        pair_wordembs = torch.cat([sub_fuseembs, obj_wordembs, rel_wordembs], 1)
        pair_feats, expand_1_pool5, expand_1_fc7, expand_1_fleats, expand_0_pool5, expand_0_fc7, expand_0_fleats = self.pair_encoder(pool5, fc7, ann_pool5, ann_fc7, ann_fleats)
        pair_attn = self.pair_attn(pair_wordembs, pair_feats)
        pair_attn_gumbel = F.gumbel_softmax(pair_attn, tau=1, hard=True)

        # sub attention
        ###############
        #sub_attn = self.sub_attn(torch.cat([sub_classembs, sub_wordembs], 1), expand_1_fc7)
        #sub_attn = self.sub_attn(sub_wordembs, expand_1_fc7)
        #sub_attn = self.sub_attn(sub_classembs, expand_1_fc7)
        sub_attn = self.sub_attn(sub_fuseembs, expand_1_fc7)
        sub_attn_gumbel = F.gumbel_softmax(sub_attn, tau=1, hard=True)

        # obj attention
        ###############
        obj_attn = self.sub_attn(obj_wordembs, expand_0_fc7)
        obj_attn_gumbel = F.gumbel_softmax(obj_attn, tau=1, hard=True)













        #################################################################
        ### feat * attn #################################################
        #################################################################

        # pair_feat * attn
        ####################
        re_pair_feats = torch.matmul(pair_attn.view(sent_num, 1, ann_num*ann_num), pair_feats)
        re_pair_feats = re_pair_feats.reshape([sent_num, -1])

        # sub_feat * attn
        ###################
        #re_sub_feats = torch.matmul(pair_attn.view(sent_num, 1, ann_num*ann_num), torch.cat([expand_1_pool5,expand_1_fc7], 2))
        #re_sub_feats = torch.matmul(sub_attn.view(sent_num, 1, ann_num*ann_num), torch.cat([expand_1_pool5,expand_1_fc7], 2))
        #re_sub_feats = torch.matmul(pair_attn.view(sent_num, 1, ann_num*ann_num), expand_1_fc7)
        re_sub_feats = torch.matmul(sub_attn.view(sent_num, 1, ann_num*ann_num), expand_1_fc7)
        re_sub_feats = re_sub_feats.reshape([sent_num, -1])

        # obj_feat * attn
        ###################
        #re_obj_feats = torch.matmul(pair_attn.view(sent_num, 1, ann_num*ann_num), torch.cat([expand_0_pool5,expand_0_fc7], 2))
        #re_obj_feats = torch.matmul(obj_attn.view(sent_num, 1, ann_num*ann_num), torch.cat([expand_0_pool5,expand_0_fc7], 2))
        #re_obj_feats = torch.matmul(pair_attn.view(sent_num, 1, ann_num * ann_num), expand_0_fc7)
        re_obj_feats = torch.matmul(obj_attn.view(sent_num, 1, ann_num * ann_num), expand_0_fc7)
        re_obj_feats = re_obj_feats.reshape([sent_num, -1])









        #################################################################
        ### re-construct ################################################
        #################################################################

        # sub re-construct
        ##################
        sub_result = self.visual_emb(re_sub_feats)

        # obj re-construct
        ###################
        obj_result = self.visual_emb(re_obj_feats)


        # rel re-construct
        ###################
        rel_result = self.pair_emb(re_pair_feats)






        #################################################################
        ### loss ########################################################
        #################################################################

        # sub loss
        ###########
        #sub_loss = self.mse_loss(sub_result, sub_wordembs)
        sub_loss = self.mse_loss(sub_result, sub_classembs)
        sub_loss_sum = torch.sum(sub_loss)

        # obj_loss
        ##########
        obj_loss = self.mse_loss(obj_result, obj_wordembs)
        obj_loss_sum = torch.sum(obj_loss)

        # rel loss
        ###########
        rel_loss = self.mse_loss(rel_result, rel_wordembs)
        rel_loss_sum = torch.sum(rel_loss)

        # loss sum
        ###########
        loss_sum = 1*sub_loss_sum + 1*obj_loss_sum + 1*rel_loss_sum

        final_attn = 2*sub_attn + 1*obj_attn + 1*pair_attn

        return final_attn, loss_sum, sub_loss_sum, obj_loss_sum, rel_loss_sum
