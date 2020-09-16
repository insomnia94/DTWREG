from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json
import h5py
import time
from pprint import pprint

import torch
import torch.nn.functional as F
from torch.autograd import Variable



# IoU function
def computeIoU(box1, box2):
  # each box is of [x1, y1, w, h]
  inter_x1 = max(box1[0], box2[0])
  inter_y1 = max(box1[1], box2[1])
  inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
  inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

  if inter_x1 < inter_x2 and inter_y1 < inter_y2:
    inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
  else:
    inter = 0
  union = box1[2]*box1[3] + box2[2]*box2[3] - inter
  return float(inter)/union


def eval_split(loader, model, split, opt):

    verbose = opt.get('verbose', True)
    num_sents = opt.get('num_sents', -1)
    assert split != 'train', 'Check the evaluation split.'

    model.eval()

    loader.resetIterator(split)
    loss_sum = 0
    loss_evals = 0
    acc = 0
    predictions = []
    finish_flag = False
    model_time = 0

    while True:
        data = loader.getTestBatch(split, opt)
        att_weights = loader.get_attribute_weights()
        sent_ids = data['sent_ids']
        Feats = data['Feats']
        labels = data['labels']
        enc_labels = data['enc_labels']
        dec_labels = data['dec_labels']
        image_id = data['image_id']
        ann_ids = data['ann_ids']
        att_labels, select_ixs = data['att_labels'], data['select_ixs']
        #sim = data['sim']

        ######### new data  ############

        #sub_nounids = data['sub_nounids']
        #sub_wordids = data['sub_wordids']
        sub_wordembs = data['sub_wordembs']

        #sub_classids = data['sub_classids']
        #sub_classwordids = data['sub_classwordids']
        sub_classembs = data['sub_classembs']

        #obj_nounids = data['obj_nounids']
        #obj_wordids = data['obj_wordids']
        obj_wordembs = data['obj_wordembs']

        #rel_prepids = data['rel_prepids']
        #rel_wordids = data['rel_wordids']
        rel_wordembs = data['rel_wordembs']

        ann_pool5 = data['ann_pool5']
        ann_fc7 = data['ann_fc7']
        ann_fleats = data['ann_fleats']

        expand_ann_ids = data['expand_ann_ids']

        ################################


        for i, sent_id in enumerate(sent_ids):

            ########### new data #################

            #sub_nounid = sub_nounids[i:i+1]
            #sub_wordid = sub_wordids[i:i+1]
            sub_wordemb = sub_wordembs[i:i + 1]

            #sub_classid = sub_classids[i:i + 1]
            #sub_classwordid = sub_classwordids[i:i + 1]
            sub_classemb = sub_classembs[i:i + 1]

            #obj_nounid = obj_nounids[i:i+1]
            #obj_wordid = obj_wordids[i:i+1]
            obj_wordemb = obj_wordembs[i:i + 1]

            #rel_prepid = rel_prepids[i:i+1]
            #rel_wordid = rel_wordids[i:i+1]
            rel_wordemb = rel_wordembs[i:i + 1]

            #######################################

            enc_label = enc_labels[i:i + 1] # (1, sent_len)
            max_len = (enc_label != 0).sum().data[0]
            enc_label = enc_label[:, :max_len]  # (1, max_len)
            dec_label = dec_labels[i:i + 1]
            dec_label = dec_label[:, :max_len]

            label = labels[i:i + 1]
            max_len = (label != 0).sum().data[0]
            label = label[:, :max_len]  # (1, max_len)

            pool5 = Feats['pool5']
            fc7 = Feats['fc7']
            lfeats = Feats['lfeats']
            dif_lfeats = Feats['dif_lfeats']
            dist = Feats['dist']
            cxt_fc7 = Feats['cxt_fc7']
            cxt_lfeats = Feats['cxt_lfeats']
            #sub_sim = sim['sub_sim'][i:i+1]
            #obj_sim = sim['obj_sim'][i:i+1]
            #sub_emb = sim['sub_emb'][i:i+1]
            #obj_emb = sim['obj_emb'][i:i+1]

            att_label = att_labels[i:i + 1]
            if i in select_ixs:
                select_ix = torch.LongTensor([0]).cuda()
            else:
                select_ix = torch.LongTensor().cuda()

            tic = time.time()

            scores, loss, sub_loss, obj_loss, rel_loss = model(pool5, fc7, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist, label, enc_label, dec_label, att_label, select_ix, att_weights,
                                                      sub_wordemb, sub_classemb, obj_wordemb, rel_wordemb,
                                                      ann_pool5, ann_fc7, ann_fleats)

            scores = scores.squeeze(0)

            loss = loss.data[0].item()

            pred_ix = torch.argmax(scores)

            pred_ann_id = expand_ann_ids[pred_ix]

            gd_ix = data['gd_ixs'][i]
            loss_sum += loss
            loss_evals += 1

            pred_box = loader.Anns[pred_ann_id]['box']
            gd_box = data['gd_boxes'][i]

            IoU = computeIoU(pred_box, gd_box)
            if opt['use_IoU'] > 0:
                if IoU >= 0.5:
                    acc += 1
            else:
                if pred_ix == gd_ix:
                    acc += 1

            entry = {}
            entry['image_id'] = image_id
            entry['sent_id'] = sent_id
            entry['sent'] = loader.decode_labels(label.data.cpu().numpy())[0]  # gd-truth sent
            entry['gd_ann_id'] = data['ann_ids'][gd_ix]
            entry['pred_ann_id'] = pred_ann_id
            entry['pred_score'] = scores.tolist()[pred_ix]
            entry['IoU'] = IoU
            entry['ann_ids'] = ann_ids

            predictions.append(entry)
            toc = time.time()
            model_time += (toc - tic)

            if num_sents > 0  and loss_evals >= num_sents:
                finish_flag = True
                break
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']


        if verbose:
            print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%, (%.4f), model time (per sent) is %.2fs' % \
                  (split, ix0, ix1, acc*100.0/loss_evals, loss, model_time/len(sent_ids)))



        model_time = 0

        if finish_flag or data['bounds']['wrapped']:
            break

    return loss_sum / loss_evals, acc / loss_evals, predictions


