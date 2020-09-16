from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

import argparse
import os
import os.path as osp
import numpy as np
import torch  # put this before scipy import
from scipy.misc import imread, imresize
import sys
sys.path.insert(0, '../tools')
import cv2

from mattnet import MattNet


def mean_iou(labels, predictions):
  labels_bool = (labels == 255)
  pred_bool = (predictions == 255)

  labels_bool_sum = (labels_bool).sum()
  pred_bool_sum = (pred_bool).sum()

  if (labels_bool_sum > 0) or (pred_bool_sum > 0):
    intersect_all = np.logical_and(labels_bool, pred_bool)
    intersect = intersect_all.sum()
    union = labels_bool_sum + pred_bool_sum - intersect

  return float(intersect) / float(union)

# box functions
def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


def show_attn(img_path, box, attn):
  """
  box : [xywh]
  attn: 49
  """
  img = imread(img_path)
  attn = np.array(attn).reshape(7, 7)
  x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
  roi = img[y:y + h - 1, x:x + w - 1]
  attn = imresize(attn, [h, w])
  plt.imshow(roi)
  plt.imshow(attn, alpha=0.7)


def show_boxes(img_path, boxes, colors, texts=None, masks=None):
  # boxes [[xyxy]]
  img = imread(img_path)
  plt.imshow(img)
  ax = plt.gca()
  for k in range(boxes.shape[0]):
    box = boxes[k]
    xmin, ymin, xmax, ymax = list(box)
    coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
    color = colors[k]
    ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    if texts is not None:
      ax.text(xmin, ymin, texts[k], bbox={'facecolor': color, 'alpha': 0.5})
  # show mask
  if masks is not None:
    for k in range(len(masks)):
      mask = masks[k]
      m = np.zeros((mask.shape[0], mask.shape[1], 3))
      m[:, :, 0] = 0;
      m[:, :, 1] = 0;
      m[:, :, 2] = 1.
      ax.imshow(np.dstack([m * 255, mask * 255 * 0.4]).astype(np.uint8))

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='refcoco', help='dataset name: refclef, refcoco, refcoco+, refcocog')
parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
parser.add_argument('--model_id', type=str, default='mrcn_cmr_with_st', help='model id name')
args = parser.parse_args('')

# MattNet
mattnet = MattNet(args)

# image path

sequence_name = "walking"

root_annotation_path = "/home/smj/DataSet/DAVIS2017/Annotations/"
sequence_png_path = os.path.join(root_annotation_path, "480p_individual", sequence_name)
expr_path = os.path.join(root_annotation_path, "davis_text_annotations", "DAVIS2017", sequence_name+".txt")
performance_path = os.path.join(root_annotation_path, "language_performance", sequence_name)

root_image_path = "/home/smj/DataSet/DAVIS2017/JPEGImages/480p/"
sequence_image_path = os.path.join(root_image_path, sequence_name)

# take the name for each image frame
image_name_list = os.listdir(sequence_image_path)
image_name_list.sort()

# take the number of the frames in this sequence
frame_num = len(image_name_list)


# take the number of targets in this sequence
target_list = os.listdir(sequence_png_path)
target_list.sort()
target_num = len(target_list)

# take all expressions for this sequence
f_exprs = open(expr_path, "r+")
expr_list = f_exprs.read().splitlines()
f_exprs.close()

# make the directory and the text files to record the performance for this sequence
# make the directory
if not os.path.isdir(performance_path):
  os.mkdir(performance_path)

# make the text files for the performance for each target
for i in range(target_num):
  performance_target_path = os.path.join(performance_path, sequence_name+"_"+str(i)+".txt")
  if not os.path.exists(performance_target_path):
    f_performance = open(performance_target_path, "w")
    f_performance.close()



for target_id in range(target_num):
  print("target: " + str(target_id))
  expr = expr_list[target_id]
  # npy file path is used to save the feature of the expression
  npy_target_path = os.path.join(performance_path, sequence_name+"_"+str(target_id)+".npy")
  f_target_performace = open(os.path.join(performance_path, sequence_name+"_"+str(target_id)+".txt"), "a")
  png_target_path = os.path.join(sequence_png_path, sequence_name + "_" + str(target_id))
  png_name_list = os.listdir(png_target_path)
  png_name_list.sort()

  #for frame_id in range(frame_num):
  for frame_id in range(3):
    print("frame: " + str(frame_id))
    image_name = image_name_list[frame_id]
    image_path = os.path.join(sequence_image_path, image_name)
    img = cv2.imread(image_path)

    with torch.no_grad():
      img_data = mattnet.forward_image(image_path, nms_thresh=0.3, conf_thresh=0.50)
      entry, hidden = mattnet.comprehend(img_data, expr)
      pred_mask = entry["pred_mask"] * 255

    # save the feature of expression as a npy file
    expr_feature = hidden[0,:]
    if not os.path.exists(npy_target_path):
      np.save(npy_target_path, expr_feature)

    png_name = png_name_list[frame_id]
    png_path = os.path.join(png_target_path, png_name)
    label = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)

    mask_iou = mean_iou(pred_mask, label)

    #f_target_performace.write(str(mask_iou) + "\n")







#img_data = mattnet.forward_image(img_path, nms_thresh=0.3, conf_thresh=0.50)
#entry = mattnet.comprehend(img_data, expr)
#pred_mask = entry["pred_mask"] * 255
#mask_iou = mean_iou(pred_mask, label)

''''
plt.rcParams['figure.figsize'] = (10., 8.)
tokens = expr.split()
print('sub(%.2f):' % entry['weights'][0], ''.join(['(%s,%.2f)'% (tokens[i], s) for i, s in enumerate(entry['sub_attn'])]))
print('loc(%.2f):' % entry['weights'][1], ''.join(['(%s,%.2f)'% (tokens[i], s) for i, s in enumerate(entry['loc_attn'])]))
print('rel(%.2f):' % entry['weights'][2], ''.join(['(%s,%.2f)'% (tokens[i], s) for i, s in enumerate(entry['rel_attn'])]))
# predict attribute on the predicted object
print(entry['pred_atts'])
# show prediction
plt.rcParams['figure.figsize'] = (12., 8.)
fig = plt.figure()
plt.subplot(121)
show_boxes(img_path, xywh_to_xyxy(np.vstack([entry['pred_box']])), ['blue'], texts=None, masks=[entry['pred_mask']])
plt.subplot(122)

#plt.imshow(fig)
plt.axis('off')
plt.show()
'''















'''
# forward image
img_data = mattnet.forward_image(img_path, nms_thresh=0.3, conf_thresh=0.50)

# show masks
#plt.rcParams['figure.figsize'] = (10., 8.)

#dets = img_data['dets']

#show_boxes(img_path, xywh_to_xyxy(np.array([det['box'] for det in dets])), ['blue']*len(dets), ['%s(%.2f)' % (det['category_name'], det['score']) for det in dets])

# comprehend expression
entry = mattnet.comprehend(img_data, expr)


# visualize
tokens = expr.split()
print('sub(%.2f):' % entry['weights'][0], ''.join(['(%s,%.2f)'% (tokens[i], s) for i, s in enumerate(entry['sub_attn'])]))
print('loc(%.2f):' % entry['weights'][1], ''.join(['(%s,%.2f)'% (tokens[i], s) for i, s in enumerate(entry['loc_attn'])]))
print('rel(%.2f):' % entry['weights'][2], ''.join(['(%s,%.2f)'% (tokens[i], s) for i, s in enumerate(entry['rel_attn'])]))
# predict attribute on the predicted object
print(entry['pred_atts'])
# show prediction
plt.rcParams['figure.figsize'] = (12., 8.)
fig = plt.figure()
plt.subplot(121)
show_boxes(img_path, xywh_to_xyxy(np.vstack([entry['pred_box']])), ['blue'], texts=None, masks=[entry['pred_mask']])
plt.subplot(122)

#plt.imshow(fig)
plt.axis('off')
plt.show()

#show_attn(img_path, entry['pred_box'], entry['sub_grid_attn'])
'''
