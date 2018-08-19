import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import pdb
from pycocotools.coco import COCO
import misc.utils as utils
from datasets.factory import get_imdb

class dataset(Dataset):
    def __init__(self, opt, roidb, split='train', transform=None, target_transform=None): 

        self.opt = opt
        self._roidb = roidb
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.shape = (opt.width, opt.height)
        self.seen = opt.seen
        self._num_classes = opt.classes

    def __len__(self):
        return len(self._roidb)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        minibatch_db = self._roidb[index]

        image_id = minibatch_db['img_id']
        image_path = minibatch_db['image']

        if self.split == 'train':

            if self.opt.use_all_gt:
                gt_inds = np.where(minibatch_db['gt_classes'] != 0)[0]
            else:
                gt_inds = np.where(minibatch_db['gt_classes'] != 0 & \
                    np.all(minibatch_db['gt_overlaps'].toarray() > -1.0, axis=1))[0]

            gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)

            gt_boxes[:, 0:4] = minibatch_db['boxes'][gt_inds, :]
            gt_boxes[:, 4] = minibatch_db['gt_classes'][gt_inds] - 1

            img, box, iw, ih = utils.get_random_data(image_path, gt_boxes, self.shape, random=True)
            y_true = utils.preprocess_true_boxes(box[np.newaxis,:], self.shape, np.array(self.opt.anchors), self.opt.classes)
        else:
            gt_inds = np.where(minibatch_db['gt_classes'] != 0)[0]
            gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
            gt_boxes[:, 0:4] = minibatch_db['boxes'][gt_inds, :]
            gt_boxes[:, 4] = minibatch_db['gt_classes'][gt_inds] - 1
            img, box, iw, ih = utils.get_random_data(image_path, gt_boxes, self.shape, random=False)
            
            y_true = utils.preprocess_true_boxes(box[np.newaxis,:], self.shape, np.array(self.opt.anchors), self.opt.classes)

        y_true0 = torch.from_numpy(y_true[0].squeeze(0))
        y_true1 = torch.from_numpy(y_true[1].squeeze(0))
        y_true2 = torch.from_numpy(y_true[2].squeeze(0))

        y_shape = torch.Tensor([iw, ih])
        if self.transform:
            img = self.transform(img).float()
        
        return (img, y_true0, y_true1, y_true2, image_id, y_shape)
