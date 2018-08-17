from __future__ import print_function
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from six.moves import cPickle
from pprint import pprint

import random
import math
import os

import yaml
import pdb
import argparse
import datetime

import misc.utils as utils
import misc.yolo as yolo
import misc.dataset as dataset
from misc.roidb import combined_roidb

def train(epoch, opt):
    torch.set_grad_enabled(True)
    model.train()
    iter_train = iter(train_loader)
    start = time.time()
    tmp_losses = 0
    count = 0

    for batch_idx in range(len(train_loader)):
        opt.n_iter = opt.n_iter + 1
        opt.seen = opt.seen + opt.batch_size
        t0 = time.time()
        data = iter_train.next()
        img, label1, label2, label3, image_id = data
        
        if opt.use_cuda:
            img = img.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()
            label3 = label3.cuda()

        losses = model(img, label1, label2, label3)
        
        loss = utils.add_logger(opt, model, losses, logger, opt.n_iter, 'train')
        loss = loss.sum() / loss.numel()

        tmp_losses = tmp_losses + loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % opt.display_interval == 0 and batch_idx != 0:
            end = time.time()
            tmp_losses = tmp_losses / opt.display_interval
            print("step {}/{} (epoch {}), loss: {:f} , lr:{:f}, time/batch = {:.3f}" \
                .format(batch_idx, len(train_loader), epoch, tmp_losses, optimizer.param_groups[-1]['lr'], end - start))

            start = time.time()
            tmp_losses = 0

def validation(epoch, opt):
    torch.set_grad_enabled(False)
    model.eval()

    iter_val = iter(eval_loader)
    tmp_losses = 0
    count = 0

    for batch_idx in range(len(iter_val)):
        data = iter_val.next()
        img, label1, label2, label3, image_id = data

        if opt.use_cuda:
            img = img.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()
            label3 = label3.cuda()

        losses = model(img, label1, label2, label3)
        loss = utils.add_logger(opt, model, losses, logger, opt.n_iter, 'val')
        tmp_losses = tmp_losses + loss.item()
            
    print("Evaluation Loss (epoch {}), TOTAL_LOSS: {:.3f}".format(epoch, tmp_losses))

    return tmp_losses

def _get_optimizer(opt, net):

    params = []
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            if 'backbone' in key:
                params += [{'params':[value], 'lr':opt.backbone_lr}]
            else:
                params += [{'params':[value], 'lr':opt.lr}]

    # Initialize optimizer class
    if opt.optimizer == "adam":
        optimizer = optim.Adam(params, weight_decay=opt.weight_decay)
    elif opt.optimizer == "amsgrad":
        optimizer = optim.AMSgrad(params, weight_decay=opt.weight_decay,
                               amsgrad=True)
    elif opt.optimizer == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=opt.weight_decay)
    else:
        # Default to sgd
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=opt.weight_decay,
                              nesterov=(opt.optimizer == "nesterov"))
    return optimizer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # # Data input settings
    parser.add_argument('--opt_path', type=str, default='cfgs/coco_det.yml',
                     help='')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether use gpu.')
    parser.add_argument('--mGPUs', type=bool, default=False, help='whether use mgpu.')

    opt = parser.parse_args()
    if opt.opt_path is not None:
        with open(opt.opt_path, 'r') as handle:
            options_yaml = yaml.load(handle)
        utils.update_values(options_yaml, vars(opt))

    print(opt)

    log_name = str(datetime.datetime.now()) + '_' + '_' + 'bs:' + str(opt.batch_size)
    # log_name = 'test'
    print("logging to %s ..." %(log_name))
    logger = utils.set_tb_logger('logging', log_name)
    
    cudnn.benchmark = True
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    
    if opt.seed: 
        torch.manual_seed(opt.seed)
        if opt.use_cuda:
            torch.cuda.manual_seed(opt.seed)

    ####################################################################################
    # Data Loader
    ####################################################################################
    
    imdb, roidb = combined_roidb(opt.imdb_name)
    imdb_val, roidb_val = combined_roidb(opt.imdbval_name, training=False)

    kwargs = {'num_workers': opt.num_workers, 'pin_memory': True} if opt.use_cuda else {}       
    train_dataset = dataset.dataset(opt, roidb, transform=transforms.Compose([
                                        transforms.ToTensor(),]
                                        ), split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                    batch_size=opt.batch_size, **kwargs)  
    
    eval_dataset = dataset.dataset(opt, roidb_val, transform=transforms.Compose([
                                        transforms.ToTensor(),]
                                        ), split='val')
    eval_loader = torch.utils.data.DataLoader(eval_dataset, shuffle=False,
                                    batch_size=opt.batch_size, **kwargs)  

    ####################################################################################
    # Initialize the model
    ####################################################################################

    model = yolo.YOLOv3(opt)

    infos = {}
    if opt.start_from != '':
        if opt.load_best_score == 1:
            model_path = os.path.join(opt.start_from, 'model-best.pth')
            info_path = os.path.join(opt.start_from, 'infos-best.pkl')
        else:
            model_path = os.path.join(opt.start_from, 'model.pth')
            info_path = os.path.join(opt.start_from, 'infos.pkl')

        with open(info_path, 'rb') as f: 
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']

        print('Loading the model from %s ...' %(model_path))
        model.load_state_dict(torch.load(model_path))

    opt.n_iter = infos.get('n_iter', opt.n_iter)
    start_epoch = infos.get('epoch', opt.start_epoch)
    best_val_loss = infos.get('best_val_loss', 100000)
    optimizer = _get_optimizer(opt, model)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.8)

    if opt.use_cuda:
        if opt.mGPUs:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    for epoch in range(start_epoch, opt.max_epochs):

        train(epoch, opt)
        val_loss = validation(epoch, opt)
        scheduler.step()
        # Save model if is improving on validation loss
        best_flag = False
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_flag = True

        checkpoint_path = os.path.join(opt.save_path, opt.backbones_type)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if opt.mGPUs > 1:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_path, 'model.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model.pth'))

        print("model saved to {}".format(checkpoint_path))

        infos['n_iter'] = opt.n_iter
        infos['epoch'] = epoch
        infos['best_val_loss'] = best_val_loss
        infos['opt'] = opt

        with open(os.path.join(checkpoint_path, 'infos.pkl'), 'wb') as f:
            cPickle.dump(infos, f)

        if best_flag:
            if opt.mGPUs > 1:
                torch.save(model.module.state_dict(), os.path.join(checkpoint_path, 'model-best.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model-best.pth'))

            print("model saved to {} with best total loss {:.3f}".format(os.path.join(checkpoint_path, \
                                'model-best.pth'), best_val_loss))
            
            with open(os.path.join(checkpoint_path, 'infos-best.pkl'), 'wb') as f:
                cPickle.dump(infos, f)