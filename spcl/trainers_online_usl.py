from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils.meters import AverageMeter


class SpCLTrainer_USL(object):
    def __init__(self, encoder, memory, source_classes,writer,hierarchy_lp,target_samples,source_samples,update_iters=400):
        super(SpCLTrainer_USL, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.source_classes = source_classes
        self.writer=writer
        self.hierarchy_lp=hierarchy_lp
        self.target_samples=target_samples
        self.source_samples=source_samples
        self.update_iters=update_iters

    def train(self, epoch, data_loader_target,
                    optimizer, print_freq=10, train_iters=400,all_iters=-1,lr_scheduler=None,pretrain=0,only_update_label=0):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            # load data
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            t_inputs, _, t_indexes = self._parse_data(target_inputs)


            # forward
            f_out = self._forward(t_inputs)

            loss = self.memory(f_out, t_indexes)

            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward() #update target feature here
            optimizer.step()

            if epoch>0:
                self.hierarchy_lp.inference(t_indexes,self.memory,1) #target domain-->inference
            #postprocess

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            #import pdb;pdb.set_trace()
            if (i + 1) % print_freq == 0:
                #print("t_indexes:",t_indexes)
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))
            if (all_iters !=-1):
                if (all_iters+i)%self.update_iters==0 and all_iters!=0:
                    print('{}----update lr scheduler--------'.format((all_iters+i)//self.update_iters))
                    lr_scheduler.step()

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
