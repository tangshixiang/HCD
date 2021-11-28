from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils.meters import AverageMeter


class SpCLTrainer_UDA(object):
    def __init__(self, encoder, memory, source_classes,writer,hierarchy_lp,target_samples,source_samples,update_iters=400):
        super(SpCLTrainer_UDA, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.source_classes = source_classes
        self.writer=writer
        self.hierarchy_lp=hierarchy_lp
        self.target_samples=target_samples
        self.source_samples=source_samples
        self.update_iters=update_iters

    def train(self, epoch, data_loader_source, data_loader_target,
                    optimizer, print_freq=10, train_iters=400,all_iters=-1,lr_scheduler=None,pretrain=0,only_update_label=0):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_s = AverageMeter()
        losses_t = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            #import pdb;pdb.set_trace()
            s_inputs, s_targets, s_indexes = self._parse_data(source_inputs)
            t_inputs, _, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s_inputs.size()
            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)
            # forward
            f_out = self._forward(inputs)
            # de-arrange batch
            f_out = f_out.view(device_num, -1, f_out.size(-1))
            f_out_s, f_out_t = f_out.split(f_out.size(1)//2, dim=1)
            f_out_s, f_out_t = f_out_s.contiguous().view(-1, f_out.size(-1)), f_out_t.contiguous().view(-1, f_out.size(-1))

            loss_s = self.memory(f_out_s, s_targets)
            loss_t = self.memory(f_out_t, t_indexes+self.source_classes)
            loss=loss_s+loss_t

            #gcn forward
            #train=0 if (epoch==0 and i<=self.update_iters//2) else 1

            losses_s.update(loss_s.item())
            losses_t.update(loss_t.item())

            optimizer.zero_grad()
            loss.backward() #update target feature here
            optimizer.step()

            if epoch>0:
                self.hierarchy_lp.inference(t_indexes+self.source_classes,self.memory,1) #target domain-->inference
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
                      'Loss_s {:.3f} ({:.3f})\t'
                      'Loss_t {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_s.val, losses_s.avg,
                              losses_t.val, losses_t.avg))
            if (all_iters !=-1):
                if (all_iters+i)%self.update_iters==0 and all_iters!=0:
                    print('{}----update lr scheduler--------'.format((all_iters+i)//self.update_iters))
                    lr_scheduler.step()

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class SpCLTrainer_USL(object):
    def __init__(self, encoder, memory):
        super(SpCLTrainer_USL, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer,print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, _, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)

            # compute loss with the hybrid memory
            loss = self.memory(f_out, indexes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f}\t'
                      'LR {:.8f}'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,optimizer.param_groups[0]['lr']))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
