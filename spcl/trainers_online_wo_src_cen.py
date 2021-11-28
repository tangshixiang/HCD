from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils.meters import AverageMeter


class SpCLTrainer_UDA(object):
    def __init__(self, encoder, memory, source_nums,writer,hierarchy_gcn,target_samples,source_samples):
        super(SpCLTrainer_UDA, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.source_nums = source_nums
        self.writer=writer
        self.hierarchy_gcn=hierarchy_gcn
        self.target_samples=target_samples
        self.source_samples=source_samples

    def train(self, epoch, data_loader_source, data_loader_target,
                    optimizer, print_freq=10, train_iters=400,all_iters=-1,lr_scheduler=None,lr_scheduler_gcn=None,optimizer_gcn=None,pretrain=0,train_gcn=0,only_update_label=0):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_s = AverageMeter()
        losses_t = AverageMeter()
        losses_p_gcn= AverageMeter()
        losses_s_gcn= AverageMeter()
        losses_c_gcn= AverageMeter()

        end = time.time()

        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            if not pretrain:
                target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
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

            ##########initialize sub cluster##############
            if epoch==0 and i==200:#initialize src sub
                self.hierarchy_gcn.utils.initialize_sub_cluster_label(self.memory.labels,self.memory.sub_label,self.memory.features,start=0,end=self.source_nums)
            #if epoch==1 and i==0:
            #    self.hierarchy_gcn.utils.initialize_sub_cluster_label(self.memory.labels[self.source_nums:],self.memory.sub_label[self.source_nums:],self.memory.features[self.source_nums:],start=0,end=len(self.memory.sub_label)-self.source_nums)
            #############################################

            #gcn forward
            train=0 if (epoch==0 and i<=200) else 1
            loss_gcn=self.hierarchy_gcn.train(s_indexes,self.memory,train)

            loss_s = self.memory(f_out_s, s_indexes)
            loss_t = self.memory(f_out_t, t_indexes+self.source_nums)
            loss=loss_s+loss_t

            losses_s.update(loss_s.item())
            losses_t.update(loss_t.item())
            losses_p_gcn.update(loss_gcn[0].item())
            losses_s_gcn.update(loss_gcn[1].item())
            losses_c_gcn.update(loss_gcn[2].item())

            optimizer.zero_grad()
            loss.backward() #update target feature here
            optimizer.step()

            if train:
                if isinstance(optimizer_gcn,list):
                    for idx,optim in enumerate(optimizer_gcn):
                        optim.zero_grad()
                        loss_gcn[idx].backward()
                        optim.step()
                else:
                    optimizer_gcn.zero_grad()
                    loss_gcn.backward()
                    optimizer_gcn.step()
                self.hierarchy_gcn.postprocess(s_indexes,self.memory) #update sub cluster label

            if epoch>0:
                self.hierarchy_gcn.inference(t_indexes+self.source_nums,self.memory,1) #target domain-->inference

            #update src features
            #with torch.no_grad():
            #    for x, y in zip(f_out_s, s_indexes):
            #        self.memory.features[y] = self.memory.momentum * self.memory.features[y] + (1. - self.memory.momentum) * x
            #        self.memory.features[y] /= self.memory.features[y].norm()

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
                      'Loss_p_gcn {:.6f}({:.6f})\t'
                      'Loss_s_gcn {:.6f}({:.6f})\t'
                       'Loss_c_gcn {:.6f}({:.6f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_s.val, losses_s.avg,
                              losses_t.val, losses_t.avg,losses_p_gcn.val,losses_p_gcn.avg,
                              losses_s_gcn.val,losses_s_gcn.avg,losses_c_gcn.val,losses_c_gcn.avg))
            if (all_iters !=-1):
                if (all_iters+i)%400==0 and all_iters!=0:
                    print('{}----update lr scheduler--------'.format((all_iters+i)//400))
                    lr_scheduler.step()
                    if isinstance(lr_scheduler_gcn,list):
                        for sch in lr_scheduler_gcn:
                            sch.step()
                    else:
                        lr_scheduler_gcn.step()

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
