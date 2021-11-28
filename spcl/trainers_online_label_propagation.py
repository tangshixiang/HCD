from __future__ import print_function, absolute_import
import time
import numpy as np
import collections
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils.meters import AverageMeter


class SpCLTrainer_UDA(object):
    def __init__(self, encoder, memory, source_classes,writer,hierarchy_gcn,target_samples,source_samples,update_iters=400,update_method=1):
        super(SpCLTrainer_UDA, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.source_classes = source_classes
        self.writer=writer
        self.hierarchy_gcn=hierarchy_gcn
        self.target_samples=target_samples
        self.source_samples=source_samples
        self.update_iters=update_iters
        self.update_method=update_method

    def train(self, epoch, data_loader_source, data_loader_target,
                    optimizer, print_freq=10, train_iters=400,all_iters=-1,lr_scheduler=None,pretrain=0,train_gcn=0,only_update_label=0,base=0,update_before=0,args=None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_s = AverageMeter()
        losses_t = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            if not pretrain:
                target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            #import pdb;pdb.set_trace()
            s_inputs, s_targets, s_indexes = self._parse_data(source_inputs)
            if not pretrain:
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

            # ##########initialize sub cluster##############
            # if epoch==0 and i==self.update_iters//2:#initialize src sub
            #     self.hierarchy_gcn.utils.initialize_sub_cluster_label(self.memory.s_label,self.memory.s_sub_label,self.memory.s_features)
            # if epoch==1 and i==0:
            #     self.hierarchy_gcn.utils.initialize_sub_cluster_label(self.memory.labels[self.source_classes:],self.memory.t_sub_label[self.source_classes:],self.memory.features[self.source_classes:],start=self.source_classes)
            # #    #!!!!!t_sub_label+=self.source_classes
            #############################################

            #update before
            if update_before:
                with torch.no_grad():
                    tmp=self.memory.features[t_indexes + self.source_classes]#save ori feat
                    for x, y in zip(f_out_t, t_indexes + self.source_classes):
                        self.memory.features[y] = self.memory.momentum * self.memory.features[y] + (1. - self.memory.momentum) * x
                        self.memory.features[y] /= self.memory.features[y].norm()
                if self.update_method == 1:
                    if base == 0 and epoch % 2 == 1:  # 2-->change
                        self.hierarchy_gcn.inference(t_indexes + self.source_classes, self.memory,
                                                     1)  # target domain-->inference
                if self.update_method == 2:
                    if base == 0 and ((all_iters + i) // self.update_iters) % 2 == 1:  # update after self.update_iters
                        self.hierarchy_gcn.inference(t_indexes + self.source_classes, self.memory, 1)
                if self.update_method == 3:  #
                    if base == 0 and epoch > 0 and ((epoch - 1) % 4 == 1 or (epoch - 1) % 4 == 0):
                        self.hierarchy_gcn.inference(t_indexes + self.source_classes, self.memory, 1)
                if self.update_method == 4:
                    if base == 0 and epoch > 0:
                        self.hierarchy_gcn.inference(t_indexes + self.source_classes, self.memory, 1)
                self.memory.features[t_indexes + self.source_classes]=tmp#recover to cal loss

            loss_s = self.memory(f_out_s, s_targets)
            loss_t = self.memory(f_out_t, t_indexes+self.source_classes)
            loss=loss_s+loss_t

            #gcn forward
            #train=0 if (epoch==0 and i<=self.update_iters//2) else 1
            #train=0
            # loss_lp,loss_split_gcn=self.hierarchy_gcn.train(s_indexes,self.memory,train,f_out_s)
            # if train:
            #     loss+=0.0001*loss_lp

            losses_s.update(loss_s.item())
            losses_t.update(loss_t.item())
            #losses_p_lp.update(loss_lp.item())
            # losses_s_lp.update(loss_gcn[1].item())
            # losses_c_lp.update(loss_gcn[2].item())

            optimizer.zero_grad()
            # if train:
            #     if isinstance(optimizer_gcn,list):
            #         for idx,optim in enumerate(optimizer_gcn):
            #             optim.zero_grad()
            #     loss_split_gcn.backward()
            loss.backward() #update target feature here
            optimizer.step()

            # if train:
            #     for idx,optim in enumerate(optimizer_gcn):
            #         optim.step()
            #     self.hierarchy_gcn.postprocess(s_indexes,self.memory) #update sub cluster label

            if not update_before:
                if self.update_method==1:
                    if base==0 and epoch%2==1: #2-->change
                        self.hierarchy_gcn.inference(t_indexes+self.source_classes,self.memory,1) #target domain-->inference
                if self.update_method==2:
                    if base==0 and epoch%3==1: #2-->change
                        self.hierarchy_gcn.inference(t_indexes+self.source_classes,self.memory,1) #target domain-->inference
                    # if base==0 and ((all_iters+i)//self.update_iters)%2==1:#update after self.update_iters
                    #     self.hierarchy_gcn.inference(t_indexes + self.source_classes, self.memory, 1)
                if self.update_method==3: #
                    if base==0 and epoch%4==1: #2-->change
                        self.hierarchy_gcn.inference(t_indexes+self.source_classes,self.memory,1) #target domain-->inference
                    # if base==0 and epoch>0 and ((epoch-1)%4==1 or (epoch-1)%4==0):
                    #     self.hierarchy_gcn.inference(t_indexes+self.source_classes,self.memory,1)
                if self.update_method==4:
                    if base==0 and epoch>0:
                        self.hierarchy_gcn.inference(t_indexes+self.source_classes,self.memory,1)
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
                    #####save label -->acc########
                    assert args is not None
                    epp = (all_iters + i) // self.update_iters
                    print('------Epoch[{}]: ep_{}_label.pth.tar'.format(epp,epp))
                    for kk in range(args.iterative):
                        torch.save(self.memory.labels[kk][self.memory.source_classes:].cpu(),osp.join(args.logs_dir, 'ep_{}_label_{}.pth.tar'.format(epp,kk)))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class SpCLTrainer_USL(object):
    def __init__(self, encoder, memory, hierarchy_gcn,target_samples,update_iters=400,update_method=1):
        super(SpCLTrainer_USL, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.hierarchy_gcn = hierarchy_gcn
        self.target_samples=target_samples
        self.update_method=update_method
        self.update_iters=update_iters

    def train(self, epoch, data_loader,
                    optimizer, print_freq=10, train_iters=400,all_iters=-1,lr_scheduler=None,pretrain=0,train_gcn=0,only_update_label=0,base=0,update_before=0):
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

            assert update_before==0
            if not update_before:
                if self.update_method==1:
                    if base==0 and epoch%2==1: #2-->change
                        self.hierarchy_gcn.inference(indexes,self.memory,1) #target domain-->inference
                if self.update_method==2:
                    if base==0 and ((all_iters+i)//self.update_iters)%2==1:#update after self.update_iters
                        self.hierarchy_gcn.inference(indexes, self.memory, 1)
                if self.update_method==3: #
                    if base==0 and epoch>0 and ((epoch-1)%4==1 or (epoch-1)%4==0):
                        self.hierarchy_gcn.inference(indexes,self.memory,1)
                if self.update_method==4:
                    if base==0 and epoch>0:
                        self.hierarchy_gcn.inference(indexes,self.memory,1)

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
            if (all_iters !=-1):
                if (all_iters+i)%self.update_iters==0 and all_iters!=0:
                    print('{}----update lr scheduler--------'.format((all_iters+i)//self.update_iters))
                    lr_scheduler.step()

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
