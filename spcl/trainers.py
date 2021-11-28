from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils.meters import AverageMeter


class SpCLTrainer_UDA(object):
    def __init__(self, encoder, memory, source_classes,writer,gcn_n,gcn_s,target_samples,source_samples):
        super(SpCLTrainer_UDA, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.source_classes = source_classes
        self.writer=writer
        self.gcn_n=gcn_n
        self.gcn_s=gcn_s
        self.target_samples=target_samples
        self.source_samples=source_samples

    def train(self, epoch, data_loader_source, data_loader_target,
                    optimizer, print_freq=10, train_iters=400,all_iters=-1,lr_scheduler=None,lr_scheduler_gcn=None,optimizer_gcn=None,pretrain=0,train_gcn=0,only_update_label=0):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_s = AverageMeter()
        losses_t = AverageMeter()
        losses_gcn= AverageMeter()

        end = time.time()
        if only_update_label:
            self.encoder.eval()
            for i in range(train_iters):
                #import pdb;pdb.set_trace()
                target_inputs = data_loader_target.next()
                t_inputs, _, t_indexes = self._parse_data(target_inputs)
                device_num = torch.cuda.device_count()
                B, C, H, W = s_inputs.size()
                def reshape(inputs):
                    return inputs.view(device_num, -1, C, H, W)
                t_inputs=reshape(t_inputs)
                inputs=t_inputs.view(-1,C,H,W)
                f_out_t = self._forward(inputs)
                f_out_t = f_out_t.view(device_num, -1, f_out_t.size(-1))
                f_out_t= f_out_t.contiguous().view(-1, f_out_t.size(-1))
                with torch.no_grad():
                    for x, y in zip(f_out_t, t_indexes):
                        self.memory.features[y] = self.memory.momentum * self.memory.features[y] + (1. - self.memory.momentum) * x
                        self.memory.features[y] /= self.memory.features[y].norm()

        else:
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
                if not pretrain:
                    s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
                    inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)
                    # forward
                    f_out = self._forward(inputs)
                    # de-arrange batch
                    f_out = f_out.view(device_num, -1, f_out.size(-1))
                    f_out_s, f_out_t = f_out.split(f_out.size(1)//2, dim=1)
                    f_out_s, f_out_t = f_out_s.contiguous().view(-1, f_out.size(-1)), f_out_t.contiguous().view(-1, f_out.size(-1))
                else:
                    s_inputs=reshape(s_inputs)
                    inputs=s_inputs.view(-1,C,H,W)
                    # forward
                    f_out_s = self._forward(inputs)
                    f_out_s = f_out_s.view(device_num, -1, f_out_s.size(-1))
                    f_out_s= f_out_s.contiguous().view(-1, f_out_s.size(-1))
                    #print('f_out_s.shape:',f_out_s.shape)

                # compute loss with the hybrid memory
                #import pdb;pdb.set_trace()
                loss_s = self.memory(f_out_s, s_targets,0,self.gcn_n,self.gcn_s)
                if not pretrain:
                    domain=0 if epoch==0 else 1
                    #domain=1
                    loss_t = self.memory(f_out_t, t_indexes+self.source_classes,domain,self.gcn_n,self.gcn_s)
                    domain=0 if (epoch==0 and i<=200) else 1
                    if epoch==0 and i==200:#initialize sub label
                        pass
                    #domain=1
                    #loss_gcn_n=torch.tensor(0)
                    if len(self.memory.s_features)!=self.source_samples: #add tgt info
                        self.memory.s_features[self.source_samples:]=self.memory.features[-self.target_samples:]
                        self.memory.s_label[self.source_samples:]=self.memory.labels[self.source_classes:]
                        #loss_gcn_n,all_neighbor,all_pred=self.gcn_n(s_indexes,self.memory.s_features,self.memory.s_label,domain,return_loss=True)

                        #use all features
                        all_index=torch.cat((s_indexes,t_indexes+self.source_samples),0)
                    else:
                        all_index=s_indexes
                    try:
                        loss_gcn_n,all_neighbor,all_pred=self.gcn_n(all_index,self.memory.s_features,self.memory.s_label,domain,return_loss=True)
                    except:
                        import pdb;pdb.set_trace()
                else:
                    domain=1 if train_gcn else 0
                    #combine source & target
                    self.memory.s_features[-self.target_samples:]=self.memory.features[-self.target_samples:]
                    loss_gcn_n,all_neighbor,all_pred=self.gcn_n(s_indexes,self.memory.s_features,self.memory.s_label,domain,return_loss=True)
                    #loss_gcn_n1,all_neighbor1,all_pred1=self.gcn_n(s_indexes,self.memory.s_features,self.memory.s_label,domain,return_loss=True)
                    #loss_gcn_n2,all_neighbor2,all_pred2=self.gcn_n(t_indexes+self.source_classes,self.memory.features,self.memory.labels,domain,return_loss=True)
                    #loss_gcn_n=loss_gcn_n1+loss_gcn_n2
                #update source domain features
                with torch.no_grad():
                    for x, y in zip(f_out_s, s_indexes):
                        self.memory.s_features[y] = self.memory.momentum * self.memory.s_features[y] + (1. - self.memory.momentum) * x
                        self.memory.s_features[y] /= self.memory.s_features[y].norm()

                    #update sub cluster label

                if not pretrain:
                    loss = loss_s+loss_t#+loss_gcn_n#+loss_gcn_s
                    self.writer.add_scalar('loss', loss, global_step=epoch*train_iters+i)
                    self.writer.add_scalar('loss_s', loss_s, global_step=epoch*train_iters+i)
                    self.writer.add_scalar('loss_t', loss_t, global_step=epoch*train_iters+i)
                else:
                    loss=loss_s
                    loss_t=torch.tensor(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if domain:
                    optimizer_gcn.zero_grad()
                    loss_gcn_n.backward()
                    optimizer_gcn.step()

                losses_s.update(loss_s.item())
                losses_t.update(loss_t.item())
                losses_gcn.update(loss_gcn_n.item())

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
                          'Loss_gcn {:.6f}({:.6f})\t'
                          'LR {:.8f}'
                          .format(epoch, i + 1, train_iters,
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses_s.val, losses_s.avg,
                                  losses_t.val, losses_t.avg,losses_gcn.val,losses_gcn.avg,optimizer.param_groups[0]['lr']))
                if (all_iters !=-1):
                    if (all_iters+i)%400==0 and all_iters!=0:
                        print('{}----update lr scheduler--------'.format((all_iters+i)//400))
                        lr_scheduler.step()
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
