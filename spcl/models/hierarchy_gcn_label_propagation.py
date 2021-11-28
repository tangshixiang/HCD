#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from spcl.models.utils import GraphConv, MeanAggregator
from spcl.utils.faiss_rerank import compute_jaccard_distance,compute_jaccard_distance_step1,compute_jaccard_distance_inital_rank,compute_knn

class Point_Level_LP(nn.Module):
    def __init__(self,alpha,beta=1,method=1,connect_num=20,topk_num=0.45):
        super(Point_Level_LP, self).__init__()
        #self.loss=torch.nn.CrossEntropyLoss().cuda()
        self.loss=torch.nn.BCEWithLogitsLoss()
        self.eps = np.finfo(float).eps
        self.w_topk=-1
        self.alpha=alpha
        self.beta=beta
        self.once_forward=1
        self.method=method
        self.connect_num=connect_num
        self.topk_num=topk_num
    def forward(self,indexes,features,neighbor_num,ori_0,ori_knn_neighbor,gt_conf=None,f_s=None,train=0,two_hop=0):
        bs=len(indexes)

        if two_hop:
            neighbor_num=400
            Y=torch.zeros((bs,neighbor_num,neighbor_num)).cuda()
            W0=torch.zeros((bs,neighbor_num,neighbor_num)).cuda()
            all_neighbors=torch.zeros((bs,neighbor_num)).long().cuda()-1
            ori_knn_neighbor=ori_knn_neighbor.cpu().numpy()
            for i in range(bs):
                unique_hop_2_neighbor=list(set(np.unique(ori_0[ori_knn_neighbor[i,1:]][:,1:]).tolist())-set(ori_knn_neighbor[i].tolist()))
                all_neighbor=torch.from_numpy(np.concatenate((ori_knn_neighbor[i],np.array(unique_hop_2_neighbor))))
                all_neighbor_feat=features[all_neighbor.long()]
                W0[i,:len(all_neighbor_feat),:len(all_neighbor_feat)]=all_neighbor_feat.mm(all_neighbor_feat.t())
                Y[i,:len(all_neighbor_feat),:len(all_neighbor_feat)]=torch.eye(len(all_neighbor_feat))
                all_neighbors[i,:len(all_neighbor_feat)]=all_neighbor.clone()
            Y[:, 0, 0] = 0  # wo self
        else:
            all_neighbors=ori_knn_neighbor
            # cal Y
            Y = torch.zeros(bs, neighbor_num, neighbor_num).cuda()
            Y[:, :, :neighbor_num] = torch.eye(neighbor_num).unsqueeze(0).repeat(bs, 1, 1)
            Y[:, 0, 0] = 0  # wo self
            # cal W
            index_feat = features[ori_knn_neighbor.view(-1)].view(bs, -1, 2048)
            # index_feat=torch.cat((f_s,index_feat[:,1:]))

            i_feat = torch.cat((f_s.unsqueeze(1), index_feat[:, 1:neighbor_num]), dim=1)
            W0 = i_feat.bmm(i_feat.permute(0, 2, 1))

        mask=(1-torch.eye(neighbor_num)).unsqueeze(0).cuda()

        if self.method==2:
            #import pdb;pdb.set_trace()
            #step1-->k_reciprocal_index
            topk_num=20
            topk, indices = torch.topk(W0, topk_num, dim=2)
            mask_top = torch.zeros_like(W0)
            mask_top = mask_top.scatter(2, indices, 1)
            mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
            #step2-->softmax
            W0 = torch.exp(-(2 - 2 * W0))
            W0*=mask_top
            W0/=torch.sum(W0+self.eps,dim=-1,keepdim=True)
            #avg
            k2=6
            W=torch.zeros_like(W0)
            tmp=torch.arange(bs).unsqueeze(1).expand_as(indices[:,:,0])
            for kk in range(k2):
                W+=W0[tmp,indices[:,:,kk]]
            W/=k2
            W0=W.clone()#keep for split part
            #1-jarrcard distance-->indexes
            W=torch.sum(torch.min(W[:,0,:].unsqueeze(1).expand_as(W),W),dim=-1)
            preds=torch.zeros((bs,neighbor_num,neighbor_num)).cuda()
            preds[:,0]=W/(2-W)
            preds[:,0,0]=0
        else:
            #topk_num=10
            topk, indices = torch.topk(W0, self.connect_num,dim=2)
            mask_top = torch.zeros_like(W0)
            mask_top = mask_top.scatter(-1, indices, 1)
            mask_top = ((mask_top>0)&(mask_top.permute((0,2,1))>0)).type(torch.float32)

            ##for debug###
            #print('W:',(W0[0][0][:topk_num]).tolist())
            ##############
            #W0=torch.exp(W0)
            # #change y to softmax
            # with torch.no_grad():
            #     sim=W0[:,0,:]
            #     Y[:,:,-1]=F.softmax(sim,dim=1)

            #W=(W0/(topk_num-1))*mask.expand_as(W0)
            W = torch.exp(-(2 - 2 * W0))
            W*=mask_top
            mask_top = (W0 > self.topk_num).long()  # thre
            W *= mask_top

            D= W.sum(1)
            D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
            D1      = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,neighbor_num)
            D2      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,neighbor_num,1)
            W       = D1*W*D2

            W*=mask.expand_as(W)
            #import pdb;pdb.set_trace()
            preds  = torch.matmul(torch.inverse(torch.eye(neighbor_num).unsqueeze(0).expand_as(W).cuda()-self.alpha*W+self.eps), Y)
        ##for debug###
        #print('preds:',(preds[0][0][:topk_num]).tolist())
        #import pdb;pdb.set_trace()

        if train:
            # loss=0
            # for F0 in preds:
            #     #normalize
            #     F0[0]/=F0[0].max().item()
            #     loss+=self.loss(F0[0,1:neighbor_num],gt_conf[i,1:neighbor_num])
            # loss/=len(indexes)
            # return loss
            with torch.no_grad():
                max_num,_=preds.max(2)
            preds/=max_num.unsqueeze(2).expand_as(preds)
            loss=self.loss(preds[:,0,1:neighbor_num],gt_conf[:,1:neighbor_num])
            if torch.isnan(loss):
                print('nan')
                import pdb;pdb.set_trace()
            return loss
        else:
            return preds,W0,all_neighbors

class Sub_Cluster_Level_LP(nn.Module):
    def __init__(self, alpha,topk_num=5,beta=1,method=1):
        super(Sub_Cluster_Level_LP, self).__init__()
        self.alpha=alpha
        self.eps = np.finfo(float).eps
        self.loss=torch.nn.BCEWithLogitsLoss()
        self.once_forward=1
        self.beta=beta
        self.topk_num=topk_num
        self.w_use_dist=1
        self.method=method

    def forward(self,indexes,features,neighbor_num,ori_0,ori_knn_neighbor,gt_conf_ori=None,f_s=None,train=0,sub_label=None,gt_sub_label=None,gt_label=None,debug_label=None,bias=0):
        bs=len(indexes)
        sub_sum = torch.zeros(sub_label.max()+1, 2048).float().cuda()
        sub_sum.index_add_(0, sub_label, features)
        nums = torch.zeros(sub_label.max()+1, 1).float().cuda()
        nums.index_add_(0, sub_label, torch.ones(len(sub_label),1).float().cuda())
        mask = (nums>0).float()
        sub_sum /= (mask*nums+(1-mask)).clone().expand_as(sub_sum)
        if not train:
            print('sub max:',nums.max())

        #cal Y
        Y=torch.zeros(bs,neighbor_num,neighbor_num+1).cuda()
        indices=[]
        mapping=[]
        if train:
            gt_conf=torch.zeros_like(gt_conf_ori)
        for i in range(bs):
            output, inverse_indices,cnts=torch.unique(gt_sub_label[i],return_inverse=True,return_counts=True)
            if train:
                output2,inverse_indices2=torch.unique(gt_label[i],return_inverse=True)
                #change gt
                out_gt=torch.unique(inverse_indices[inverse_indices2==inverse_indices2[0]])
                gt_conf[i,out_gt]=1
            indices.append(inverse_indices)
            mapping.append(output)
            Y[i,torch.arange(neighbor_num),inverse_indices]=1
            Y[i,:,:len(cnts)]/=cnts.unsqueeze(0)
        Y[:,0]=0
        # masks[:,0,0]=0
        # masks[:,0,1:]=1

        Y[:,1:,neighbor_num]=self.beta/(neighbor_num-1) #bias
        #cal W
        index_feat=sub_sum[sub_label[ori_knn_neighbor.view(-1)]].view(bs,-1,2048)
        #index_feat=torch.cat((f_s,index_feat[:,1:]))
        if self.method==4:
            i_feat=torch.cat((f_s.unsqueeze(1),index_feat[:,1:neighbor_num]),dim=1)
            i_feat/=torch.norm(i_feat,dim=2,keepdim=True)
            W0=i_feat.bmm(i_feat.permute(0,2,1))
            #

        else:
            if self.method!=3 and self.method!=2:
                i_feat=torch.cat((f_s.unsqueeze(1),index_feat[:,1:neighbor_num]),dim=1)
                W0=i_feat.bmm(i_feat.permute(0,2,1))
            else:
                i_feat=torch.cat((f_s.unsqueeze(1),index_feat[:,1:neighbor_num]),dim=1)
                i_feat/=torch.norm(i_feat,dim=2,keepdim=True)
                W0=i_feat.bmm(i_feat.permute(0,2,1))

            topk_num=self.topk_num
            topk, indices = torch.topk(W0, topk_num,dim=2)
            mask_top = torch.zeros_like(W0)
            mask_top = mask_top.scatter(2, indices, 1)
            mask_top = ((mask_top>0)&(mask_top.permute((0,2,1))>0)).type(torch.float32)
            #mask_top = ((mask_top>0)+mask_top.permute((0,2,1))>0).type(torch.float32) #union

            #print('sub W:',(W0[0][0][:topk_num]).tolist())
            #W0=torch.exp(W0)
            masks=(1-torch.eye(neighbor_num)).unsqueeze(0).cuda()
            if not self.w_use_dist:
                W=(W0/4)*masks.expand_as(W0)
                W*=mask_top
            else:
                if self.method==1 or self.method==3:#mask-->norm
                    W0=torch.exp(-(2-2*W0)) #dist
                    W=W0*mask_top
                    #normalize
                    D= W.sum(1)
                    D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
                    D1      = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,neighbor_num)
                    D2      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,neighbor_num,1)
                    W       = D1*W*D2
                    W*=masks.expand_as(W)
                elif self.method==2:#norm-->mask
                    W=torch.exp(-(2-2*W0)) #dist
                    #normalize
                    D= W.sum(1)
                    D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
                    D1      = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,neighbor_num)
                    D2      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,neighbor_num,1)
                    W       = D1*W*D2
                    W*=masks.expand_as(W)
                    W*=mask_top
        # D= W.sum(1)
        # D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
        # D1      = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,neighbor_num)
        # D2      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,neighbor_num,1)
        # S       = D1*W*D2
        preds  = torch.matmul(torch.inverse(torch.eye(neighbor_num).unsqueeze(0).expand_as(W).cuda()-self.alpha*W+self.eps), Y)
        merge_match_id=torch.argmax(Y[0],dim=0)
        merge_sim=W0[0][0][merge_match_id[preds[0][0]>bias]]
        print('sub merge sim:',merge_sim)
        #import pdb;pdb.set_trace()
        #print('sub preds:',(preds[0][0][:topk_num]).tolist())
        if train:
            with torch.no_grad():
                max_num,_=preds.max(2)
            preds/=max_num.unsqueeze(2).expand_as(preds)
            loss=self.loss(preds[:,0,1:neighbor_num],gt_conf[:,1:neighbor_num])
            if torch.isnan(loss):
                print('nan')
                import pdb;pdb.set_trace()
            return loss
        else:
            return preds,sub_sum,nums,mapping,indices

class Cluster_Level_LP(nn.Module):
    def __init__(self, alpha,topk_num,beta=1,method=1,point_wei=0.6,connect_num=20):
        super(Cluster_Level_LP, self).__init__()
        self.alpha=alpha
        self.eps = np.finfo(float).eps
        self.loss=torch.nn.BCEWithLogitsLoss()
        self.beta=beta
        self.only_consider_once=1
        self.topk_num=topk_num
        self.w_use_dist=1
        self.method=method
        self.point_wei=point_wei
        self.connect_num=connect_num

    def forward(self, indexes, features,neighbor_num0,ori_0,ori_knn_neighbor,gt_conf_ori=None,f_s=None,train=0,labels=None,gt_label=None,debug_label=None,bias=0,step=2,point_W=None,two_hop=0,memory=None,point_pred=None):
        bs=len(indexes)
        neighbor_num=ori_knn_neighbor.size(1)
        #masks[:,0,0]=0
        #masks[:,0,1:]=1
        clu_sum = torch.zeros(labels.max() + 1, 2048).float().cuda()
        clu_sum.index_add_(0, labels, features)
        nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(len(labels), 1).float().cuda())
        mask = (nums > 0).float()
        clu_sum /= (mask * nums + (1 - mask)).clone().expand_as(clu_sum)
        if not train:
            print('step {} max:'.format(step), nums.max())
        #Y[:,1:,neighbor_num]=self.beta/(neighbor_num-1) #bias
        #cal W
        if self.method==15: #first point topk+cluster thre
            Y = torch.zeros(bs, neighbor_num, neighbor_num).cuda()
            indices = []
            mapping = []
            # for debug
            clu_cnts = []

            tmp = []
            for i in range(bs):
                output, inverse_indices, cnts = torch.unique(gt_label[i][ori_knn_neighbor[i] > -1], return_inverse=True,
                                                             return_counts=True)
                indices.append(inverse_indices)
                mapping.append(output)
                tmp.extend(output.tolist())
                Y[i, torch.arange(len(inverse_indices)), inverse_indices] = 1
                clu_cnts.append(len(output))

            index_feat = clu_sum[labels[ori_knn_neighbor.view(-1)]].view(bs, -1, 2048)

            Y[:, 0] = 0

            # normalize
            i_feat = index_feat
            # i_feat /= torch.norm(i_feat, dim=2, keepdim=True)
            W0 = i_feat.bmm(i_feat.permute(0, 2, 1))

            mask_top = torch.ones_like(W0)  # only keep one
            indes = []
            for i in range(bs):
                filter_lab, inde = np.unique((gt_label[i, 1:][ori_knn_neighbor[i, 1:] > -1]).cpu().numpy(),
                                             return_index=True)
                inde += 1
                indes.append(inde)
                del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                mask_top[i, del_list] = 0
                mask_top[i, :, del_list] = 0
                Y[i, del_list] = 0
            masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()

            # normalize
            W = torch.exp(-(2 - 2 * W0))  # dist
            W[torch.eye(neighbor_num).unsqueeze(0).expand_as(W).long() > 0] = 1  # self-->1
            W *= mask_top  # unique

            # import pdb;pdb.set_trace()
            # topk_num = self.topk_num
            # topk, indices = torch.topk(W, topk_num, dim=2)
            # mask_top = torch.zeros_like(W)
            # mask_top = mask_top.scatter(2, indices, 1)
            # mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
            # connect method####
            topk, indices = torch.topk(point_W, self.connect_num, dim=-1)
            mask_top = torch.zeros_like(W)
            mask_top = mask_top.scatter(2, indices, 1)
            mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
            W *= mask_top

            # mask_top = (W0 > self.topk_num).long()  # thre
            W *= ((W0 > self.topk_num).long())

            # normalize
            D = W.sum(1)
            D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, neighbor_num)
            D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, neighbor_num, 1)
            W = D1 * W * D2
            W *= masks.expand_as(W)
        if self.method==14:
            Y = torch.zeros(bs, neighbor_num, neighbor_num).cuda()
            indices = []
            mapping = []
            # for debug
            clu_cnts = []

            tmp = []
            for i in range(bs):
                output, inverse_indices, cnts = torch.unique(gt_label[i][ori_knn_neighbor[i] > -1], return_inverse=True,
                                                             return_counts=True)
                indices.append(inverse_indices)
                mapping.append(output)
                tmp.extend(output.tolist())
                Y[i, torch.arange(len(inverse_indices)), inverse_indices] = 1
                clu_cnts.append(len(output))

            index_feat = clu_sum[labels[ori_knn_neighbor.view(-1)]].view(bs, -1, 2048)

            Y[:, 0] = 0

            # normalize
            i_feat = index_feat
            # i_feat /= torch.norm(i_feat, dim=2, keepdim=True)
            W0 = i_feat.bmm(i_feat.permute(0, 2, 1))

            mask_top = torch.ones_like(W0)  # only keep one
            indes = []
            for i in range(bs):
                filter_lab, inde = np.unique((gt_label[i, 1:][ori_knn_neighbor[i, 1:] > -1]).cpu().numpy(),
                                             return_index=True)
                inde += 1
                indes.append(inde)
                del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                mask_top[i, del_list] = 0
                mask_top[i, :, del_list] = 0
                Y[i, del_list] = 0
            masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()

            # normalize
            W = torch.exp(-(2 - 2 * W0))  # dist
            W[torch.eye(neighbor_num).unsqueeze(0).expand_as(W).long() > 0] = 1  # self-->1
            W *= mask_top  # unique

            # import pdb;pdb.set_trace()
            # topk_num = self.topk_num
            # topk, indices = torch.topk(W, topk_num, dim=2)
            # mask_top = torch.zeros_like(W)
            # mask_top = mask_top.scatter(2, indices, 1)
            # mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
            #connect method####
            topk, indices = torch.topk(point_W, self.connect_num, dim=-1)
            mask_top = torch.zeros_like(W)
            mask_top = mask_top.scatter(2, indices, 1)
            mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
            W *= mask_top

            # mask_top = (W0 > self.topk_num).long()  # thre
            W *= ((point_W > self.topk_num).long())

            # normalize
            D = W.sum(1)
            D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, neighbor_num)
            D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, neighbor_num, 1)
            W = D1 * W * D2
            W *= masks.expand_as(W)
        if self.method==13:#jaccard debug
            indices = []
            mapping = []
            indes = []
            for i in range(bs):
                indices.append(torch.arange(neighbor_num))
                mapping.append(gt_label[i])
            preds=(point_pred>=0.4).int()
            preds[:,0]*=(torch.sum(preds[:,0],dim=-1,keepdim=True)>1)
            return preds, clu_sum, nums, mapping, indices, indes
        if self.method==12: #change weights
            point_wei = self.point_wei
            Y = torch.zeros(bs, neighbor_num, neighbor_num).cuda()
            indices = []
            mapping = []
            # for debug
            clu_cnts = []

            tmp = []
            for i in range(bs):
                output, inverse_indices, cnts = torch.unique(gt_label[i][ori_knn_neighbor[i] > -1], return_inverse=True,
                                                             return_counts=True)
                indices.append(inverse_indices)
                mapping.append(output)
                tmp.extend(output.tolist())
                Y[i, torch.arange(len(inverse_indices)), inverse_indices] = 1
                clu_cnts.append(len(output))

            #add weights
            index_feat = point_wei*features[ori_knn_neighbor.view(-1)].view(bs,-1,2048)+(1-point_wei)*clu_sum[labels[ori_knn_neighbor.view(-1)]].view(bs, -1, 2048)

            Y[:, 0] = 0

            # normalize
            i_feat = index_feat
            # i_feat /= torch.norm(i_feat, dim=2, keepdim=True)
            W0 = i_feat.bmm(i_feat.permute(0, 2, 1))

            mask_top = torch.ones_like(W0)  # only keep one
            indes = []
            for i in range(bs):
                filter_lab, inde = np.unique((gt_label[i, 1:][ori_knn_neighbor[i, 1:] > -1]).cpu().numpy(),
                                             return_index=True)
                inde += 1
                indes.append(inde)
                del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                mask_top[i, del_list] = 0
                mask_top[i, :, del_list] = 0
                Y[i, del_list] = 0
            masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()

            # normalize
            W = torch.exp(-(2 - 2 * W0))  # dist
            W[torch.eye(neighbor_num).unsqueeze(0).expand_as(W).long() > 0] = 1  # self-->1
            W *= mask_top  # unique

            # import pdb;pdb.set_trace()
            # topk_num = self.topk_num
            # topk, indices = torch.topk(W, topk_num, dim=2)
            # mask_top = torch.zeros_like(W)
            # mask_top = mask_top.scatter(2, indices, 1)
            # mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
            topk, indices = torch.topk(W0, self.connect_num, dim=2)
            mask_top = torch.zeros_like(W)
            mask_top = mask_top.scatter(2, indices, 1)
            mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
            W *= mask_top

            #mask_top = (W0 > self.topk_num).long()  # thre
            W *= ((W0 > self.topk_num).long())

            # normalize
            D = W.sum(1)
            D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, neighbor_num)
            D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, neighbor_num, 1)
            W = D1 * W * D2
            W *= masks.expand_as(W)
        if self.method==11: #change weights
            point_wei = self.point_wei
            if two_hop:
                neighbor_num=400
                W0=torch.zeros((bs,neighbor_num,neighbor_num)).cuda()
                Y=torch.zeros((bs,neighbor_num,neighbor_num)).cuda()
                indices = []
                mapping = []
                # for debug
                clu_cnts = []
                for i in range(bs):
                    output, inverse_indices, cnts = torch.unique(gt_label[i][ori_knn_neighbor[i]>-1], return_inverse=True, return_counts=True)
                    indices.append(inverse_indices)
                    mapping.append(output)
                    Y[i, torch.arange(len(inverse_indices)), inverse_indices] = 1
                    clu_cnts.append(len(output))
                    ind_feat=point_wei*features[ori_knn_neighbor[i][ori_knn_neighbor[i]>-1]]+(1-point_wei)*clu_sum[labels[ori_knn_neighbor[i][ori_knn_neighbor[i]>-1]]]
                    W0[i,:len(ind_feat),:len(ind_feat)]=ind_feat.mm(ind_feat.t())
                Y[:,0]=0

                mask_top = torch.ones_like(W0)  # only keep one
                indes = []
                for i in range(bs):
                    filter_lab, inde = np.unique(gt_label[i, 1:][ori_knn_neighbor[i,1:]>-1].cpu().numpy(), return_index=True)
                    inde += 1
                    indes.append(inde)
                    del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                    mask_top[i, del_list] = 0
                    mask_top[i, :, del_list] = 0
                    Y[i, del_list] = 0
                masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()
                # normalize
                W = torch.exp(-(2 - 2 * W0))  # dist
                W[torch.eye(neighbor_num).unsqueeze(0).expand_as(W).long() > 0] = 1  # self-->1
                W *= mask_top  # unique

                #bug-->cluster num
                # topk, indices = torch.topk(W, 15, dim=2)
                # mask_top = torch.zeros_like(W)
                # mask_top = mask_top.scatter(2, indices, 1)
                # mask_top = ((mask_top > 0) | (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
                # W *= mask_top
                #0217
                # point_feat=features[ori_knn_neighbor.view(-1)].view(bs,-1,2048)
                # W_point=point_feat.bmm(point_feat.permute((0,2,1)))
                #0218
                topk, indices = torch.topk(W0, 20, dim=2)
                mask_top = torch.zeros_like(W)
                mask_top = mask_top.scatter(2, indices, 1)
                mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
                W *= mask_top
                #0218-->move before
                mask_top = (W0 > self.topk_num).long()  # thre
                W *= mask_top

                # normalize
                D = W.sum(1)
                D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, neighbor_num)
                D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, neighbor_num, 1)
                W = D1 * W * D2
                W *= masks.expand_as(W)
            else:
                Y = torch.zeros(bs, neighbor_num, neighbor_num).cuda()
                indices = []
                mapping = []
                # for debug
                clu_cnts = []

                tmp = []
                for i in range(bs):
                    output, inverse_indices, cnts = torch.unique(gt_label[i][ori_knn_neighbor[i] > -1], return_inverse=True,
                                                                 return_counts=True)
                    indices.append(inverse_indices)
                    mapping.append(output)
                    tmp.extend(output.tolist())
                    Y[i, torch.arange(len(inverse_indices)), inverse_indices] = 1
                    clu_cnts.append(len(output))

                #add weights
                index_feat = point_wei*features[ori_knn_neighbor.view(-1)].view(bs,-1,2048)+(1-point_wei)*clu_sum[labels[ori_knn_neighbor.view(-1)]].view(bs, -1, 2048)

                Y[:, 0] = 0

                # normalize
                i_feat = index_feat
                # i_feat /= torch.norm(i_feat, dim=2, keepdim=True)
                W0 = i_feat.bmm(i_feat.permute(0, 2, 1))

                mask_top = torch.ones_like(W0)  # only keep one
                indes = []
                for i in range(bs):
                    filter_lab, inde = np.unique((gt_label[i, 1:][ori_knn_neighbor[i, 1:] > -1]).cpu().numpy(),
                                                 return_index=True)
                    inde += 1
                    indes.append(inde)
                    del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                    mask_top[i, del_list] = 0
                    mask_top[i, :, del_list] = 0
                    Y[i, del_list] = 0
                masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()

                # normalize
                W = torch.exp(-(2 - 2 * W0))  # dist
                W[torch.eye(neighbor_num).unsqueeze(0).expand_as(W).long() > 0] = 1  # self-->1
                W *= mask_top  # unique

                # import pdb;pdb.set_trace()
                # topk_num = self.topk_num
                # topk, indices = torch.topk(W, topk_num, dim=2)
                # mask_top = torch.zeros_like(W)
                # mask_top = mask_top.scatter(2, indices, 1)
                # mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
                mask_top = (W0 > self.topk_num).long()  # thre
                W *= mask_top

                # normalize
                D = W.sum(1)
                D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, neighbor_num)
                D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, neighbor_num, 1)
                W = D1 * W * D2
                W *= masks.expand_as(W)

        if self.method==10:#pick 3 for each cluster
            candi_num=3

            Y = torch.zeros(bs, neighbor_num, neighbor_num).cuda()
            indices = []
            mapping = []
            # for debug
            clu_cnts = []
            #all_output,all_inverse,all_cnts=torch.unique(labels.cpu().clone(),return_inverse=True,return_counts=True)
            #all_output_3=set(all_output[all_cnts>candi_num].tolist())
            all_output_3=set(torch.arange(nums.size(0))[nums.cpu().view(-1)>candi_num].tolist())

            tmp=[]
            for i in range(bs):
                output, inverse_indices, cnts = torch.unique(gt_label[i][ori_knn_neighbor[i] > -1], return_inverse=True,
                                                             return_counts=True)
                indices.append(inverse_indices)
                mapping.append(output)
                tmp.extend(output.tolist())
                Y[i, torch.arange(len(inverse_indices)), inverse_indices] = 1
                clu_cnts.append(len(output))

            # only pick 3
            #import pdb;pdb.set_trace()
            filter_out = list(set(tmp) & all_output_3)
            if len(filter_out)>0:
                print('len(filter_out):',len(filter_out))
            for cc in filter_out:
                # re cal the index feat
                cc_feat=features[labels==cc]
                cc_sim=(cc_feat.mm(clu_sum[cc].unsqueeze(1))).view(-1)
                topk, indices = torch.topk(cc_sim, candi_num)
                clu_sum[cc]=torch.mean(cc_feat[indices],dim=0)

            index_feat = clu_sum[labels[ori_knn_neighbor.view(-1)]].view(bs, -1, 2048)

            Y[:, 0] = 0

            # normalize
            i_feat = index_feat
            # i_feat /= torch.norm(i_feat, dim=2, keepdim=True)
            W0 = i_feat.bmm(i_feat.permute(0, 2, 1))

            mask_top = torch.ones_like(W0)  # only keep one
            indes=[]
            for i in range(bs):
                filter_lab, inde = np.unique((gt_label[i, 1:][ori_knn_neighbor[i, 1:] > -1]).cpu().numpy(),
                                             return_index=True)
                inde += 1
                indes.append(inde)
                del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                mask_top[i, del_list] = 0
                mask_top[i, :, del_list] = 0
                Y[i, del_list] = 0
            masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()

            # normalize
            W = torch.exp(-(2 - 2 * W0))  # dist
            W[torch.eye(neighbor_num).unsqueeze(0).expand_as(W).long() > 0] = 1  # self-->1
            W *= mask_top  # unique

            # import pdb;pdb.set_trace()
            # topk_num = self.topk_num
            # topk, indices = torch.topk(W, topk_num, dim=2)
            # mask_top = torch.zeros_like(W)
            # mask_top = mask_top.scatter(2, indices, 1)
            # mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
            mask_top = (W0 > self.topk_num).long()  # thre
            W *= mask_top

            # normalize
            D = W.sum(1)
            D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, neighbor_num)
            D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, neighbor_num, 1)
            W = D1 * W * D2
            W *= masks.expand_as(W)

        if self.method==7:
            index_feat = features[ori_knn_neighbor.view(-1)].view(bs,-1,2048)#clu_sum[labels[ori_knn_neighbor.view(-1)]].view(bs, -1, 2048)
            # cal Y
            Y = torch.zeros(bs, neighbor_num, neighbor_num + 1).cuda()

            indices = []
            mapping = []
            for i in range(bs):
                output, inverse_indices, cnts = torch.unique(gt_label[i], return_inverse=True, return_counts=True)
                indices.append(inverse_indices)
                mapping.append(output)
                Y[i, torch.arange(neighbor_num), inverse_indices] = 1
            Y[:, 0] = 0
            i_feat = torch.cat((f_s.unsqueeze(1), index_feat[:, 1:neighbor_num]), dim=1)
            #i_feat /= torch.norm(i_feat, dim=2, keepdim=True)
            W0 = i_feat.bmm(i_feat.permute(0, 2, 1))

            topk, indices = torch.topk(W0, self.topk_num, dim=2)  # use point level topk
            mask_top = torch.zeros_like(W0)
            mask_top = mask_top.scatter(2, indices, 1)
            mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)

            for i in range(bs):  # unique
                filter_lab, inde = np.unique(gt_label[i, 1:].cpu().numpy(), return_index=True)
                inde += 1
                del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                mask_top[i, del_list] = 0
                mask_top[i, :, del_list] = 0
                Y[i, del_list] = 0

            W = torch.exp(-(2 - 2 * W0))  # dist
            W *= mask_top
            # normalize
            D = W.sum(1)
            D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, neighbor_num)
            D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, neighbor_num, 1)
            W = D1 * W * D2
            masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()  # wo self
            W *= masks.expand_as(W)

        if self.method==8:
            index_feat = clu_sum[labels[ori_knn_neighbor.view(-1)]].view(bs, -1, 2048)
            # cal Y
            Y = torch.zeros(bs, neighbor_num, neighbor_num + 1).cuda()

            indices = []
            mapping = []
            for i in range(bs):
                output, inverse_indices, cnts = torch.unique(gt_label[i], return_inverse=True, return_counts=True)
                indices.append(inverse_indices)
                mapping.append(output)
                Y[i, torch.arange(neighbor_num), inverse_indices] = 1
            Y[:, 0] = 0
            i_feat = torch.cat((f_s.unsqueeze(1), index_feat[:, 1:neighbor_num]), dim=1)
            i_feat /= torch.norm(i_feat, dim=2, keepdim=True)
            W0 = i_feat.bmm(i_feat.permute(0, 2, 1))

            topk, indices = torch.topk(point_W, self.topk_num, dim=2)  # use point level topk
            mask_top = torch.zeros_like(W0)
            mask_top = mask_top.scatter(2, indices, 1)
            mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)

            for i in range(bs):  # unique
                filter_lab, inde = np.unique(gt_label[i, 1:].cpu().numpy(), return_index=True)
                inde += 1
                del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                mask_top[i, del_list] = 0
                mask_top[i, :, del_list] = 0
                Y[i, del_list] = 0

            W = torch.exp(-(2 - 2 * W0))  # dist
            W *= mask_top
            masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()  # wo self
            W *= masks.expand_as(W)
            # normalize
            D = W.sum(1)
            D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, neighbor_num)
            D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, neighbor_num, 1)
            W = D1 * W * D2
        if self.method==6:
            index_feat = clu_sum[labels[ori_knn_neighbor.view(-1)]].view(bs, -1, 2048)
            # cal Y
            Y = torch.zeros(bs, neighbor_num, neighbor_num + 1).cuda()

            indices = []
            mapping = []
            for i in range(bs):
                output, inverse_indices, cnts = torch.unique(gt_label[i], return_inverse=True, return_counts=True)
                indices.append(inverse_indices)
                mapping.append(output)
                Y[i, torch.arange(neighbor_num), inverse_indices] = 1
            Y[:, 0] = 0
            i_feat = torch.cat((f_s.unsqueeze(1), index_feat[:, 1:neighbor_num]), dim=1)
            i_feat /= torch.norm(i_feat, dim=2, keepdim=True)
            W0 = i_feat.bmm(i_feat.permute(0, 2, 1))

            topk, indices = torch.topk(point_W, int(self.topk_num), dim=2)#use point level topk
            mask_top = torch.zeros_like(W0)
            mask_top = mask_top.scatter(2, indices, 1)
            mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)

            for i in range(bs):#unique
                filter_lab, inde = np.unique(gt_label[i, 1:].cpu().numpy(), return_index=True)
                inde += 1
                del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                mask_top[i, del_list] = 0
                mask_top[i, :, del_list] = 0
                Y[i, del_list] = 0

            W = torch.exp(-(2 - 2 * W0))  # dist
            W *= mask_top
            # normalize
            D = W.sum(1)
            D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, neighbor_num)
            D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, neighbor_num, 1)
            W = D1 * W * D2
            masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()#wo self
            W *= masks.expand_as(W)

        if self.method==5:#0128
            if two_hop:
                neighbor_num=99
                W0=torch.zeros((bs,neighbor_num,neighbor_num)).cuda()
                Y=torch.zeros((bs,neighbor_num,neighbor_num)).cuda()
                indices = []
                mapping = []
                # for debug
                clu_cnts = []
                for i in range(bs):
                    output, inverse_indices, cnts = torch.unique(gt_label[i][ori_knn_neighbor[i]>-1], return_inverse=True, return_counts=True)
                    indices.append(inverse_indices)
                    mapping.append(output)
                    Y[i, torch.arange(len(inverse_indices)), inverse_indices] = 1
                    clu_cnts.append(len(output))
                    ind_feat=clu_sum[labels[ori_knn_neighbor[i][ori_knn_neighbor[i]>-1]]]
                    W0[i,:len(ind_feat),:len(ind_feat)]=ind_feat.mm(ind_feat.t())
                Y[:,0]=0

                mask_top = torch.ones_like(W0)  # only keep one
                for i in range(bs):
                    filter_lab, inde = np.unique(gt_label[i, 1:][ori_knn_neighbor[i,1:]>-1].cpu().numpy(), return_index=True)
                    inde += 1
                    del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                    mask_top[i, del_list] = 0
                    mask_top[i, :, del_list] = 0
                    Y[i, del_list] = 0
                masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()

            else:
                index_feat = clu_sum[labels[ori_knn_neighbor.view(-1)]].view(bs, -1, 2048)
                # cal Y
                Y = torch.zeros(bs, neighbor_num, neighbor_num).cuda()
                indices = []
                mapping = []
                #for debug
                clu_cnts=[]
                for i in range(bs):
                    output, inverse_indices, cnts = torch.unique(gt_label[i][ori_knn_neighbor[i]>-1], return_inverse=True, return_counts=True)
                    indices.append(inverse_indices)
                    mapping.append(output)
                    Y[i, torch.arange(len(inverse_indices)), inverse_indices] = 1
                    clu_cnts.append(len(output))
                print('clu cnts:',clu_cnts)
                Y[:, 0] = 0

                # normalize
                i_feat = index_feat
                #i_feat /= torch.norm(i_feat, dim=2, keepdim=True)
                W0 = i_feat.bmm(i_feat.permute(0, 2, 1))

                mask_top = torch.ones_like(W0) #only keep one
                for i in range(bs):
                    filter_lab, inde = np.unique((gt_label[i, 1:][ori_knn_neighbor[i,1:]>-1]).cpu().numpy(), return_index=True)
                    inde += 1
                    del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                    mask_top[i, del_list] = 0
                    mask_top[i, :, del_list] = 0
                    Y[i, del_list] = 0
                masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()

            # normalize
            W = torch.exp(-(2 - 2 * W0))  # dist
            W[torch.eye(neighbor_num).unsqueeze(0).expand_as(W).long()>0] = 1  # self-->1
            W *= mask_top #unique

            #import pdb;pdb.set_trace()
            # topk_num = self.topk_num
            # topk, indices = torch.topk(W, topk_num, dim=2)
            # mask_top = torch.zeros_like(W)
            # mask_top = mask_top.scatter(2, indices, 1)
            # mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
            mask_top=(W0>self.topk_num).long()#thre
            W*=mask_top

            # normalize
            D = W.sum(1)
            D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, neighbor_num)
            D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, neighbor_num, 1)
            W = D1 * W * D2
            W *= masks.expand_as(W)
        #method 4-->fix bug (based on method 3)
        if self.method==9:#0210
            if two_hop:
                neighbor_num=100
                W0=torch.zeros((bs,neighbor_num,neighbor_num)).cuda()
                Y=torch.zeros((bs,neighbor_num,neighbor_num)).cuda()
                indices = []
                mapping = []
                # for debug
                clu_cnts = []
                for i in range(bs):
                    output, inverse_indices, cnts = torch.unique(gt_label[i][ori_knn_neighbor[i]>-1], return_inverse=True, return_counts=True)
                    indices.append(inverse_indices)
                    mapping.append(output)
                    Y[i, torch.arange(len(inverse_indices)), inverse_indices] = 1
                    clu_cnts.append(len(output))
                    ind_feat=clu_sum[labels[ori_knn_neighbor[i][ori_knn_neighbor[i]>-1]]]
                    W0[i,:len(ind_feat),:len(ind_feat)]=ind_feat.mm(ind_feat.t())
                Y[:,0]=0

                mask_top = torch.ones_like(W0)  # only keep one
                for i in range(bs):
                    filter_lab, inde = np.unique(gt_label[i, 1:][ori_knn_neighbor[i,1:]>-1].cpu().numpy(), return_index=True)
                    inde += 1
                    del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                    mask_top[i, del_list] = 0
                    mask_top[i, :, del_list] = 0
                    Y[i, del_list] = 0
                masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()

            else:
                index_feat = clu_sum[labels[ori_knn_neighbor.view(-1)]].view(bs, -1, 2048)
                # cal Y
                Y = torch.zeros(bs, neighbor_num, neighbor_num).cuda()
                indices = []
                mapping = []
                #for debug
                clu_cnts=[]
                for i in range(bs):
                    output, inverse_indices, cnts = torch.unique(gt_label[i], return_inverse=True, return_counts=True)
                    indices.append(inverse_indices)
                    mapping.append(output)
                    Y[i, torch.arange(neighbor_num), inverse_indices] = 1
                    clu_cnts.append(len(output))
                print('clu cnts:',clu_cnts)
                Y[:, 0] = 0

                # normalize
                i_feat = index_feat
                #i_feat /= torch.norm(i_feat, dim=2, keepdim=True)
                W0 = i_feat.bmm(i_feat.permute(0, 2, 1))

                mask_top = torch.ones_like(W0) #only keep one
                for i in range(bs):
                    filter_lab, inde = np.unique(gt_label[i, 1:].cpu().numpy(), return_index=True)
                    inde += 1
                    del_list = list(set(np.arange(1, neighbor_num).tolist()) - set(inde))
                    mask_top[i, del_list] = 0
                    mask_top[i, :, del_list] = 0
                    Y[i, del_list] = 0
                masks = (1 - torch.eye(neighbor_num)).unsqueeze(0).cuda()

            # normalize
            W = torch.exp(W0)  # dist
            W[torch.eye(neighbor_num).unsqueeze(0).expand_as(W).long()>0] = torch.exp(torch.tensor(1).float())  # self-->1
            W *= mask_top #unique

            #import pdb;pdb.set_trace()
            # topk_num = self.topk_num
            # topk, indices = torch.topk(W, topk_num, dim=2)
            # mask_top = torch.zeros_like(W)
            # mask_top = mask_top.scatter(2, indices, 1)
            # mask_top = ((mask_top > 0) & (mask_top.permute((0, 2, 1)) > 0)).type(torch.float32)
            mask_top=(W0>self.topk_num).long()#thre
            W*=mask_top

            # normalize
            #import pdb;pdb.set_trace()
            D = W.sum(1)
            W/=(D.unsqueeze(2).expand_as(W)+self.eps)
            W *= masks.expand_as(W)
        if self.method==4:#iterative
            #norm
            #i_feat=torch.cat((f_s.unsqueeze(1),index_feat[:,1:neighbor_num]),dim=1)
            # print('debug')
            # import pdb;pdb.set_trace()
            with torch.no_grad():
                i_sim=f_s.mm(clu_sum.t())
            #topk cluster
            topk, indices = torch.topk(i_sim, neighbor_num,dim=1)
            i_feat=clu_sum[indices.view(-1)].view(bs,neighbor_num,2048)
            i_feat/=torch.norm(i_feat,dim=2,keepdim=True)
            W0=i_feat.bmm(i_feat.permute(0,2,1))

            #cal Y
            Y=torch.eye(neighbor_num).unsqueeze(0).expand_as(W0).cuda()
            Y[:,0]=0
            mapping=indices.clone()

            topk, indices = torch.topk(W0, self.topk_num,dim=2)
            mask_top = torch.zeros_like(W0)
            mask_top = mask_top.scatter(2, indices, 1)
            mask_top = ((mask_top>0)&(mask_top.permute((0,2,1))>0)).type(torch.float32)

            W=torch.exp(-(2-2*W0)) #dist
            W*=mask_top
            #normalize
            D= W.sum(1)
            D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
            D1      = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,neighbor_num)
            D2      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,neighbor_num,1)
            W      = D1*W*D2

            masks=(1-torch.eye(neighbor_num)).unsqueeze(0).cuda()
            W*=masks.expand_as(W)
        elif self.method<4:
            index_feat=clu_sum[labels[ori_knn_neighbor.view(-1)]].view(bs,-1,2048)
            #cal Y
            Y=torch.zeros(bs,neighbor_num,neighbor_num+1).cuda()
            masks=torch.ones((bs,neighbor_num,neighbor_num)).cuda()
            indices=[]
            mapping=[]
            if train:
                gt_conf=torch.zeros_like(gt_conf_ori)
            for i in range(bs):
                output, inverse_indices,cnts=torch.unique(gt_label[i],return_inverse=True,return_counts=True)
                if train:
                    #change gt
                    gt_conf[i,inverse_indices[0]]=1
                indices.append(inverse_indices)
                mapping.append(output)
                #masks[i,torch.arange(neighbor_num),inverse_indices]=0
                Y[i,torch.arange(neighbor_num),inverse_indices]=1
                if not self.only_consider_once:
                    Y[i,:,:len(cnts)]/=cnts.unsqueeze(0)
            Y[:,0]=0
            #index_feat=torch.cat((f_s,index_feat[:,1:]))
            if self.method!=3 and self.method!=2:
                i_feat=torch.cat((f_s.unsqueeze(1),index_feat[:,1:neighbor_num]),dim=1)
                W0=i_feat.bmm(i_feat.permute(0,2,1))
            else:
                #normalize
                i_feat=torch.cat((f_s.unsqueeze(1),index_feat[:,1:neighbor_num]),dim=1)
                i_feat/=torch.norm(i_feat,dim=2,keepdim=True)
                W0=i_feat.bmm(i_feat.permute(0,2,1))

            topk_num=self.topk_num
            topk, indices = torch.topk(W0, topk_num,dim=2)
            mask_top = torch.zeros_like(W0)
            mask_top = mask_top.scatter(2, indices, 1)
            mask_top = ((mask_top>0)&(mask_top.permute((0,2,1))>0)).type(torch.float32)
            #mask_union_top=((mask_top>0)+mask_top.permute((0,2,1))>0).type(torch.float32)
            #mask_top = ((mask_top>0)+mask_top.permute((0,2,1))>0).type(torch.float32) #union

            if self.only_consider_once:
                for i in range(bs):
                    filter_lab,inde=np.unique(gt_label[i,1:].cpu().numpy(),return_index=True)
                    inde+=1
                    del_list=list(set(np.arange(1,neighbor_num).tolist())-set(inde))
                    mask_top[i,del_list]=0
                    mask_top[i,:,del_list]=0
                    Y[i,del_list]=0
            # if not train:
            #     import pdb;pdb.set_trace()
            print('clu W:',(W0[0][0][:10]).tolist())
            #W0=torch.exp(W0)
            masks=(1-torch.eye(neighbor_num)).unsqueeze(0).cuda()
            if not self.w_use_dist:
                W=(W0/4)*masks.expand_as(W0)
                W*=mask_top
            else:
                if self.method==1:
                    W=torch.exp(-(2-2*W0)) #dist
                    W*=mask_top
                    #normalize
                    D= W.sum(1)
                    D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
                    D1      = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,neighbor_num)
                    D2      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,neighbor_num,1)
                    W      = D1*W*D2
                    W*=masks.expand_as(W)
                elif self.method==2:
                    W=torch.exp(-(2-2*W0)) #dist
                    #normalize
                    D= W.sum(1)
                    D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
                    D1      = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,neighbor_num)
                    D2      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,neighbor_num,1)
                    W      = D1*W*D2
                    W*=masks.expand_as(W)
                    W*=mask_top
                elif self.method==3:
                    #normalize
                    W=torch.exp(-(2-2*W0)) #dist
                    W*=mask_top
                    #normalize
                    D= W.sum(1)
                    D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
                    D1      = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,neighbor_num)
                    D2      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,neighbor_num,1)
                    W      = D1*W*D2
                    W*=masks.expand_as(W)
        #change y to softmax
        # with torch.no_grad():
        #     sim=W0[:,0,:]
        #     Y[:,:,-1]=F.softmax(sim,dim=1)

        # D= W.sum(1)
        # D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
        # D1      = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,neighbor_num)
        # D2      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,neighbor_num,1)
        # S       = D1*W*D2
        preds  = torch.matmul(torch.inverse(torch.eye(neighbor_num).unsqueeze(0).expand_as(W).cuda()-self.alpha*W+self.eps), Y)
        # if self.method==4:
        #     preds[:,0,0]=bias+1#add self
        #import pdb;pdb.set_trace()
        #for debug
        # if self.method==4:
        #     Y[0,0,0]=1
        # merge_match_id=torch.argmax(Y[0],dim=0)
        # merge_sim=W0[0][0][merge_match_id[preds[0][0]>bias]]
        # print('step {} merge sim:'.format(step),merge_sim)
        #import pdb;pdb.set_trace()

        # if train:
        #     with torch.no_grad():
        #         max_num,_=preds.max(2)
        #     preds/=max_num.unsqueeze(2).expand_as(preds)
        #     loss=self.loss(preds[:,0,1:neighbor_num],gt_conf[:,1:neighbor_num])
        #     if torch.isnan(loss):
        #         print('nan')
        #         import pdb;pdb.set_trace()
        #     return loss
        # else:
        del W,Y
        return preds,clu_sum,nums,mapping,indices,indes


class Split_GCN(nn.Module):
    def __init__(self, feature_dim, nhid, feature_size,source_classes,nclass, momentum=0.2,dropout=0,cal_num=30):
        super(Split_GCN, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)

        self.nclass = 2
        self.classifier = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(nhid),
                                        nn.Linear(nhid, self.nclass))
        self.loss=torch.nn.CrossEntropyLoss().cuda()
        self.source_classes=source_classes
    def forward(self,indexes,features,labels,train,sub_label=0,outliers_label=None,ori_knn_neighbor=None,gt=None,sub_labels=None):
        index_feat=features[indexes]
        all_idxs=torch.arange(len(labels)).cuda()
        loss=0
        if not train: # inference
            for n,idx in enumerate(indexes):
                split_idxs=all_idxs[labels==labels[idx]]
                if len(split_idxs)==1:
                    continue
                split_feat=features[labels==labels[idx]]
                split_sim=features[idx].unsqueeze(0).mm(split_feat.t())
                anchor_idx=split_idxs[torch.argmin(split_sim)]

                X=torch.cat([split_feat.unsqueeze(0),split_feat.unsqueeze(0)],dim=0)
                A=X.bmm(X.permute(0,2,1))
                A=F.softmax(A,dim=2)
                X[0]-=features[idx]
                X[1]-=features[anchor_idx]

                X=self.conv1(X, A)
                dout=X.size(-1)
                x_0=X.view(-1,dout)
                all_pred=F.softmax(self.classifier(x_0),dim=1)
                all_pred=all_pred.view(2,-1,2)
                all_pred=torch.argmin(all_pred[:,:,1],dim=0)
                if sub_label:
                    labs=torch.tensor([idx.item(),anchor_idx.item()]).cuda()
                    labels[split_idxs]=labs[all_pred]
                else:
                    labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                    sub_idx=sub_labels[split_idxs]
                    sub_lab,cnts=torch.unique(sub_idx,return_counts=True)
                    for sub_i,sub in enumerate(sub_lab):
                        if torch.sum(all_pred[sub_idx==sub])>cnts[sub_i]/2:
                            labels[sub_labels==sub]=labs[1]
                        else:
                            labels[sub_labels==sub]=labs[0]
                labels[split_idxs]=labs[all_pred]
        else:
            X=features[ori_knn_neighbor.view(-1)].view(len(indexes),-1,2048)
            A=X.bmm(X.permute(0,2,1))
            A=F.softmax(A,dim=2)
            X-=X[:,0].view(-1,1,2048)

            X=self.conv1(X, A)
            dout=X.size(-1)
            x_0=X.view(-1,dout)
            all_pred=self.classifier(x_0)
            gt=gt.view(-1)
            loss=self.loss(all_pred,gt)
            return loss

class Split_LP(nn.Module):
    def __init__(self,alpha,split_num,anchor_thre,connect_num=20):
        super(Split_LP, self).__init__()
        self.alpha=alpha
        self.eps = np.finfo(float).eps
        self.method=9
        self.split_num=split_num
        self.anchor_thre=anchor_thre
        self.connect_num=connect_num
    def forward(self,indexes,features,labels,sub_level=0,sub_labels=None,outliers_label=None,ori_knn_neighbor=None,memory=None,two_hop=0,point_pred=None,point_W=None):
        all_idxs=torch.arange(len(labels)).cuda()
        split_nums=[]
        if self.method==0:
            if sub_level:
                for n,idx in enumerate(indexes):
                    split_idxs=all_idxs[labels==labels[idx]]
                    if len(split_idxs)<=4:
                        continue
                    split_feat=features[labels==labels[idx]]
                    split_sim=features[idx].unsqueeze(0).mm(split_feat.t())
                    anchor_idx=split_idxs[torch.argmin(split_sim)]
                    split_sim_2=features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                    anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]

                    Y=torch.zeros((len(split_idxs),2)).cuda()
                    i_0,i_1=torch.argmax(split_sim_2),torch.argmin(split_sim_2)
                    Y[i_0,0]=1
                    Y[i_1,1]=1

                    W=torch.exp(split_feat.mm(split_feat.t()))
                    mask=torch.ones_like(W)
                    mask[i_0,i_0]=0
                    mask[i_1,i_1]=0
                    W*=mask
                    D       = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
                    D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,len(split_idxs))
                    D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(len(split_idxs),1)
                    S       = D1*W*D2

                    pred  = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda()-self.alpha*S+self.eps), Y)
                    pred=torch.argmax(pred,dim=1)
                    lab=torch.tensor([labels[anchor_idx].item(),labels[anchor_idx_2].item()]).cuda()
                    labels[split_idxs]=lab[pred]
                    #for debug
                    split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
            else:
                for n,idx in enumerate(indexes):
                    batch_idx=all_idxs[labels==labels[idx]]
                    batch_sub_label=sub_labels[batch_idx]
                    split_idxs=list(set((batch_sub_label).tolist())) #sub label
                    if len(split_idxs)<3:
                        continue
                    split_feat=features[split_idxs]
                    split_sim=features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
                    anchor_idx=split_idxs[torch.argmin(split_sim)]
                    split_sim_2=features[sub_labels[anchor_idx]].unsqueeze(0).mm(split_feat.t())
                    anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]

                    Y=torch.zeros((len(split_idxs),2)).cuda()
                    i_0,i_1=torch.argmax(split_sim_2),torch.argmin(split_sim_2)
                    Y[i_0,0]=1
                    Y[i_1,1]=1

                    W=torch.exp(split_feat.mm(split_feat.t()))
                    mask=torch.ones_like(W)
                    mask[i_0,i_0]=0
                    mask[i_1,i_1]=0
                    W*=mask
                    D       = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
                    D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,len(split_idxs))
                    D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(len(split_idxs),1)
                    S       = D1*W*D2

                    pred  = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda()-self.alpha*S+self.eps), Y)
                    pred=torch.argmax(pred,dim=1)

                    labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                    for sub,pre in zip(split_idxs,pred):
                        labels[batch_idx[batch_sub_label==sub]]=labs[pre]
                    split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
        elif self.method==1:
            split_num=0
            if sub_level:
                print_cnts=0
                for n,idx in enumerate(indexes):
                    split_idxs=all_idxs[labels==labels[idx]]
                    if len(split_idxs)<=self.split_num:
                        continue
                    split_feat=features[labels==labels[idx]]
                    anchor_idxs=[]
                    anchor_indices=[]
                    #0
                    split_sim=features[idx].unsqueeze(0).mm(split_feat.t())

                    if (torch.sum(split_sim)-1.0)/(len(split_idxs)-1)>=0.7: #confident core
                        print('sub hei')
                        continue
                    split_num+=1

                    anchor_idx=split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx.item())
                    anchor_indices.append(torch.argmin(split_sim).item())
                    for sp in range(1,self.split_num):
                        split_sim_2=features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2<split_sim]=split_sim[split_sim_2<split_sim]
                        anchor_idx=split_idxs[torch.argmin(split_sim_2)]
                        anchor_idxs.append(anchor_idx.item())
                        anchor_indices.append(torch.argmin(split_sim_2).item())
                        split_sim=split_sim_2.clone()
                    #anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==idx:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    #fix bug 104

                    Y=torch.zeros((len(split_idxs),self.split_num)).cuda()
                    Y[anchor_indices,torch.arange(self.split_num)]=1
                    # i_0,i_1=torch.argmin(split_sim),torch.argmin(split_sim_2)
                    # Y[i_0,0]=1
                    # Y[i_1,1]=1

                    #104-->fix bug
                    W=torch.exp(split_feat.mm(split_feat.t()))
                    mask=(1-torch.eye(len(split_feat))).cuda()
                    W*=mask

                    D       = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
                    D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,len(split_idxs))
                    D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(len(split_idxs),1)
                    S       = D1*W*D2

                    pred  = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda()-self.alpha*S+self.eps), Y)
                    pred=torch.argmax(pred,dim=1)
                    #lab=torch.tensor([anchor_idx.item(),anchor_idx_2.item()]).cuda()
                    lab=torch.tensor(anchor_idxs).cuda()
                    labels[split_idxs]=lab[pred]
                    labels[idx]=idx
                    #for debug
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
            else:
                print_cnts=0
                for n,idx in enumerate(indexes):
                    batch_idx=all_idxs[labels==labels[idx]]
                    batch_sub_label=sub_labels[batch_idx]
                    split_idxs, split_ind, split_cnts = np.unique(batch_sub_label.cpu().numpy(), return_index=True,
                                                                  return_counts=True)
                    split_idxs=split_idxs.tolist() #sub label
                    if len(split_idxs)<=self.split_num:
                        continue
                    anchor_idxs=[]
                    anchor_indices=[]

                    split_feat=features[split_idxs]

                    mean_cen=torch.from_numpy(split_cnts).cuda().unsqueeze(1)*split_feat
                    if (torch.sum(memory.features[idx]*mean_cen)-1.0)/(len(batch_idx)-1)>=0.6: #confident core
                        print('clu hei')
                        continue
                    split_num+=1

                    split_sim=features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
                    anchor_idx=split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(torch.argmin(split_sim).item())

                    for sp in range(1,self.split_num):
                        #fix bug 20210116
                        split_sim_2=features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2<split_sim]=split_sim[split_sim_2<split_sim]
                        anchor_idx=split_idxs[torch.argmin(split_sim_2)]
                        anchor_idxs.append(anchor_idx)
                        anchor_indices.append(torch.argmin(split_sim_2).item())
                        split_sim=split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==sub_labels[idx]:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]

                    Y=torch.zeros((len(split_idxs),self.split_num)).cuda()
                    Y[anchor_indices,torch.arange(self.split_num)]=1

                    #104-->fix bug
                    W=split_feat.mm(split_feat.t())
                    W = torch.exp(-(2 - 2 * W))
                    mask=(1-torch.eye(len(split_feat))).cuda()
                    W*=mask

                    D       = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
                    D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,len(split_idxs))
                    D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(len(split_idxs),1)
                    S       = D1*W*D2

                    pred  = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda()-self.alpha*S+self.eps), Y)
                    pred=torch.argmax(pred,dim=1)

                    #labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                    labs=outliers_label[torch.arange(n,len(outliers_label),step=len(indexes))]
                    ori_label=labels[idx].item()
                    for sub,pre in zip(split_idxs,pred):
                        labels[batch_idx[batch_sub_label==sub]]=labs[pre]
                    labels[batch_idx[batch_sub_label==sub_labels[idx]]]=ori_label#outliers_label[(self.split_num-1)*len(indexes)+n]
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
                    #split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
            print('split num:',split_num)
        elif self.method==2:
            split_num=0
            if sub_level:
                print_cnts=0
                for n,idx in enumerate(indexes):
                    split_idxs=all_idxs[labels==labels[idx]]

                    core_candidate = torch.tensor(
                        list(set(split_idxs.tolist()) & (set(ori_knn_neighbor[n].tolist())))).long().cuda()
                    if len(core_candidate)<=self.split_num:
                        continue
                    split_num+=1
                    tmp_map = {}
                    for tmp_id, x in enumerate(split_idxs):
                        tmp_map[x.item()] = tmp_id
                    anchor_idxs=[]
                    anchor_indices=[]
                    #0
                    split_feat = features[core_candidate]
                    split_sim=features[idx].unsqueeze(0).mm(split_feat.t())
                    anchor_idx=core_candidate[torch.argmin(split_sim)].item()
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(tmp_map[anchor_idx])

                    for sp in range(1,self.split_num):
                        split_sim_2=features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2<split_sim]=split_sim[split_sim_2<split_sim]
                        anchor_idx=core_candidate[torch.argmin(split_sim_2)].item()
                        anchor_idxs.append(anchor_idx)
                        anchor_indices.append(tmp_map[anchor_idx])
                        split_sim=split_sim_2.clone()
                    #anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==idx:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    #fix bug 104
                    split_feat = features[split_idxs]
                    Y=torch.zeros((len(split_idxs),self.split_num)).cuda()
                    Y[anchor_indices,torch.arange(self.split_num)]=1
                    # i_0,i_1=torch.argmin(split_sim),torch.argmin(split_sim_2)
                    # Y[i_0,0]=1
                    # Y[i_1,1]=1

                    #104-->fix bug
                    W=torch.exp(split_feat.mm(split_feat.t()))
                    mask=(1-torch.eye(len(split_feat))).cuda()
                    W*=mask

                    D       = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
                    D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,len(split_idxs))
                    D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(len(split_idxs),1)
                    S       = D1*W*D2

                    pred  = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda()-self.alpha*S+self.eps), Y)
                    pred=torch.argmax(pred,dim=1)
                    #lab=torch.tensor([anchor_idx.item(),anchor_idx_2.item()]).cuda()
                    lab=torch.tensor(anchor_idxs).cuda()
                    labels[split_idxs]=lab[pred]
                    labels[idx]=idx
                    #for debug
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
            else:
                print_cnts=0
                for n,idx in enumerate(indexes):
                    batch_idx=all_idxs[labels==labels[idx]]
                    batch_sub_label=sub_labels[batch_idx]
                    split_idxs=list(set((batch_sub_label).tolist())) #sub label

                    nei_sub_label = sub_labels[ori_knn_neighbor[n]]
                    core_candidate = torch.tensor(
                        list(set(split_idxs) & (set(nei_sub_label.tolist())))).long().cuda()
                    if len(core_candidate)<=self.split_num:
                        continue
                    split_num += 1
                    tmp_map = {}
                    for tmp_id, x in enumerate(split_idxs):
                        tmp_map[x] = tmp_id
                    anchor_idxs=[]
                    anchor_indices=[]

                    split_feat=features[core_candidate]
                    split_sim=features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
                    anchor_idx=core_candidate[torch.argmin(split_sim)].item()
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(tmp_map[anchor_idx])

                    for sp in range(1,self.split_num):
                        split_sim_2=features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2<split_sim]=split_sim[split_sim_2<split_sim]
                        anchor_idx=core_candidate[torch.argmin(split_sim_2)].item()
                        anchor_idxs.append(anchor_idx)
                        anchor_indices.append(tmp_map[anchor_idx])
                        split_sim=split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==sub_labels[idx]:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]

                    split_feat = features[split_idxs]
                    Y=torch.zeros((len(split_idxs),self.split_num)).cuda()
                    Y[anchor_indices,torch.arange(self.split_num)]=1

                    #104-->fix bug
                    W=split_feat.mm(split_feat.t())
                    W = torch.exp(-(2 - 2 * W))
                    mask=(1-torch.eye(len(split_feat))).cuda()
                    W*=mask

                    D       = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
                    D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,len(split_idxs))
                    D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(len(split_idxs),1)
                    S       = D1*W*D2

                    pred  = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda()-self.alpha*S+self.eps), Y)
                    pred=torch.argmax(pred,dim=1)

                    #labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                    labs=outliers_label[torch.arange(n,len(outliers_label),step=len(indexes))]
                    ori_label=labels[idx].item()
                    for sub,pre in zip(split_idxs,pred):
                        labels[batch_idx[batch_sub_label==sub]]=labs[pre]
                    labels[batch_idx[batch_sub_label==sub_labels[idx]]]=ori_label#outliers_label[(self.split_num-1)*len(indexes)+n]
            print('split num:',split_num)
        elif self.method==3: #method1+cluster self(<8)
            split_num = 0
            ori_labels=labels[indexes]
            unique_label=set(labels[indexes].tolist())
            unique_map={}
            if sub_level:
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    if ori_labels[n].item() in unique_label:
                        unique_label=unique_label-set([ori_labels[n].item()])
                    else:
                        ori_knn_neighbor[n,-self.split_num:]=unique_map[ori_labels[n].item()]
                        continue
                    split_idxs = all_idxs[labels == labels[idx]]
                    if len(split_idxs) <= self.split_num:
                        unique_map[ori_labels[n].item()] = ori_knn_neighbor[n, -self.split_num:]
                        continue
                    split_feat = features[labels == labels[idx]]
                    anchor_idxs = []
                    anchor_indices = []
                    # 0
                    split_sim = features[idx].unsqueeze(0).mm(split_feat.t())

                    if (torch.sum(split_sim) - 1.0) / (len(split_idxs) - 1) >= 0.7:  # confident core
                        print('sub hei')
                        unique_map[ori_labels[n].item()] = ori_knn_neighbor[n, -self.split_num:]
                        continue
                    split_num += 1

                    anchor_idx = split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx.item())
                    anchor_indices.append(torch.argmin(split_sim).item())
                    for sp in range(1, self.split_num):
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        anchor_idx = split_idxs[torch.argmin(split_sim_2)]
                        anchor_idxs.append(anchor_idx.item())
                        anchor_indices.append(torch.argmin(split_sim_2).item())
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==idx:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # fix bug 104

                    Y = torch.zeros((len(split_idxs), self.split_num)).cuda()
                    Y[anchor_indices, torch.arange(self.split_num)] = 1
                    # i_0,i_1=torch.argmin(split_sim),torch.argmin(split_sim_2)
                    # Y[i_0,0]=1
                    # Y[i_1,1]=1

                    # 104-->fix bug
                    W = torch.exp(split_feat.mm(split_feat.t()))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)
                    # lab=torch.tensor([anchor_idx.item(),anchor_idx_2.item()]).cuda()
                    lab = torch.tensor(anchor_idxs).cuda()
                    labels[split_idxs] = lab[pred]
                    labels[idx] = idx

                    #append anchor[for two hop]
                    ori_knn_neighbor[n,-self.split_num:]=torch.tensor(anchor_idxs)
                    unique_map[ori_labels[n].item()]=torch.tensor(anchor_idxs)
                    # for debug
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
            else:
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    #reduce duplicate
                    if ori_labels[n].item() in unique_label:
                        unique_label=unique_label-set([ori_labels[n].item()])
                    else:
                        ori_knn_neighbor[n, -self.split_num:]=unique_map[ori_labels[n].item()]
                        continue
                    batch_idx = all_idxs[labels == labels[idx]]
                    batch_sub_label = sub_labels[batch_idx]
                    split_idxs, split_ind,split_cnts = np.unique(batch_sub_label.cpu().numpy(), return_index=True,return_counts=True)
                    split_idxs = split_idxs.tolist()  # sub label
                    if len(split_idxs) <= self.split_num:
                        unique_map[ori_labels[n].item()]=ori_knn_neighbor[n, -self.split_num:]
                        continue
                    anchor_idxs = []
                    anchor_indices = []

                    split_feat = features[split_idxs]

                    mean_cen = torch.from_numpy(split_cnts).cuda().unsqueeze(1) * split_feat
                    if (torch.sum(memory.features[idx] * mean_cen) - 1.0) / (
                            len(batch_idx) - 1) >= 0.6:  # confident core
                        print('clu hei')
                        unique_map[ori_labels[n].item()] = ori_knn_neighbor[n, -self.split_num:]
                        continue
                    split_num += 1

                    split_sim = features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
                    anchor_idx = split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(torch.argmin(split_sim).item())

                    for sp in range(1, self.split_num):
                        # fix bug 20210116
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        anchor_idx = split_idxs[torch.argmin(split_sim_2)]
                        anchor_idxs.append(anchor_idx)
                        anchor_indices.append(torch.argmin(split_sim_2).item())
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==sub_labels[idx]:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]

                    Y = torch.zeros((len(split_idxs), self.split_num)).cuda()
                    Y[anchor_indices, torch.arange(self.split_num)] = 1

                    # 104-->fix bug
                    W = split_feat.mm(split_feat.t())
                    W = torch.exp(-(2 - 2 * W))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)

                    # labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                    labs = outliers_label[torch.arange(n, len(outliers_label), step=len(indexes))]
                    ori_label = labels[idx].item()
                    for sub, pre in zip(split_idxs, pred):
                        labels[batch_idx[batch_sub_label == sub]] = labs[pre]
                    labels[batch_idx[batch_sub_label == sub_labels[
                        idx]]] = ori_label  # outliers_label[(self.split_num-1)*len(indexes)+n]

                    #add split guys
                    split_ind=torch.from_numpy(split_ind).cuda()
                    ori_knn_neighbor[n, -self.split_num:] = batch_idx[split_ind[anchor_indices]]
                    unique_map[ori_labels[n].item()]=batch_idx[split_ind[anchor_indices]]
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
                    # split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
            print('split num:', split_num)
        elif self.method == 4:  # method1+anchor thre
            split_num = 0
            if sub_level:
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    split_idxs = all_idxs[labels == labels[idx]]
                    if len(split_idxs) <= self.split_num:
                        continue
                    split_feat = features[labels == labels[idx]]
                    anchor_idxs = []
                    anchor_indices = []
                    # 0
                    split_sim = features[idx].unsqueeze(0).mm(split_feat.t())

                    if torch.min(split_sim)>=self.anchor_thre:
                        continue
                    split_num += 1

                    anchor_idx = split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx.item())
                    anchor_indices.append(torch.argmin(split_sim).item())

                    for sp in range(1, self.split_num):
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        if torch.min(split_sim_2)>=self.anchor_thre:
                            continue
                        anchor_idx = split_idxs[torch.argmin(split_sim_2)]
                        anchor_idxs.append(anchor_idx.item())
                        anchor_indices.append(torch.argmin(split_sim_2).item())
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==idx:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # fix bug 104

                    Y = torch.zeros((len(split_idxs), len(anchor_idxs))).cuda()
                    Y[anchor_indices, torch.arange(len(anchor_idxs))] = 1
                    # i_0,i_1=torch.argmin(split_sim),torch.argmin(split_sim_2)
                    # Y[i_0,0]=1
                    # Y[i_1,1]=1

                    # 104-->fix bug
                    W = torch.exp(split_feat.mm(split_feat.t()))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)
                    # lab=torch.tensor([anchor_idx.item(),anchor_idx_2.item()]).cuda()
                    lab = torch.tensor(anchor_idxs).cuda()
                    labels[split_idxs] = lab[pred]
                    labels[idx] = idx

                    # append anchor[for two hop]
                    if ori_knn_neighbor[n,-1]<0:
                        start=min(len(ori_knn_neighbor[n])-self.split_num,torch.argmin(ori_knn_neighbor[n]).item())
                    else:
                        start=len(ori_knn_neighbor[n])-self.split_num
                    ori_knn_neighbor[n, start:start+len(anchor_idxs)]=torch.tensor(anchor_idxs)
                    print('{} | sub split idxs:'.format(len(split_idxs)),len(anchor_idxs))
                    # for debug
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
            else:
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    # reduce duplicate
                    batch_idx = all_idxs[labels == labels[idx]]
                    batch_sub_label = sub_labels[batch_idx]
                    split_idxs, split_ind, split_cnts = np.unique(batch_sub_label.cpu().numpy(), return_index=True,
                                                                  return_counts=True)
                    split_idxs = split_idxs.tolist()  # sub label
                    if len(split_idxs) <= self.split_num:
                        continue
                    anchor_idxs = []
                    anchor_indices = []

                    split_feat = features[split_idxs]

                    split_sim = features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
                    if torch.min(split_sim) >= self.anchor_thre:
                        continue

                    split_num += 1

                    anchor_idx = split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(torch.argmin(split_sim).item())

                    for sp in range(1, self.split_num):
                        # fix bug 20210116
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        if torch.min(split_sim_2)>=self.anchor_thre:
                            continue
                        anchor_idx = split_idxs[torch.argmin(split_sim_2)]
                        anchor_idxs.append(anchor_idx)
                        anchor_indices.append(torch.argmin(split_sim_2).item())
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==sub_labels[idx]:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]

                    Y = torch.zeros((len(split_idxs), len(anchor_idxs))).cuda()
                    Y[anchor_indices, torch.arange(len(anchor_idxs))] = 1

                    # 104-->fix bug
                    W = split_feat.mm(split_feat.t())
                    W = torch.exp(-(2 - 2 * W))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)
                    if len(batch_idx)>3000:
                        print('pred:',pred)

                    # labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                    labs = outliers_label[torch.arange(n, len(outliers_label), step=len(indexes))]
                    ori_label = labels[idx].item()
                    for sub, pre in zip(split_idxs, pred):
                        labels[batch_idx[batch_sub_label == sub]] = labs[pre]
                    labels[batch_idx[batch_sub_label == sub_labels[
                        idx]]] = ori_label  # outliers_label[(self.split_num-1)*len(indexes)+n]

                    # add split guys
                    split_ind = torch.from_numpy(split_ind).cuda()
                    if ori_knn_neighbor[n, -1] < 0:
                        start = min(len(ori_knn_neighbor[n]) - self.split_num, torch.argmin(ori_knn_neighbor[n]).item())
                    else:
                        start=len(ori_knn_neighbor[n]) - self.split_num
                    ori_knn_neighbor[n, start:start+len(anchor_idxs)] = batch_idx[split_ind[anchor_indices]]
                    print('{}| clu split idxs:'.format(len(batch_idx)), len(anchor_idxs))
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
                    # split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
            print('split num:', split_num)
        elif self.method == 5:  # method1+anchor thre+wo split self alone
            split_num = 0
            if sub_level:
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    split_idxs = all_idxs[labels == labels[idx]]
                    if len(split_idxs) <= self.split_num:
                        continue
                    split_feat = features[labels == labels[idx]]
                    anchor_idxs = []
                    anchor_indices = []
                    # 0
                    split_sim = features[idx].unsqueeze(0).mm(split_feat.t())

                    if torch.min(split_sim)>=self.anchor_thre:
                        continue
                    split_num += 1

                    anchor_idxs.append(split_idxs[torch.argmax(split_sim)].item())#index self
                    anchor_indices.append(torch.argmax(split_sim).item())

                    anchor_idx = split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx.item())
                    anchor_indices.append(torch.argmin(split_sim).item())

                    for sp in range(2, self.split_num):
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        if torch.min(split_sim_2)>=self.anchor_thre:
                            continue
                        anchor_idx = split_idxs[torch.argmin(split_sim_2)]
                        anchor_idxs.append(anchor_idx.item())
                        anchor_indices.append(torch.argmin(split_sim_2).item())
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==idx:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # fix bug 104

                    Y = torch.zeros((len(split_idxs), len(anchor_idxs))).cuda()
                    Y[anchor_indices, torch.arange(len(anchor_idxs))] = 1
                    # i_0,i_1=torch.argmin(split_sim),torch.argmin(split_sim_2)
                    # Y[i_0,0]=1
                    # Y[i_1,1]=1

                    # 104-->fix bug
                    W = torch.exp(split_feat.mm(split_feat.t()))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)
                    # lab=torch.tensor([anchor_idx.item(),anchor_idx_2.item()]).cuda()
                    lab = torch.tensor(anchor_idxs).cuda()
                    labels[split_idxs] = lab[pred]
                    #labels[idx] = idx

                    # append anchor[for two hop]
                    if len(anchor_idxs)==self.split_num:
                        ori_knn_neighbor[n, -self.split_num:] = torch.tensor(anchor_idxs)
                    else:
                        ori_knn_neighbor[n, -self.split_num:-self.split_num+len(anchor_idxs)]=torch.tensor(anchor_idxs)
                    print('{} | sub split idxs:'.format(len(split_idxs)),len(anchor_idxs))
                    # for debug
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
            else:
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    # reduce duplicate
                    batch_idx = all_idxs[labels == labels[idx]]
                    # if len(batch_idx)>3000:
                    #     print('------------>3000-----------')
                    #     import pdb;pdb.set_trace()

                    batch_sub_label = sub_labels[batch_idx]
                    split_idxs, split_ind, split_cnts = np.unique(batch_sub_label.cpu().numpy(), return_index=True,
                                                                  return_counts=True)
                    split_idxs = split_idxs.tolist()  # sub label
                    if len(split_idxs) <= self.split_num:
                        continue
                    anchor_idxs = []
                    anchor_indices = []

                    split_feat = features[split_idxs]

                    split_sim = features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
                    if torch.min(split_sim) >= self.anchor_thre:
                        continue

                    split_num += 1

                    anchor_idxs.append(split_idxs[torch.argmax(split_sim).item()])  # index self
                    anchor_indices.append(torch.argmax(split_sim).item())

                    anchor_idx = split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(torch.argmin(split_sim).item())

                    for sp in range(2, self.split_num):
                        # fix bug 20210116
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        if torch.min(split_sim_2)>=self.anchor_thre:
                            continue
                        anchor_idx = split_idxs[torch.argmin(split_sim_2)]
                        anchor_idxs.append(anchor_idx)
                        anchor_indices.append(torch.argmin(split_sim_2).item())
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==sub_labels[idx]:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]

                    Y = torch.zeros((len(split_idxs), len(anchor_idxs))).cuda()
                    Y[anchor_indices, torch.arange(len(anchor_idxs))] = 1

                    # if len(batch_idx)>3000:
                    #     print('------------>3000-----------')
                    #     import pdb;pdb.set_trace()

                    # 104-->fix bug
                    W = split_feat.mm(split_feat.t())
                    W = torch.exp(-(2 - 2 * W))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)
                    if len(batch_idx)>3000:
                        print('pred:',pred)

                    # labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                    labs = outliers_label[torch.arange(n, len(outliers_label), step=len(indexes))]
                    #ori_label = labels[idx].item()
                    for sub, pre in zip(split_idxs, pred):
                        labels[batch_idx[batch_sub_label == sub]] = labs[pre]
                    # labels[batch_idx[batch_sub_label == sub_labels[
                    #     idx]]] = ori_label  # outliers_label[(self.split_num-1)*len(indexes)+n]

                    # add split guys
                    split_ind = torch.from_numpy(split_ind).cuda()
                    if len(anchor_idxs)==self.split_num:
                        ori_knn_neighbor[n, -self.split_num:] = batch_idx[split_ind[anchor_indices]]
                    else:
                        ori_knn_neighbor[n, -self.split_num:-self.split_num+len(anchor_idxs)] = batch_idx[split_ind[anchor_indices]]
                    print('{}| clu split idxs:{} | {}'.format(len(batch_idx),len(anchor_idxs),split_cnts[anchor_indices]))
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
                    # split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
        elif self.method == 6:  # method1+anchor thre+wo split self alone
            empty_label = set(torch.arange(labels.max() + 1).tolist()) - set(labels.tolist())
            split_num = 0
            if sub_level:
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    split_idxs = all_idxs[labels == labels[idx]]
                    inter = list(set(ori_knn_neighbor[i].tolist()) & set(split_idxs.tolist()))
                    if len(inter)==0:
                        continue
                    # if len(split_idxs) <= self.split_num:
                    #     continue

                    split_feat = features[labels == labels[idx]]
                    anchor_idxs = []
                    anchor_indices = []
                    # 0
                    split_sim = features[idx].unsqueeze(0).mm(split_feat.t())

                    if torch.min(split_sim)>=0.4:
                        continue
                    split_num += 1

                    anchor_idxs.append(split_idxs[torch.argmax(split_sim)].item())#index self
                    anchor_indices.append(torch.argmax(split_sim).item())

                    anchor_idx = split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx.item())
                    anchor_indices.append(torch.argmin(split_sim).item())

                    for sp in range(2, self.split_num):
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        if torch.min(split_sim_2)>=self.anchor_thre:
                            continue
                        anchor_idx = split_idxs[torch.argmin(split_sim_2)]
                        anchor_idxs.append(anchor_idx.item())
                        anchor_indices.append(torch.argmin(split_sim_2).item())
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==idx:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # fix bug 104

                    Y = torch.zeros((len(split_idxs), len(anchor_idxs))).cuda()
                    Y[anchor_indices, torch.arange(len(anchor_idxs))] = 1
                    # i_0,i_1=torch.argmin(split_sim),torch.argmin(split_sim_2)
                    # Y[i_0,0]=1
                    # Y[i_1,1]=1

                    # 104-->fix bug
                    W = torch.exp(split_feat.mm(split_feat.t()))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)
                    # lab=torch.tensor([anchor_idx.item(),anchor_idx_2.item()]).cuda()
                    lab = torch.tensor(anchor_idxs).cuda()
                    labels[split_idxs] = lab[pred]
                    #labels[idx] = idx

                    # append anchor[for two hop]
                    if len(anchor_idxs)==self.split_num:
                        ori_knn_neighbor[n, -self.split_num:] = torch.tensor(anchor_idxs)
                    else:
                        ori_knn_neighbor[n, -self.split_num:-self.split_num+len(anchor_idxs)]=torch.tensor(anchor_idxs)
                    print('{} | sub split idxs:'.format(len(split_idxs)),len(anchor_idxs))
                    # for debug
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
            else:
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    empty_label_list=list(empty_label)
                    # reduce duplicate
                    batch_idx = all_idxs[labels == labels[idx]]
                    if len(batch_idx)<=self.split_num:
                        continue
                    # if len(batch_idx)>3000:
                    #     print('------------>3000-----------')
                    #     import pdb;pdb.set_trace()

                    batch_sub_label = sub_labels[batch_idx]
                    # split_idxs, split_ind, split_cnts = np.unique(batch_sub_label.cpu().numpy(), return_index=True,
                    #                                               return_counts=True)
                    split_idxs=batch_idx
                    split_idxs = split_idxs.tolist()  # sub label
                    inter = list(set(ori_knn_neighbor[n].tolist()) & set(split_idxs))
                    if len(inter) <= 1:
                        continue

                    tmp_map = {}
                    for inter_n, inter_idx in enumerate(ori_knn_neighbor[n].tolist()):
                        tmp_map[inter_idx] = inter_n
                    inter_idxs = []
                    for aa in inter:
                        inter_idxs.append(tmp_map[aa])

                    #compute inter
                    for inter_idx in inter:
                        inter_n=tmp_map[inter_idx]
                        W_tmp = torch.sum(torch.min(point_W[n, inter_n, :].unsqueeze(1).expand_as(point_W[n,:,:]), point_W[n]), dim=-1)
                        point_pred[n,inter_n]=W_tmp / (2 - W_tmp)
                        point_pred[n,inter_n,inter_n]=0
                    #import pdb;pdb.set_trace()
                    anchor_idxs = []
                    anchor_indices = []

                    #split_feat = features[split_idxs]

                    #split_sim = features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())

                    split_sim=point_pred[n,tmp_map[idx.item()]][inter_idxs]
                    if torch.min(split_sim) >= 0.4:
                        continue

                    batch_map={}
                    for aa_n,aa in enumerate(batch_idx.tolist()):
                        batch_map[aa]=aa_n

                    split_num += 1

                    anchor_idxs.append(idx.item())  # index self
                    anchor_indices.append(batch_map[idx.item()])

                    anchor_idx = inter[torch.argmin(split_sim).item()]
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(batch_map[anchor_idx])

                    for sp in range(2,len(inter)):
                        # fix bug 20210116
                        split_sim_2=point_pred[n,tmp_map[anchor_idx]][inter_idxs]
                        #split_sim_2 = #features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        if torch.min(split_sim_2)>=0.4:
                            continue
                        #anchor_idx = split_idxs[torch.argmin(split_sim_2)]
                        anchor_idx=inter[torch.argmin(split_sim_2).item()]
                        anchor_idxs.append(anchor_idx)
                        anchor_indices.append(batch_map[anchor_idx])
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==sub_labels[idx]:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]

                    Y = torch.zeros((len(split_idxs), len(anchor_idxs))).cuda()
                    Y[anchor_indices, torch.arange(len(anchor_idxs))] = 1

                    # if len(batch_idx)>3000:
                    #     print('------------>3000-----------')
                    #     import pdb;pdb.set_trace()

                    # 104-->fix bug
                    split_feat=features[labels == labels[idx]]
                    W = split_feat.mm(split_feat.t())
                    W = torch.exp(-(2 - 2 * W))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)

                    # labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                    #labs = outliers_label[torch.arange(n, len(outliers_label), step=len(indexes))]
                    if len(empty_label_list)>=len(anchor_idxs):
                        labs=empty_label_list[:len(anchor_idxs)]
                        empty_label=empty_label-set(labs)
                    else:
                        labs=torch.arange(labels.max() + 1,labels.max() + 1+len(anchor_idxs))
                    #ori_label = labels[idx].item()
                    for sub, pre in zip(split_idxs, pred):
                        labels[batch_idx[batch_sub_label == sub]] = labs[pre]
                    # labels[batch_idx[batch_sub_label == sub_labels[
                    #     idx]]] = ori_label  # outliers_label[(self.split_num-1)*len(indexes)+n]
                    print('{}| clu split idxs:{}'.format(len(batch_idx),len(anchor_idxs)))
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
                    # split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
        elif self.method == 7:  # method1+anchor thre+anchor idx in nei
            split_num = 0
            if sub_level:
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    split_idxs = all_idxs[labels == labels[idx]]
                    if len(split_idxs) <= self.split_num:
                        continue
                    anchor_idxs = []
                    anchor_indices = []

                    inter=list(set(split_idxs.tolist()) & set(ori_knn_neighbor[n].tolist()))
                    if len(inter)<=1:
                        continue

                    split_map={}
                    for sp_n,sp_idx in enumerate(split_idxs.tolist()):
                        split_map[sp_idx]=sp_n

                    # 0
                    split_feat=features[inter]
                    split_sim = features[idx].unsqueeze(0).mm(split_feat.t())

                    if torch.min(split_sim)>=self.anchor_thre:
                        continue
                    split_num += 1

                    anchor_idxs.append(idx.item())#index self
                    anchor_indices.append(split_map[idx.item()])

                    anchor_idx = inter[torch.argmin(split_sim).item()]
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(split_map[anchor_idx])

                    for sp in range(2, self.split_num):
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        if torch.min(split_sim_2)>=self.anchor_thre:
                            continue
                        anchor_idx = inter[torch.argmin(split_sim_2).item()]
                        anchor_idxs.append(anchor_idx)
                        anchor_indices.append(split_map[anchor_idx])
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==idx:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # fix bug 104
                    split_feat = features[labels == labels[idx]]
                    Y = torch.zeros((len(split_idxs), len(anchor_idxs))).cuda()
                    Y[anchor_indices, torch.arange(len(anchor_idxs))] = 1
                    # i_0,i_1=torch.argmin(split_sim),torch.argmin(split_sim_2)
                    # Y[i_0,0]=1
                    # Y[i_1,1]=1

                    # 104-->fix bug
                    W = torch.exp(split_feat.mm(split_feat.t()))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)
                    # lab=torch.tensor([anchor_idx.item(),anchor_idx_2.item()]).cuda()
                    lab = torch.tensor(anchor_idxs).cuda()
                    labels[split_idxs] = lab[pred]
                    #labels[idx] = idx

                    # append anchor[for two hop]
                    if len(anchor_idxs)==self.split_num:
                        ori_knn_neighbor[n, -self.split_num:] = torch.tensor(anchor_idxs)
                    else:
                        ori_knn_neighbor[n, -self.split_num:-self.split_num+len(anchor_idxs)]=torch.tensor(anchor_idxs)
                    print('{} | sub split idxs:'.format(len(split_idxs)),len(anchor_idxs))
                    # for debug
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
            else:
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    # reduce duplicate
                    batch_idx = all_idxs[labels == labels[idx]]
                    # if len(batch_idx)>3000:
                    #     print('------------>3000-----------')
                    #     import pdb;pdb.set_trace()

                    batch_sub_label = sub_labels[batch_idx]
                    split_idxs, split_ind, split_cnts = np.unique(batch_sub_label.cpu().numpy(), return_index=True,
                                                                  return_counts=True)
                    split_idxs = split_idxs.tolist()  # sub label
                    if len(split_idxs) <= self.split_num:
                        continue
                    anchor_idxs = []
                    anchor_indices = []

                    inter=list(set(split_idxs.tolist()) & set(sub_labels[ori_knn_neighbor[n]].tolist()))
                    if len(inter)<=1:
                        continue
                    split_feat=features[inter]
                    split_map = {}
                    for sp_n, sp_idx in enumerate(split_idxs.tolist()):
                        split_map[sp_idx] = sp_n

                    split_sim = features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
                    if torch.min(split_sim) >= self.anchor_thre:
                        continue

                    split_num += 1

                    anchor_idxs.append(idx.item())  # index self
                    anchor_indices.append(split_map[idx.item()])

                    anchor_idx = inter[torch.argmin(split_sim).item()]
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(split_map[anchor_idx])

                    for sp in range(2, self.split_num):
                        # fix bug 20210116
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        if torch.min(split_sim_2)>=self.anchor_thre:
                            continue
                        anchor_idx = inter[torch.argmin(split_sim_2).item()]
                        anchor_idxs.append(anchor_idx)
                        anchor_indices.append(split_map[anchor_idx])
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==sub_labels[idx]:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]

                    split_feat = features[split_idxs]
                    Y = torch.zeros((len(split_idxs), len(anchor_idxs))).cuda()
                    Y[anchor_indices, torch.arange(len(anchor_idxs))] = 1

                    # if len(batch_idx)>3000:
                    #     print('------------>3000-----------')
                    #     import pdb;pdb.set_trace()

                    # 104-->fix bug
                    W = split_feat.mm(split_feat.t())
                    W = torch.exp(-(2 - 2 * W))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)
                    if len(batch_idx)>3000:
                        print('pred:',pred)

                    # labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                    labs = outliers_label[torch.arange(n, len(outliers_label), step=len(indexes))]
                    #ori_label = labels[idx].item()
                    for sub, pre in zip(split_idxs, pred):
                        labels[batch_idx[batch_sub_label == sub]] = labs[pre]
                    # labels[batch_idx[batch_sub_label == sub_labels[
                    #     idx]]] = ori_label  # outliers_label[(self.split_num-1)*len(indexes)+n]

                    print('{}| clu split idxs:{} | {}'.format(len(batch_idx),len(anchor_idxs),split_cnts[anchor_indices]))
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
                    # split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
        elif self.method == 8:  # method1+anchor thre+anchor idx in nei+wo num restriction
            split_num = 0
            if sub_level:
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    split_idxs = all_idxs[labels == labels[idx]]
                    # if len(split_idxs) <= self.split_num:
                    #     continue
                    anchor_idxs = []
                    anchor_indices = []

                    inter=list(set(split_idxs.tolist()) & set(ori_knn_neighbor[n].tolist()))
                    if len(inter)<=1:
                        continue

                    split_map={}
                    for sp_n,sp_idx in enumerate(split_idxs.tolist()):
                        split_map[sp_idx]=sp_n

                    # 0
                    split_feat=features[inter]
                    split_sim = features[idx].unsqueeze(0).mm(split_feat.t())

                    if torch.min(split_sim)>=self.anchor_thre:
                        continue
                    split_num += 1

                    anchor_idxs.append(idx.item())#index self
                    anchor_indices.append(split_map[idx.item()])

                    anchor_idx = inter[torch.argmin(split_sim).item()]
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(split_map[anchor_idx])

                    for sp in range(2, len(inter)):
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        if torch.min(split_sim_2)>=self.anchor_thre:
                            continue
                        anchor_idx = inter[torch.argmin(split_sim_2).item()]
                        anchor_idxs.append(anchor_idx)
                        anchor_indices.append(split_map[anchor_idx])
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==idx:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # fix bug 104
                    split_feat = features[labels == labels[idx]]
                    Y = torch.zeros((len(split_idxs), len(anchor_idxs))).cuda()
                    Y[anchor_indices, torch.arange(len(anchor_idxs))] = 1
                    # i_0,i_1=torch.argmin(split_sim),torch.argmin(split_sim_2)
                    # Y[i_0,0]=1
                    # Y[i_1,1]=1

                    # 104-->fix bug
                    W = torch.exp(split_feat.mm(split_feat.t()))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)
                    # lab=torch.tensor([anchor_idx.item(),anchor_idx_2.item()]).cuda()
                    lab = torch.tensor(anchor_idxs).cuda()
                    labels[split_idxs] = lab[pred]
                    #labels[idx] = idx

                    # append anchor[for two hop]
                    if len(anchor_idxs)==self.split_num:
                        ori_knn_neighbor[n, -self.split_num:] = torch.tensor(anchor_idxs)
                    else:
                        ori_knn_neighbor[n, -self.split_num:-self.split_num+len(anchor_idxs)]=torch.tensor(anchor_idxs)
                    print('{} | sub split idxs:'.format(len(split_idxs)),len(anchor_idxs))
                    # for debug
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
            else:
                empty_label = set(torch.arange(labels.max() + 1).tolist()) - set(labels.tolist())
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    empty_label_list = list(empty_label)
                    # reduce duplicate
                    batch_idx = all_idxs[labels == labels[idx]]
                    # if len(batch_idx)>3000:
                    #     print('------------>3000-----------')
                    #     import pdb;pdb.set_trace()

                    batch_sub_label = sub_labels[batch_idx]
                    split_idxs, split_ind, split_cnts = np.unique(batch_sub_label.cpu().numpy(), return_index=True,
                                                                  return_counts=True)
                    split_idxs = split_idxs.tolist()  # sub label
                    # if len(split_idxs) <= self.split_num:
                    #     continue
                    anchor_idxs = []
                    anchor_indices = []

                    inter=list(set(split_idxs) & set(sub_labels[ori_knn_neighbor[n]].tolist()))
                    if len(inter)<=1:
                        continue
                    split_feat=features[inter]
                    split_map = {}
                    for sp_n, sp_idx in enumerate(split_idxs.tolist()):
                        split_map[sp_idx] = sp_n

                    split_sim = features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
                    if torch.min(split_sim) >= self.anchor_thre:
                        continue

                    split_num += 1

                    anchor_idxs.append(idx.item())  # index self
                    anchor_indices.append(split_map[idx.item()])

                    anchor_idx = inter[torch.argmin(split_sim).item()]
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(split_map[anchor_idx])

                    for sp in range(2, len(inter)):
                        # fix bug 20210116
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        if torch.min(split_sim_2)>=self.anchor_thre:
                            continue
                        anchor_idx = inter[torch.argmin(split_sim_2).item()]
                        anchor_idxs.append(anchor_idx)
                        anchor_indices.append(split_map[anchor_idx])
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==sub_labels[idx]:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]

                    split_feat = features[split_idxs]
                    Y = torch.zeros((len(split_idxs), len(anchor_idxs))).cuda()
                    Y[anchor_indices, torch.arange(len(anchor_idxs))] = 1

                    # if len(batch_idx)>3000:
                    #     print('------------>3000-----------')
                    #     import pdb;pdb.set_trace()

                    # 104-->fix bug
                    W = split_feat.mm(split_feat.t())
                    W = torch.exp(-(2 - 2 * W))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)
                    if len(batch_idx)>3000:
                        print('pred:',pred)

                    # labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                    if len(empty_label_list)>=len(anchor_idxs):
                        labs=empty_label_list[:len(anchor_idxs)]
                        empty_label=empty_label-set(labs)
                    else:
                        labs=torch.arange(labels.max() + 1,labels.max() + 1+len(anchor_idxs))
                    #ori_label = labels[idx].item()
                    for sub, pre in zip(split_idxs, pred):
                        labels[batch_idx[batch_sub_label == sub]] = labs[pre]
                    # labels[batch_idx[batch_sub_label == sub_labels[
                    #     idx]]] = ori_label  # outliers_label[(self.split_num-1)*len(indexes)+n]

                    print('{}| clu split idxs:{} | {}'.format(len(batch_idx),len(anchor_idxs),split_cnts[anchor_indices]))
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
                    # split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
        elif self.method == 9: #final one
            split_num = 0
            if sub_level:
                for n, idx in enumerate(indexes):
                    split_idxs = all_idxs[labels == labels[idx]]
                    if len(split_idxs) <= self.split_num:
                        continue
                    split_feat = features[labels == labels[idx]]
                    anchor_idxs = []
                    anchor_indices = []
                    # 0
                    split_sim = features[idx].unsqueeze(0).mm(split_feat.t())

                    if torch.min(split_sim) >= self.anchor_thre:
                        continue
                    split_num += 1

                    anchor_idxs.append(split_idxs[torch.argmax(split_sim)].item())  # index self
                    anchor_indices.append(torch.argmax(split_sim).item())

                    anchor_idx = split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx.item())
                    anchor_indices.append(torch.argmin(split_sim).item())

                    for sp in range(2, self.split_num):
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        if torch.min(split_sim_2) >= self.anchor_thre:
                            continue
                        anchor_idx = split_idxs[torch.argmin(split_sim_2)]
                        anchor_idxs.append(anchor_idx.item())
                        anchor_indices.append(torch.argmin(split_sim_2).item())
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==idx:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # fix bug 104

                    Y = torch.zeros((len(split_idxs), len(anchor_idxs))).cuda()
                    Y[anchor_indices, torch.arange(len(anchor_idxs))] = 1
                    # i_0,i_1=torch.argmin(split_sim),torch.argmin(split_sim_2)
                    # Y[i_0,0]=1
                    # Y[i_1,1]=1

                    # 104-->fix bug
                    W = torch.exp(split_feat.mm(split_feat.t()))
                    #0227#####
                    if W.size(-1)>self.connect_num:
                        topk, indices = torch.topk(W, self.connect_num, dim=-1)
                        mask_top = torch.zeros_like(W)
                        mask_top = mask_top.scatter(-1, indices, 1)
                        mask_top = ((mask_top > 0) & (mask_top.t() > 0)).type(torch.float32)
                        W *= mask_top
                    ############
                    W = torch.exp(-(2 - 2 * W))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)
                    # lab=torch.tensor([anchor_idx.item(),anchor_idx_2.item()]).cuda()
                    lab = torch.tensor(anchor_idxs).cuda()
                    labels[split_idxs] = lab[pred]
                    # labels[idx] = idx

                    # append anchor[for two hop]
                    # if len(anchor_idxs) == self.split_num:
                    #     ori_knn_neighbor[n, -self.split_num:] = torch.tensor(anchor_idxs)
                    # else:
                    #     ori_knn_neighbor[n, -self.split_num:-self.split_num + len(anchor_idxs)] = torch.tensor(
                    #         anchor_idxs)
                    print('{} | sub split idxs:'.format(len(split_idxs)), len(anchor_idxs))
                    # for debug
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
            else:
                print_cnts = 0
                for n, idx in enumerate(indexes):
                    # reduce duplicate
                    batch_idx = all_idxs[labels == labels[idx]]
                    # if len(batch_idx)>3000:
                    #     print('------------>3000-----------')
                    #     import pdb;pdb.set_trace()

                    batch_sub_label = sub_labels[batch_idx]
                    split_idxs, split_ind, split_cnts = np.unique(batch_sub_label.cpu().numpy(), return_index=True,
                                                                  return_counts=True)
                    split_idxs = split_idxs.tolist()  # sub label
                    if len(split_idxs) <= self.split_num:
                        continue
                    anchor_idxs = []
                    anchor_indices = []

                    split_feat = features[split_idxs]

                    split_sim = features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
                    if torch.min(split_sim) >= self.anchor_thre:
                        continue

                    split_num += 1

                    anchor_idxs.append(split_idxs[torch.argmax(split_sim).item()])  # index self
                    anchor_indices.append(torch.argmax(split_sim).item())

                    anchor_idx = split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(torch.argmin(split_sim).item())

                    for sp in range(2, self.split_num):
                        # fix bug 20210116
                        split_sim_2 = features[anchor_idx].unsqueeze(0).mm(split_feat.t())
                        split_sim_2[split_sim_2 < split_sim] = split_sim[split_sim_2 < split_sim]
                        if torch.min(split_sim_2) >= self.anchor_thre:
                            continue
                        anchor_idx = split_idxs[torch.argmin(split_sim_2)]
                        anchor_idxs.append(anchor_idx)
                        anchor_indices.append(torch.argmin(split_sim_2).item())
                        split_sim = split_sim_2.clone()
                    # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
                    # if anchor_idx_2==sub_labels[idx]:
                    #     split_sim_2[0,torch.argmin(split_sim_2)]=1
                    #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]

                    Y = torch.zeros((len(split_idxs), len(anchor_idxs))).cuda()
                    Y[anchor_indices, torch.arange(len(anchor_idxs))] = 1

                    # if len(batch_idx)>3000:
                    #     print('------------>3000-----------')
                    #     import pdb;pdb.set_trace()

                    # 104-->fix bug
                    W = split_feat.mm(split_feat.t())
                    # 0227#####
                    if W.size(-1) > self.connect_num:
                        topk, indices = torch.topk(W, self.connect_num, dim=-1)
                        mask_top = torch.zeros_like(W)
                        mask_top = mask_top.scatter(-1, indices, 1)
                        mask_top = ((mask_top > 0) & (mask_top.t() > 0)).type(torch.float32)
                        W *= mask_top
                    ############
                    W = torch.exp(-(2 - 2 * W))
                    mask = (1 - torch.eye(len(split_feat))).cuda()
                    W *= mask

                    D = W.sum(0)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + self.eps))
                    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(split_idxs))
                    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(split_idxs), 1)
                    S = D1 * W * D2

                    pred = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda() - self.alpha * S + self.eps), Y)
                    pred = torch.argmax(pred, dim=1)
                    # if len(batch_idx) > 3000:
                    #     print('pred:', pred)

                    # labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                    labs = outliers_label[torch.arange(n, len(outliers_label), step=len(indexes))]
                    # ori_label = labels[idx].item()
                    for sub, pre in zip(split_idxs, pred):
                        labels[batch_idx[batch_sub_label == sub]] = labs[pre]
                    # labels[batch_idx[batch_sub_label == sub_labels[
                    #     idx]]] = ori_label  # outliers_label[(self.split_num-1)*len(indexes)+n]

                    # add split guys
                    # split_ind = torch.from_numpy(split_ind).cuda()
                    # if len(anchor_idxs) == self.split_num:
                    #     ori_knn_neighbor[n, -self.split_num:] = batch_idx[split_ind[anchor_indices]]
                    # else:
                    #     ori_knn_neighbor[n, -self.split_num:-self.split_num + len(anchor_idxs)] = batch_idx[
                    #         split_ind[anchor_indices]]
                    print('{}| clu split idxs:{} | {}'.format(len(batch_idx), len(anchor_idxs),
                                                              split_cnts[anchor_indices]))
                    # if print_cnts==0:
                    #     print(pred)
                    #     print_cnts=1
                    # split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
        return ori_knn_neighbor

class Hierarchy_GCN(object):
    def __init__(self,point_level_lp,sub_cluster_level_lp,cluster_level_lp,split_lp,utils,neighbor_num=64,thre=[0,06.15,0.1],
                debug_label=[],merge_wo_outlier=0,jaccard_debug=0):
        self.point_level_lp=point_level_lp
        self.sub_cluster_level_lp=sub_cluster_level_lp
        self.cluster_level_lp=cluster_level_lp
        self.split_lp=split_lp
        self.utils=utils #utils

        self.neighbor_num=neighbor_num #knn
        self.thre=thre
        self.debug_label=debug_label
        self.debug_label_num=None
        self.merge_wo_outlier=merge_wo_outlier
        self.two_hop=0
        self.jaccard_debug=jaccard_debug

    def train(self,s_indexes,memory,train,f_s):
        if train:
            #for loss_s backward
            cal_feat=memory.momentum*f_s+(1. -memory.momentum)*memory.s_features[s_indexes]
            with torch.no_grad():
                norm=cal_feat.norm(dim=1).unsqueeze(1)
            cal_feat/=norm
            #cal knn neighbor
            ori_0=compute_knn(memory.s_features.clone(),k1=self.neighbor_num)
            ori_knn_neighbor=torch.from_numpy(ori_0[s_indexes.cpu().numpy(),:]).cuda()
            #compute gt
            all_gt_label=memory.s_label[ori_knn_neighbor.view(-1)].view(len(s_indexes),-1)
            all_gt_sub_label=memory.s_sub_label[ori_knn_neighbor.view(-1)].view(len(s_indexes),-1)
            #all_gt_sub_label=memory.s_sub_label[ori_knn_neighbor.view(-1)].view(len(s_indexes),-1)
            gt_conf=(all_gt_label==all_gt_label[:,0].unsqueeze(1).expand_as(all_gt_label)).float()

            loss_point_level=self.point_level_lp(s_indexes,memory.s_features,self.neighbor_num,ori_0,ori_knn_neighbor,gt_conf,f_s=cal_feat,train=1)
            loss_sub_level=self.sub_cluster_level_lp(s_indexes,memory.s_features,self.neighbor_num,ori_0,ori_knn_neighbor,gt_conf,f_s=cal_feat,sub_label=memory.s_sub_label,gt_sub_label=all_gt_sub_label,gt_label=all_gt_label,train=1)
            loss_cluster_level=self.cluster_level_lp(s_indexes,memory.s_features,self.neighbor_num,ori_0,ori_knn_neighbor,gt_conf,f_s=cal_feat,labels=memory.s_label,gt_label=all_gt_label,train=1)

        #update feat
        with torch.no_grad():
            for x, y in zip(f_s, s_indexes):
                memory.s_features[y] = memory.momentum * memory.s_features[y] + (1. - memory.momentum) * x
                memory.s_features[y] /= memory.s_features[y].norm()

        if train:
            #train split
            loss_split_gcn=self.split_gcn(s_indexes,memory.s_features,memory.s_label,train=1,ori_knn_neighbor=ori_knn_neighbor,gt=gt_conf.long())
            loss_all=loss_point_level+loss_sub_level+loss_cluster_level
            #print('point:{} sub:{} clu: {}'.format(loss_point_level,loss_sub_level,loss_cluster_level))
            return loss_all,loss_split_gcn#[loss_point_level,loss_sub_level,loss_cluster_level]
        else:
            return torch.tensor(0),torch.tensor(0)

    def inference(self,t_indexes,memory,infer):
        torch.cuda.empty_cache()
        #debug
        #accs = []
        # for i in range(len(t_indexes)):
        #     batch_lab = self.debug_label[memory.labels[memory.source_classes:] == memory.labels[t_indexes[i]]]
        #     if len(batch_lab) > 3:
        #         acc = 1.0 * torch.sum(batch_lab == self.debug_label[t_indexes[i] - memory.source_classes]) / len(
        #             batch_lab)
        #         accs.append('[{}] {:.2f} {}/{}'.format(t_indexes[i].item(),acc, len(batch_lab),int(self.debug_label_num[(self.debug_label[t_indexes[i]-memory.source_classes]).item()])))
        # print('before acc:', accs)
        # del batch_lab

        #cal knn
        cal_feat=memory.features[t_indexes]
        if self.two_hop:
            ori_0 = compute_knn(memory.features, k1=20)#20*20
        else:
            ori_0=compute_knn(memory.features,k1=self.neighbor_num)
        ori_knn_neighbor=torch.from_numpy(ori_0[t_indexes.cpu().numpy(),:]).cuda()
        ########label-->outlier
        #step1 change sub label
        ori_labels=memory.t_sub_label[t_indexes]
        change_sub_label=t_indexes[ori_labels==t_indexes]
        memory.t_sub_label[t_indexes]=0
        if len(change_sub_label)>0:
            all_label=torch.arange(len(memory.features)).cuda()
            for change_lab in change_sub_label:
                change_idx=all_label[memory.t_sub_label==change_lab]
                if len(change_idx)>0:
                    memory.t_sub_label[change_idx]=change_idx[0]
        memory.t_sub_label[t_indexes]=t_indexes
        #step2 change label
        empty_label=set(torch.arange(memory.labels.max()+1).tolist())-set(memory.labels.tolist())
        if len(empty_label)<2*len(t_indexes):
            outliers_label=torch.arange(memory.labels.max()+1,memory.labels.max()+1+2*len(t_indexes)).cuda()
        else:
            empty_label=list(empty_label)[-2*len(t_indexes):]
            outliers_label=torch.tensor(empty_label).cuda()
        memory.labels[t_indexes]=outliers_label[:len(t_indexes)]

        ###########################
        #point level pred
        all_pred,point_W,point_neighbor=self.point_level_lp(t_indexes,memory.features,self.neighbor_num,ori_0,ori_knn_neighbor,f_s=cal_feat,train=0,two_hop=self.two_hop)
        near_neigh,merge_idxs=self.point_level_merge_split(t_indexes,all_pred,point_neighbor,memory)

        # if not self.two_hop:
        #     point_neighbor=torch.cat((point_neighbor,-1+torch.zeros((point_neighbor.size(0),self.split_lp.split_num)).long().cuda()),dim=-1)
        #     assert point_neighbor.size(1)==self.neighbor_num+self.split_lp.split_num

        if self.jaccard_debug !=1:
            # sub
            # split sub cluster 1-->2 split gcn
            point_neighbor = self.split_lp(t_indexes, memory.features, memory.t_sub_label, sub_level=1,
                                           ori_knn_neighbor=point_neighbor, two_hop=self.two_hop, point_pred=all_pred)

            all_gt_sub_label=memory.t_sub_label[point_neighbor.view(-1)].view(len(t_indexes),-1)
            #self.split_gcn(t_indexes,memory.features,memory.t_sub_label,0,sub_label=1)
            #all_pred_sub,self.sub_sum,self.sub_num,sub_mapping_0,sub_mapping_1=self.sub_cluster_level_lp(t_indexes,memory.features,self.neighbor_num,ori_0,ori_knn_neighbor,f_s=cal_feat,sub_label=memory.t_sub_label,gt_sub_label=all_gt_sub_label,debug_label=self.debug_label,bias=self.thre[1])
            all_pred_sub,sub_sum,_,sub_mapping_0,sub_mapping_1,sub_mapping_2=self.cluster_level_lp(t_indexes,memory.features,self.neighbor_num,ori_0,point_neighbor,f_s=cal_feat,labels=memory.t_sub_label,gt_label=all_gt_sub_label,
                                                                                                     debug_label=self.debug_label,bias=self.thre[1],step=1,point_W=point_W,two_hop=self.two_hop,memory=memory,point_pred=all_pred)
            self.sub_cluster_level_merge_split(t_indexes,memory,all_pred_sub,point_neighbor,sub_mapping_0,sub_mapping_1,sub_mapping_2,near_neigh,merge_idxs)
        else:
            sub_sum=memory.features
        #clu level pred
        #outliers
        empty_label=set(torch.arange(memory.labels.max()+1).tolist())-set(memory.labels.tolist())
        if len(empty_label)<self.split_lp.split_num*len(t_indexes)+1:
            outliers_label=torch.arange(memory.labels.max()+1,memory.labels.max()+2+self.split_lp.split_num*len(t_indexes)).cuda()
        else:
            empty_label=list(empty_label)[-(1+self.split_lp.split_num*len(t_indexes)):]
            outliers_label=torch.tensor(empty_label).cuda()

        point_neighbor=self.split_lp(t_indexes,sub_sum,memory.labels,sub_level=0,sub_labels=memory.t_sub_label,outliers_label=outliers_label,ori_knn_neighbor=point_neighbor,memory=memory,two_hop=self.two_hop,point_pred=all_pred,point_W=point_W)
        all_gt_label=memory.labels[point_neighbor.view(-1)].view(len(t_indexes),-1)
        #self.split_gcn(t_indexes,memory.features,memory.labels,0,sub_label=0,outliers_label=outliers_label[-len(t_indexes):],sub_labels=memory.t_sub_label)
        all_pred_clu,_,_,clu_mapping_0,clu_mapping_1,clu_mapping_2=self.cluster_level_lp(t_indexes,memory.features,self.neighbor_num,ori_0,point_neighbor,f_s=cal_feat, labels=memory.labels,gt_label=all_gt_label,
                                                                                         debug_label=self.debug_label,bias=self.thre[2],step=2,point_W=point_W,two_hop=self.two_hop,memory=memory,point_pred=all_pred)

        self.cluster_level_merge_split(t_indexes,memory,all_pred_clu,point_neighbor,clu_mapping_0,clu_mapping_1,clu_mapping_2,near_neigh,merge_idxs)
        #import pdb;pdb.set_trace()
        #cluster acc
        # accs=[]
        # for i in range(len(t_indexes)):
        #     batch_lab=self.debug_label[memory.labels[memory.source_classes:]==memory.labels[t_indexes[i]]]
        #     if len(batch_lab)>3:
        #         acc=1.0*torch.sum(batch_lab==self.debug_label[t_indexes[i]-memory.source_classes])/len(batch_lab)
        #         accs.append('[{}] {:.2f} {}/{}'.format(t_indexes[i].item(),acc, len(batch_lab), int(
        #             self.debug_label_num[(self.debug_label[t_indexes[i] - memory.source_classes]).item()])))
        # print('after acc:',accs)
        if self.jaccard_debug !=1:
            del sub_mapping_0,sub_mapping_1,all_pred_sub
        del clu_mapping_0,clu_mapping_1,all_pred_clu,sub_sum,point_neighbor,ori_knn_neighbor

    def point_level_merge_split(self,indexes,all_pred,ori_knn_neighbor,memory):
        topk=10 # indicate the chaos
        bias=self.thre[0]
        conf,near_nei=torch.max(all_pred[:,0],dim=1)
        near_neig=ori_knn_neighbor[torch.arange(len(indexes)),near_nei]
        #bias=all_pred[:,0,-1]
        #merge
        merge_idx=indexes[(near_nei<topk) & (conf>bias) & (near_neig>=memory.source_classes)] #wo consider source domain
        merge_nei=near_neig[(near_nei<topk) & (conf>bias) & (near_neig>=memory.source_classes)].long()

        itera = len(set(near_neig.tolist()) & set(indexes.tolist()))#fix bug
        # if itera>1:
        #     print('--------itera:{}-------'.format(itera))
        for i in range(itera + 1):
            memory.t_sub_label[merge_idx] = memory.t_sub_label[merge_nei]
            memory.labels[merge_idx] = memory.labels[merge_nei]

        if self.merge_wo_outlier:
            unq_lab,unq_cnt=np.unique(memory.labels.cpu().numpy(),return_counts=True)
            self.outlier_clu=set(unq_lab[unq_cnt<2].tolist())
        #outlier-->keep ori label
        #print('outliers num:',len(indexes)-len(merge_idx))
        #print('outliers:',list(set(indexes.tolist())-set(merge_idx.tolist())))
        return near_neig,merge_idx

    def sub_cluster_level_merge_split(self,indexes,memory,all_pred_sub,ori_knn_neighbor,sub_mapping_0,sub_mapping_1,sub_mapping_2,near_neighbor,merge_idxs):
        #bias=all_pred_sub[:,0,-1]
        bias=self.thre[1]
        sub_lab=sub_mapping_0
        #lab=memory.labels[sub_lab.view(-1)].view(len(indexes),-1)
        #lab=memory.labels[ori_knn_neighbor.view(-1)].view(len(indexes),-1)

        #####merge
        merge_map={}
        for i in range(len(indexes)):
            if self.merge_wo_outlier and indexes[i] not in merge_idxs:
                continue
            #import pdb;pdb.set_trace()
            # keep_idx=ori_knn_neighbor[i][sub_mapping_2[i]]
            # if torch.min(keep_idx)==-1:
            #     print('-1')
            #     import pdb;pdb.set_trace()
            # lab=memory.labels[keep_idx]
            lab=memory.labels[sub_lab[i]]
            merge_idx=set(sub_lab[i][(all_pred_sub[i,0,:len(sub_lab[i])]>bias) & (lab==memory.labels[indexes[i]])].tolist())
            merge_idx.add(memory.t_sub_label[indexes[i]].item())
            merge_idx=list(merge_idx)
            if memory.t_sub_label[near_neighbor[i]].item() not in merge_idx: #reliable neighbor
                continue
            if len(merge_idx)>1:
                merge_label_0=-1
                inter=set(merge_idx) & set(merge_map.keys())
                if len(inter)>0: #
                    inter_label=list(inter)
                    merge_label_0=merge_map[inter_label[0]]
                    if len(inter_label)>1:
                        change_guys=[]
                        for label in inter_label:
                            change_guys.append(merge_map[label])
                        for change_label,update_label in merge_map.items():
                            if update_label in change_guys:
                                merge_map[change_label]=merge_label_0
                merge_label_0=merge_idx[0] if merge_label_0==-1 else merge_label_0
                for label in merge_idx:
                    merge_map[label]=merge_label_0
        for change_label,update_label in merge_map.items():
            memory.t_sub_label[memory.t_sub_label==int(change_label)]=int(update_label)
        print('sub merge:',len(merge_map))

        #split cluster 1-->2 split gcn

    def cluster_level_merge_split(self,indexes,memory,all_pred_clu,ori_knn_neighbor,clu_mapping_0,clu_mapping_1,clu_mapping_2,near_neighbor,merge_idxs):
        #bias=all_pred_clu[:,0,-1]
        #lab=memory.labels[ori_knn_neighbor.view(-1)].view(len(indexes),-1)
        bias=self.thre[2]
        lab=clu_mapping_0

        #####merge
        merge_map={}
        for i in range(len(indexes)):
            if self.merge_wo_outlier and indexes[i] not in merge_idxs: #only consider merge idx as core
                continue
            merge_idx=set(lab[i][(all_pred_clu[i,0,:len(lab[i])]>bias) & (lab[i]>=memory.source_classes)].tolist())
            merge_idx.add(memory.labels[indexes[i]].item())
            if self.merge_wo_outlier:
                merge_idx=(merge_idx-self.outlier_clu)
            merge_idx=list(merge_idx)
            if memory.labels[near_neighbor[i]].item() not in merge_idx:
                continue

            # if len(merge_idx)>10:
            #     print('---->10-------')
            #     import pdb;pdb.set_trace()
            if len(merge_idx)>1:
                merge_label_0=-1
                inter=set(merge_idx) & set(merge_map.keys())
                if len(inter)>0: #
                    inter_label=list(inter)
                    merge_label_0=merge_map[inter_label[0]]
                    if len(inter_label)>1:
                        change_guys=[]
                        for label in inter_label:
                            change_guys.append(merge_map[label])
                        for change_label,update_label in merge_map.items():
                            if update_label in change_guys:
                                merge_map[change_label]=merge_label_0
                merge_label_0=merge_idx[0] if merge_label_0==-1 else merge_label_0
                for label in merge_idx:
                    merge_map[label]=merge_label_0
        for change_label,update_label in merge_map.items():
            memory.labels[memory.labels==int(change_label)]=int(update_label)
        print('clu merge:',len(merge_map))

    def postprocess(self,s_indexes,memory):
        self.utils.update_sub_cluster_label(s_indexes,memory)
        #step1 merge&split
        #step2 update sub cluster label-->src


#others
def p_lp(alpha, method,**kwargs):
    model = Point_Level_LP(alpha=alpha,method=method)
    model.cuda()
    #model = nn.DataParallel(model)
    return model

def s_lp(alpha,topk_num,method,**kwargs):
    model = Sub_Cluster_Level_LP(alpha=alpha,topk_num=topk_num,method=method)
    model.cuda()
    #model = nn.DataParallel(model)
    return model

def c_lp(alpha, topk_num,method,point_wei,**kwargs):
    model = Cluster_Level_LP(alpha=alpha,topk_num=topk_num,method=method,point_wei=point_wei)
    model.cuda()
    #model = nn.DataParallel(model)
    return model

def split_gcn(feature_dim, nhid,feature_size, source_classes,nclass=1, dropout=0.,cal_num=30,**kwargs):
    model=Split_GCN(feature_dim=feature_dim,
                  nhid=nhid,
                  feature_size=feature_size,
                  source_classes=source_classes,
                  nclass=nclass,
                  dropout=dropout,
                  cal_num=cal_num)
    model.cuda()
    return model

def split_lp(alpha,split_num,anchor_thre,**kwargs):
    model=Split_LP(
        alpha=alpha,
        split_num=split_num,
        anchor_thre=anchor_thre
    )
    model.cuda()
    return model

class Utils(object): #in order to update * in memory
    def __init__(self,k1,k2,thre):
        self.k1=k1
        self.k2=k2
        self.thre=thre

        self.density_sim=0.5
        self.density_core_thre=0.7 #point sim>thre-->the same core

    def initialize_sub_cluster_label(self,label,sub_cluster_label,features,start=0):
        print('initialize sub cluster bank...')
        if len(features)>20000:
            tmp=features.cpu()
            sim=tmp.mm(tmp.t())
            density=torch.sum(torch.gt(sim,self.density_sim),dim=1).cuda()
            density_core=torch.gt(sim,self.density_core_thre).cuda()
        else:
            sim=features.mm(features.t())
            density=torch.sum(torch.gt(sim,self.density_sim),dim=1)
            density_core=torch.gt(sim,self.density_core_thre)
        all_idx=torch.arange(len(label)).cuda()

        unique_label=list(set(label.tolist()))
        for un_idx,i in enumerate(unique_label):
            if un_idx%100==0:
                print('[{}/{}]'.format(un_idx,len(unique_label)))
            i_idx=(all_idx[label==i])
            i_density=density[i_idx]
            if torch.sum(torch.ge(i_density,self.thre))>0:
                #find connection
                high_density_idx=i_idx[i_density>=self.thre]
                i_density_core=density_core[high_density_idx][:,high_density_idx]

                sub_cluster_label[i_idx]=-1 #clean
                #core
                for left_id in range(len(high_density_idx)):
                    neighbor=high_density_idx[i_density_core[left_id]>0]
                    if len(neighbor)>1:
                        neighbor_label=sub_cluster_label[neighbor]
                        neighbor_label=neighbor_label[neighbor_label>-1]
                        if len(neighbor_label)>0:
                            sub_cluster_label[high_density_idx[left_id].item()]=neighbor_label[0].item()
                        else:
                            sub_cluster_label[high_density_idx[left_id].item()]=high_density_idx[left_id].item()
                    else:
                        sub_cluster_label[high_density_idx[left_id].item()]=high_density_idx[left_id].item()
                #others
                i_sim=sim[i_idx][:,i_idx]
                i_sim[:,i_density<self.thre]=-1
                match=torch.argmax(i_sim,dim=1)
                sub_cluster_label[i_idx]=sub_cluster_label[i_idx[match]]+start
            else:
                match=torch.argmax(i_density)
                sub_cluster_label[i_idx]=i_idx[match]+start

        assert torch.min(sub_cluster_label)>=0

    def initialize_sub_cluster_label_ori(self,label,sub_cluster_label,features,start=0): #initialize
        print('initialize sub cluster bank...')
        #compute density
        sim=features.mm(features.t())
        density=torch.sum(torch.gt(sim,self.density_sim),dim=1)
        print('high density num:',torch.sum(density>self.thre))
        #combine_density=torch.gt(sim,self.density_combine_thre)

        rerank_dist = torch.from_numpy(compute_jaccard_distance_inital_rank(features, k1=self.k1, k2=self.k2)).cuda()
        all_idx=torch.arange(len(label))
        unique_label=list(set(label.tolist()))
        for i in unique_label:
            #import pdb;pdb.set_trace()
            i_rerank_dist=rerank_dist[label==i][:,label==i]
            #i_combine=combine_density[label==i][:,label==i]
            i_density=density[label==i]
            i_features=features[label==i]
            i_idx=(all_idx[label==i])
            if torch.sum(torch.ge(i_density,self.thre))>0: #have high density guys
                i_rerank_dist[:,torch.lt(i_density,self.thre)]=1
                match=torch.argmin(i_rerank_dist,dim=1)
                try:
                    sub_cluster_label[label==i]=(i_idx[match]).cuda()+start
                except:
                    print('sub_cluster_labe error')
                    import pdb;pdb.set_trace()
                #sub_cluster_featurebank[label==i]=features[label==i][match]
            else: #all low density-->one sub cluster
                match=torch.argmax(i_density)
                try:
                    sub_cluster_label[label==i]=(i_idx[match]).cuda()+start
                except:
                    print('sub_cluster_label single error')
                    import pdb;pdb.set_trace()
                #sub_cluster_featurebank[label==i]=i_idx[match]

        assert torch.max(sub_cluster_label)-start<len(sub_cluster_label)
        del sim,rerank_dist
        print('Done')

    def update_sub_cluster_label(self,indexes,memory): #update online
        index_label=list(set(memory.s_label[indexes].tolist()))
        all_idx=torch.arange(len(memory.s_label)).cuda()
        #update sub label for these lael
        core_nums=[]
        for label in index_label:
            i_idx=all_idx[(memory.s_label==label)]
            feat=memory.s_features[i_idx]
            sim=feat.mm(feat.t())
            #density
            i_density=torch.sum(torch.gt(sim,self.density_sim),dim=1)
            if torch.sum(torch.ge(i_density,self.thre))>0:
                i_density_core=torch.gt(sim,self.density_core_thre)
                high_density_idx=i_idx[i_density>=self.thre]
                i_density_core=i_density_core[i_density>=self.thre][:,i_density>=self.thre]

                memory.s_sub_label[i_idx]=-1

                for left_id in range(len(high_density_idx)):
                    neighbor=high_density_idx[i_density_core[left_id]>0]
                    if len(neighbor)>1:
                        neighbor_label=memory.s_sub_label[neighbor]
                        neighbor_label=neighbor_label[neighbor_label>-1]
                        if len(neighbor_label)>0:
                            memory.s_sub_label[high_density_idx[left_id].item()]=neighbor_label[0].item()
                        else:
                            memory.s_sub_label[high_density_idx[left_id].item()]=high_density_idx[left_id].item()
                    else:
                        memory.s_sub_label[high_density_idx[left_id].item()]=high_density_idx[left_id].item()

                #others
                sim[:,i_density<self.thre]=-1
                match=torch.argmax(sim,dim=1)
                memory.s_sub_label[i_idx]=memory.s_sub_label[i_idx[match]]
                core_nums.append(len(set(memory.s_sub_label[i_idx].tolist())))
            else:
                match=torch.argmax(i_density)
                memory.s_sub_label[i_idx]=i_idx[match]
                core_nums.append(1)

        #print('core_nums:',core_nums)
