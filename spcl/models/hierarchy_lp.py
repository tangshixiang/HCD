#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
from collections import defaultdict

from spcl.models.utils import GraphConv, MeanAggregator
from spcl.utils.faiss_rerank import compute_jaccard_distance,compute_jaccard_distance_step1,compute_jaccard_distance_inital_rank,compute_knn

class Point_Level(nn.Module):
    def __init__(self):
        super(Point_Level, self).__init__()
        #self.loss=torch.nn.CrossEntropyLoss().cuda()
        self.eps = np.finfo(float).eps
        self.alpha=0.99
        self.topk_num=10
    def forward(self,indexes,features,label,ori_knn_neighbor,neighbor_num,batch_neighbor,W):
        bs=len(indexes)
        #find topk neighbor-->cluster
        ##########normalize#############
        W *= (1-torch.eye(batch_neighbor.size(1))).unsqueeze(0).expand_as(W).cuda()#diag-->0

        topk, indices = torch.topk(W, self.topk_num,dim=2)
        mask_top = torch.zeros_like(W)
        mask_top = mask_top.scatter(2, indices, 1)
        mask_top = ((mask_top>0)&(mask_top.permute((0,2,1))>0)).type(torch.float32)
        W0=W*mask_top

        D = W0.sum(1)
        D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
        D1      = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,batch_neighbor.size(1))
        D2      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,batch_neighbor.size(1),1)
        S      = D1*W0*D2
        ###############################

        Y=torch.eye(batch_neighbor.size(1)).unsqueeze(0).expand_as(W).cuda()
        Y[:,0]=0 #not include self
        preds  = (1-self.alpha)*torch.matmul(torch.inverse(torch.eye(batch_neighbor.size(1)).unsqueeze(0).expand_as(W).cuda()-self.alpha*S+self.eps), Y)
        return preds

    def forward_v2(self,indexes,features,label,ori_knn_neighbor,neighbor_num,batch_neighbor,W):#simplify jaccard distance
        topk, indices = torch.topk(W, self.topk_num,dim=2)
        mask_top = torch.zeros_like(W)
        mask_top = mask_top.scatter(2, indices, 1)
        mask_top = ((mask_top>0)&(mask_top.permute((0,2,1))>0)).type(torch.float32)
        mask_top[W==0]=0
        W0=W*mask_top
        #softmax
        W0/=torch.sum(W0+self.eps,dim=2,keepdim=True)
        W0=W0.cpu().numpy()
        indices=indices.cpu().numpy()

        index_jacc=np.zeros((len(indexes),batch_neighbor.size(1)))
        for i in range(len(indexes)):
            temp_min=np.zeros((1,batch_neighbor.size(1)))
            indNonZero=indices[i][0]
            for j in range(len(indNonZero)):
                temp_min[0,indices[i][indNonZero[j]]]=temp_min[0,indices[i][indNonZero[j]]]+np.minimum(W0[i,0,indNonZero[j]],W0[i,indices[i][indNonZero[j]],indNonZero[j]])
            index_jacc[i] = 1 - temp_min / (2.0 - temp_min)
        #import pdb;pdb.set_trace()
        return torch.from_numpy(index_jacc)

class Merge_LP(nn.Module):
    def __init__(self, alpha,topk_num,temp):
        super(Merge_LP, self).__init__()
        self.alpha=alpha
        self.eps = np.finfo(float).eps
        self.only_consider_once=1
        self.topk_num=topk_num
        self.w_use_dist=1
        self.temp=temp
        self.num_penalty=0

    def forward(self, indexes,batch_feat,cen_feat,label,batch_neighbor,ori_knn_neighbor,S,neighbor_num,step=1,nums=None):
        bs=len(indexes)
        #cal Y
        Y=torch.zeros((bs,batch_neighbor.size(1),batch_neighbor.size(1))).cuda() #neighbor_num-->max cluster num
        labs,inverses=[],[]
        # #debug
        # for i in range(bs):
        #     label_cpu=label.cpu()
        #     batch_neighbor_cpu=batch_neighbor.cpu()
        #     try:
        #         lab,inverse=torch.unique(label_cpu[batch_neighbor_cpu[i][batch_neighbor_cpu[i]>=0]], return_inverse=True)
        #     except:
        #         print('error')
        #         import pdb;pdb.set_trace()
        # ###########
        for i in range(bs):
            lab,inverse=torch.unique(label[batch_neighbor[i][batch_neighbor[i]>=0]], return_inverse=True)
            lab_cen=cen_feat[lab]
            sim=batch_feat[i].mm(lab_cen.t())
            Y[i,torch.arange(len(inverse)),inverse]=torch.exp(sim[torch.arange(len(inverse)),inverse]/self.temp)
            labs.append(lab)
            inverses.append(inverse)
        #softmax
        Y/=torch.sum(Y+self.eps,dim=1,keepdim=True)#norm
        if self.num_penalty:
            for i in range(bs):
                Y[i, :, :len(labs[i])] /= torch.exp(nums[labs[i]].view(-1).float() / 500).unsqueeze(0).expand_as(Y[i, :, :len(labs[i])])
        preds  = (1-self.alpha)*torch.matmul(torch.inverse(torch.eye(batch_neighbor.size(1)).unsqueeze(0).expand_as(S).cuda()-self.alpha*S+self.eps), Y)
        return preds,labs,inverses,Y

    def forward_v2(self, indexes,batch_feat,cen_feat,sims,label,batch_neighbor,ori_knn_neighbor,S,neighbor_num,step=1,max_num=-1):
        bs = len(indexes)
        # cal Y
        Y = torch.zeros((bs, batch_neighbor.size(1), batch_neighbor.size(1))).cuda()  # neighbor_num-->max cluster num
        labs, inverses = [], []

        for i in range(bs):
            lab, inverse,cnts = torch.unique(label[batch_neighbor[i][batch_neighbor[i] >= 0]], return_inverse=True,return_counts=True)
            lab_cen = cen_feat[lab]
            Y[i,:len(inverse),:len(lab)]=torch.exp(batch_feat[i,:len(inverse)].mm(lab_cen.t())/self.temp)
            if self.num_penalty:
                Y[i,:,:len(lab)]/=torch.exp(cnts.float()/500).unsqueeze(0).expand_as(Y[i,:,:len(lab)])
            labs.append(lab)
            inverses.append(inverse)
        Y/=torch.sum(Y+self.eps,dim=-1,keepdim=True)
        preds = (1 - self.alpha) * torch.matmul(torch.inverse(
            torch.eye(batch_neighbor.size(1)).unsqueeze(0).expand_as(S).cuda() - self.alpha * S + self.eps), Y)

        return preds,labs,inverses,Y

class Split_LP(nn.Module):
    def __init__(self,alpha,split_num,confidence_thre=0.7,temp=0.05):
        super(Split_LP, self).__init__()
        self.alpha=alpha
        self.eps = np.finfo(float).eps
        self.method=2 #random walk
        self.split_num=split_num
        self.confidence_thre=confidence_thre
        self.temp=temp
    def forward(self,indexes,features,nums,sims,labels,sub_level=0,sub_labels=None,outliers_label=None,memory=None):
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
        elif self.method==1:#few shot D
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
                    split_idxs=list(set((batch_sub_label).tolist())) #sub label
                    if len(split_idxs)<self.split_num:
                        continue
                    anchor_idxs=[sub_labels[idx]]
                    anchor_indices=[]

                    split_feat=features[split_idxs]
                    split_nums=nums[split_idxs]
                    #confidence check###
                    cen_feat = torch.sum((split_nums.unsqueeze(1) * sims[split_idxs]), dim=0) / torch.sum(split_nums)
                    if torch.sum(memory.features[idx] * cen_feat) >= self.confidence_thre:
                        #print('-----------------')
                        continue
                    ###################

                    split_sim=features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
                    anchor_idx=split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(torch.argmax(split_sim).item())#self
                    anchor_indices.append(torch.argmin(split_sim).item())

                    for sp in range(2,self.split_num):
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

                    #labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                    labs=outliers_label[torch.arange(n,len(outliers_label),step=len(indexes))]
                    for sub,pre in zip(split_idxs,pred):
                        labels[batch_idx[batch_sub_label==sub]]=labs[pre]
                    if print_cnts==0:
                        print('split:',len(set(pred.tolist())))
                        print_cnts=1
                    #split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
        elif self.method==2:
            print_cnts=0
            for n,idx in enumerate(indexes):
                batch_idx=all_idxs[labels==labels[idx]]
                batch_sub_label=sub_labels[batch_idx]
                split_idxs=list(set((batch_sub_label).tolist())) #sub label
                if len(split_idxs)<self.split_num:
                    continue
                anchor_idxs=[sub_labels[idx]]
                anchor_indices=[]

                split_feat=features[split_idxs]
                split_nums=nums[split_idxs]
                #confidence check###sim -->last iteration
                cen_feat=torch.sum((split_nums.unsqueeze(1)*sims[split_idxs]),dim=0)/torch.sum(split_nums)
                if torch.sum(memory.features[idx]*cen_feat)>=self.confidence_thre:
                    print('-----------------')
                    continue
                ###################

                split_sim=features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
                anchor_idx=split_idxs[torch.argmin(split_sim)]
                anchor_idxs.append(anchor_idx)
                anchor_indices.append(torch.argmax(split_sim).item()) #self
                anchor_indices.append(torch.argmin(split_sim).item())

                for sp in range(2,self.split_num):
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

                # Y=torch.zeros((len(split_idxs),self.split_num)).cuda()
                # Y[anchor_indices,torch.arange(self.split_num)]=1
                #import pdb;pdb.set_trace()

                W=torch.exp(split_feat.mm(split_feat.t()))
                #W=torch.exp(-(2-2*W))
                mask=(1-torch.eye(len(split_feat))).cuda()
                W*=mask
                S=W/torch.sum(W+self.eps,dim=-1,keepdim=True)

                Y=torch.exp(split_feat.mm(split_feat[anchor_indices].t())/self.temp)
                Y/=torch.sum(Y+self.eps,dim=-1,keepdim=True)

                # D       = W.sum(0)
                # D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
                # D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,len(split_idxs))
                # D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(len(split_idxs),1)
                # S       = D1*W*D2

                pred  = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda()-self.alpha*S+self.eps), Y)
                pred=torch.argmax(pred,dim=1)

                #labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
                labs=outliers_label[torch.arange(n,len(outliers_label),step=len(indexes))]
                for sub,pre in zip(split_idxs,pred):
                    labels[batch_idx[batch_sub_label==sub]]=labs[pre]
                if print_cnts==0:
                    print('split:',len(set(pred.tolist())))
                    print_cnts=1
class Hierarchy_LP(object):
    def __init__(self,point_level,merge_lp,split_lp,utils,neighbor_num=64,debug_label=[],merge_wo_outlier=0,iterative=2,
                 start_label_num=1,merge_method=1,merge_type=1,point_bias=0.7,method5_ratio=0.8):
        self.point_level=point_level
        self.split_lp=split_lp
        self.merge_lp=merge_lp
        self.utils=utils #utils

        self.neighbor_num=neighbor_num #knn
        self.debug_label=debug_label
        self.merge_wo_outlier=merge_wo_outlier
        self.iterative=iterative # num of iteration

        self.eps = np.finfo(float).eps
        self.topk_num=15#for iterative lp
        self.start_label=torch.arange(start_label_num).long().cuda()
        self.max_keep_num=40 #for local region
        self.merge_method=merge_method
        self.merge_type=merge_type
        self.point_bias=point_bias
        self.iterative_bias=[0.02,0.02]
        self.old_method=1 #few shot D
        self.method5_ratio=method5_ratio
        if self.old_method:
            self.split_lp.method=1

        self.low_confidence_filter=-1

    def inference(self,t_indexes,memory,infer):
        #cal knn
        #cal_feat=memory.features[t_indexes]
        ori_0=compute_knn(memory.features,k1=self.neighbor_num)
        ori_knn_neighbor=torch.from_numpy(ori_0[t_indexes.cpu().numpy(),:]).cuda()

        #batch-->outlier
        for iter in range(self.iterative):
            memory.labels[iter][t_indexes]=self.empty_label(memory.labels[iter],len(t_indexes))
        ##point level
        batch_neighbor,batch_max_num=self.get_local_region(t_indexes,memory.labels[-1],ori_knn_neighbor)

        ######W init######
        feat=memory.features[batch_neighbor.view(-1)].view(len(t_indexes),-1,2048)
        feat[batch_neighbor<0]=0
        W=feat.bmm(feat.permute((0,2,1)))
        #####for jaccard distance###############
        W_point=torch.exp(-(2-2*W))
        W_point[batch_neighbor.unsqueeze(2).expand_as(W_point)<0]=0
        W_point[batch_neighbor.unsqueeze(1).expand_as(W_point)<0]=0
        all_pred=self.point_level.forward_v2(t_indexes,memory.features,memory.labels[-1],ori_knn_neighbor,self.neighbor_num,batch_neighbor,W_point)
        #all_pred=self.point_level(t_indexes,memory.features,memory.labels[-1],ori_knn_neighbor,self.neighbor_num,batch_neighbor,W_point)
        #near_neigh,merge_idxs=self.point_level_merge_split(t_indexes,all_pred,batch_neighbor,memory)
        near_neigh,merge_idxs=self.point_level_merge_split_v2(t_indexes,all_pred,batch_neighbor,memory)

        del W_point
        ###################iterative part#######################
        W=torch.exp(W)
        W[batch_neighbor.unsqueeze(2).expand_as(W)<0]=0
        W[batch_neighbor.unsqueeze(1).expand_as(W)<0]=0
        W *= (1-torch.eye(batch_neighbor.size(1))).unsqueeze(0).expand_as(W).cuda()

        topk, indices = torch.topk(W, self.topk_num,dim=2)
        mask_top = torch.zeros_like(W)
        mask_top = mask_top.scatter(2, indices, 1)
        mask_top = ((mask_top>0)&(mask_top.permute((0,2,1))>0)).type(torch.float32) #intersection
        #mask_top = ((mask_top>0)| (mask_top.permute((0,2,1))>0)).type(torch.float32) #union
        W0 = W * mask_top

        if self.old_method:
             #version1 d-1/2Wd-1/2
            D = W0.sum(1)
            D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
            D1      = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,batch_neighbor.size(1))
            D2      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,batch_neighbor.size(1),1)
            S      = D1*W0*D2
        else:
            #version2 softmax
            S=W0/torch.sum(W0+self.eps,dim=-1,keepdim=True)

        level_feat=[memory.features]#level0
        level_nums=[torch.ones(len(memory.features)).cuda()]
        level_sims=[memory.features]#feat wo norm
        for iter in range(self.iterative):
            #split
            outlier_label=self.empty_label(memory.labels[iter],len(t_indexes)*self.split_lp.split_num)
            sub_labels=self.start_label if iter==0 else memory.labels[iter-1]
            self.split_lp(t_indexes,level_feat[-1],level_nums[-1],level_sims[-1],memory.labels[iter],0,sub_labels=sub_labels,
                          outliers_label=outlier_label,memory=memory)

            #cur level feat
            level_sum = torch.zeros(memory.labels[iter].max()+1, 2048).float().cuda()
            level_sum.index_add_(0, memory.labels[iter], memory.features)
            nums = torch.zeros(memory.labels[iter].max()+1, 1).float().cuda()
            nums.index_add_(0, memory.labels[iter], torch.ones(len(memory.labels[iter]),1).float().cuda())
            mask = (nums>0).float()
            level_sum /= (mask*nums+(1-mask)).clone().expand_as(level_sum)
            level_sims.append(level_sum.clone())# wo norm-->avg linkage
            level_sum/=level_sum.norm(dim=1,keepdim=True) #sub feat
            level_feat.append(level_sum)
            level_nums.append(nums.view(-1))

            del level_feat[0],level_nums[0],level_sims[0]

            print('iterative {}: {}'.format(iter,nums.max().item()))

            #merge
            if self.old_method:
                pred,lab,inverse,Y=self.merge_lp(t_indexes,feat,level_sum,memory.labels[iter],batch_neighbor,ori_knn_neighbor,S,self.neighbor_num,nums=nums)
            else:
                pred, lab, inverse, Y = self.merge_lp.forward_v2(t_indexes, feat, level_sum, level_sims[-1],memory.labels[iter], batch_neighbor,
                                                  ori_knn_neighbor, S, self.neighbor_num,
                                                  max_num=min(batch_max_num, self.neighbor_num * self.max_keep_num))
            next_label=memory.labels[iter+1] if iter<self.iterative-1 else None
            if self.merge_method==1:
                self.merge(t_indexes,pred,memory.labels[iter],lab,inverse,batch_neighbor,next_label=next_label,source_class=memory.source_classes,merge_idxs=merge_idxs)
            elif self.merge_method==2:
                self.merge_v2(t_indexes,pred,memory.labels[iter],lab,inverse,batch_neighbor,next_label=next_label,
                            source_class=memory.source_classes,merge_idxs=merge_idxs,nums=nums,sims=level_sims[-1],memory=memory)
            elif self.merge_method==3:
                self.merge_v3(t_indexes, pred, memory.labels[iter], lab, inverse, batch_neighbor, next_label=next_label,
                              source_class=memory.source_classes, merge_idxs=merge_idxs)
            elif self.merge_method==4:
                self.merge_v4(t_indexes, pred, memory.labels[iter], lab, inverse, batch_neighbor, next_label=next_label,
                              source_class=memory.source_classes, merge_idxs=merge_idxs,bias=self.iterative_bias[iter])
            elif self.merge_method==5:
                self.merge_v5(t_indexes,pred,memory.labels[iter],lab,inverse,batch_neighbor,next_label=next_label,
                            source_class=memory.source_classes,merge_idxs=merge_idxs,nums=nums,sims=level_sims[-1],memory=memory)
            elif self.merge_method==6:
                self.merge_v6(t_indexes, pred, memory.labels[iter], lab, inverse, batch_neighbor, next_label=next_label,
                              source_class=memory.source_classes, merge_idxs=merge_idxs, nums=nums, sims=level_sims[-1],memory=memory,bias=self.iterative_bias[iter])
        del level_feat[0],level_nums[0],level_sims[0]
        #cluster acc
        accs=[]
        for i in range(len(t_indexes)):
            batch_lab=self.debug_label[memory.labels[-1][memory.source_classes:]==memory.labels[-1][t_indexes[i]]]
            if len(batch_lab)>3:
                acc=1.0*torch.sum(batch_lab==self.debug_label[t_indexes[i]-memory.source_classes])/len(batch_lab)
                accs.append('{:.2f} {}'.format(acc,len(batch_lab)))
        print('acc:',accs)
        del W,mask_top,S,D,W0

    def get_local_region(self,t_indexes,label,ori_knn_neighbor):
        all_neighbor=[]
        all_idx=torch.arange(len(label)).cuda()
        max_num=-1
        min_num=1000
        all_num=[]
        all_neis=[]
        tmp=label.cpu().numpy()
        batch_label=(label[ori_knn_neighbor.view(-1)].tolist())
        batch_all_neighbor=np.where(np.isin(tmp,batch_label)==1)[0]
        batch_all_label=tmp[batch_all_neighbor]
        for i in range(len(t_indexes)):
            all_neis.append(len(set(batch_label[i*ori_knn_neighbor.size(1)+1:(i+1)*ori_knn_neighbor.size(1)])))
            nei=batch_all_neighbor[np.where(np.isin(batch_all_label,batch_label[i*ori_knn_neighbor.size(1)+1:(i+1)*ori_knn_neighbor.size(1)]))]
            if len(nei)>200:#self.max_keep_num*self.neighbor_num:
                #print('-----filter----')
                lab,inver,cnts=np.unique(tmp[nei],return_inverse=True,return_counts=True)
                filter_inver=np.where(cnts>self.max_keep_num)[0]
                nei_filter=set(nei.tolist())
                for filter_lab in filter_inver:
                    filter= np.random.choice(nei[np.where(inver==filter_lab)[0]], size=cnts[filter_lab]-self.max_keep_num, replace=False)
                    nei_filter=nei_filter-set(filter.tolist())
                nei_filter= (nei_filter | set(ori_knn_neighbor[i,1:].tolist()))
                nei=np.array(list(nei_filter))
            if len(nei)>max_num:
                max_num=len(nei)
            if len(nei)<min_num:
                min_num=len(nei)
            all_neighbor.append(nei)

            #all_num.append(len(all_neighbor[-1]))
        all_nei=torch.zeros((len(t_indexes),max_num+1)).long()-1
        for idx,n in enumerate(all_neighbor):
            all_nei[idx,1:len(n)+1]=torch.from_numpy(n)
        all_nei=all_nei.cuda()
        all_nei[:,0]=t_indexes #self
        #debug
        print('cluster num:',all_neis)
        print('max num:{} min num: {}'.format(max_num,min_num))
        return all_nei,max_num

    def point_level_merge_split(self,indexes,all_pred,batch_neighbor,memory):
        bias=0
        conf,near_nei=torch.max(all_pred[:,0],dim=1)
        near_neig=batch_neighbor[torch.arange(len(indexes)),near_nei]
        print('conf:',conf)
        #bias=all_pred[:,0,-1]
        #merge
        merge_idx=indexes[(conf>bias) & (near_neig>=memory.source_classes)] #wo consider source domain
        merge_nei=near_neig[(conf>bias) & (near_neig>=memory.source_classes)]

        itera=len(set(near_neig.tolist()) & set(indexes.tolist()))
        for iter in range(self.iterative):
            for i in range(itera+1):
                memory.labels[iter][merge_idx]=memory.labels[iter][merge_nei]
        #memory.t_sub_label[merge_idx]=memory.t_sub_label[merge_nei]
        #memory.labels[merge_idx]=memory.labels[merge_nei]
        #outlier-->keep ori label
        print('outliers num:',len(indexes)-len(merge_idx))
        return near_neig,merge_idx

    def point_level_merge_split_v2(self,indexes,all_pred,batch_neighbor,memory):
        bias=self.point_bias
        all_pred[:,0]=1#wo self
        conf,near_nei=torch.min(all_pred,dim=1)
        #print('conf:',conf)
        conf=conf.cuda()
        near_neig=batch_neighbor[torch.arange(len(indexes)),near_nei]
        #print('conf:',conf)
        #bias=all_pred[:,0,-1]
        #merge
        merge_idx=indexes[(conf<bias) & (near_neig>=memory.source_classes)] #wo consider source domain
        merge_nei=near_neig[(conf<bias) & (near_neig>=memory.source_classes)]
        for iter in range(self.iterative):
            memory.labels[iter][merge_idx]=memory.labels[iter][merge_nei]
        #memory.t_sub_label[merge_idx]=memory.t_sub_label[merge_nei]
        #memory.labels[merge_idx]=memory.labels[merge_nei]
        #outlier-->keep ori label
        print('outliers num:',len(indexes)-len(merge_idx))
        return near_neig,merge_idx

    def merge(self,indexes,pred,cur_label,lab,inverse,batch_neighbor,next_label=None,source_class=0,merge_idxs=None):
        max_idx=torch.argmax(pred,dim=2)

        merge_map={}
        #import pdb;pdb.set_trace()
        for i in range(len(indexes)):
            if self.merge_wo_outlier:
                if indexes[i] not in merge_idxs:
                    continue
            mapping_label=max_idx[i,:len(inverse[i])]
            merge_neighbor=batch_neighbor[i,:len(inverse[i])][mapping_label==inverse[i][0]]
            merge_neighbor=merge_neighbor[merge_neighbor>=source_class]#wo source domain
            if len(merge_neighbor)<=1:
                continue
            if next_label is not None:
                merge_neighbor=merge_neighbor[next_label[merge_neighbor]==next_label[indexes[i]]]
            merge_lab=list(set(cur_label[merge_neighbor].tolist()))
            if len(merge_lab)>1:#merge
                merge_label_0=-1
                inter=set(merge_lab) & set(merge_map.keys())
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
                merge_label_0=merge_lab[0] if merge_label_0==-1 else merge_label_0
                for label in merge_lab:
                    merge_map[label]=merge_label_0

        for change_label,update_label in merge_map.items():
            #memory.labels[memory.labels==int(change_label)]=int(update_label)
            cur_label[cur_label==int(change_label)]=int(update_label)
        print('merge:',len(merge_map))
        ##for debug
        # debug=[]
        # for i in range(5):
        #     mapping_label=max_idx[i,:len(inverse[i])]
        #     #print('mapping:',mapping_label[0].item(),inverse[i][0].item())
        #     debug.append([len(lab[i]),len(torch.unique(mapping_label))])
        # print(debug)

    def merge_v2(self,indexes,pred,cur_label,lab,inverse,batch_neighbor,next_label=None,source_class=0,merge_idxs=None,nums=None,sims=None,memory=None):
        max_idx=torch.argmax(pred,dim=2)

        merge_map={}
        change_num=[]#for debug
        debug=1
        for i in range(len(indexes)):
            if self.merge_wo_outlier:
                if indexes[i] not in merge_idxs:
                    continue
            mapping_label=max_idx[i,:len(inverse[i])]
            merge_lab=set([cur_label[indexes[i]].item()]) #add self

            changelabel_guys=batch_neighbor[i,:len(inverse[i])][mapping_label!=inverse[i]]
            if len(changelabel_guys)==0:
                continue
            change_num.append(len(changelabel_guys))

            changelabel_guys_lab=cur_label[changelabel_guys]
            changelabel_guys_mapping_lab=lab[i][mapping_label[mapping_label!=inverse[i]]]

            #filter low confidence cluster
            lab_sim=sims[changelabel_guys_lab].mm(memory.features[indexes[i]].unsqueeze(0).t()).view(-1)
            mapping_lab_sim=sims[changelabel_guys_mapping_lab].mm(memory.features[indexes[i]].unsqueeze(0).t()).view(-1)
            # lab_sim=torch.sum(sims[changelabel_guys_lab]*sims[changelabel_guys_lab]/torch.norm(sims[changelabel_guys_lab],dim=-1,keepdim=True),dim=-1)
            # mapping_lab_sim=torch.sum(sims[changelabel_guys_mapping_lab]*sims[changelabel_guys_mapping_lab]/torch.norm(sims[changelabel_guys_mapping_lab],dim=-1,keepdim=True),dim=-1)
            changelabel_guys_lab=changelabel_guys_lab[(lab_sim>self.low_confidence_filter) & (mapping_lab_sim>self.low_confidence_filter)]
            changelabel_guys_mapping_lab=changelabel_guys_mapping_lab[(lab_sim>self.low_confidence_filter) & (mapping_lab_sim>self.low_confidence_filter)]

            if debug and len(changelabel_guys_mapping_lab)>10:
                print(changelabel_guys_lab)
                print('sim:',lab_sim)
                print('nums:',nums[changelabel_guys_lab].view(-1))
                print(changelabel_guys_mapping_lab)
                print('nums:',nums[changelabel_guys_mapping_lab].view(-1))
                #import pdb;pdb.set_trace()
                debug=0
            new_mapping=defaultdict(list)
            #for ori,new in zip(changelabel_guys_lab,changelabel_guys_mapping_lab):
            #    new_mapping[ori.item()].append(new.item())
            #    new_mapping[new.item()].append(ori.item())
            if self.merge_type==1:#type 1 union
                for ori,new in zip(changelabel_guys_lab,changelabel_guys_mapping_lab):
                    if ori>=source_class and new>=source_class:
                        new_mapping[ori.item()].append(new.item())
                        new_mapping[new.item()].append(ori.item())
            if self.merge_type==2:#type 2 one direction
                for ori,new in zip(changelabel_guys_lab,changelabel_guys_mapping_lab):
                    if ori>=source_class and new>=source_class:
                        new_mapping[ori.item()].append(new.item())
            if self.merge_type==3:#type 3 one direction inv
                for ori,new in zip(changelabel_guys_lab,changelabel_guys_mapping_lab):
                    if ori>=source_class and new>=source_class:
                        new_mapping[new.item()].append(ori.item())
            if self.merge_type==4:#type 4 mutual
                #import pdb;pdb.set_trace()
                filter_lab=list(set(changelabel_guys_lab.tolist()) & set(changelabel_guys_mapping_lab.tolist()))
                if len(filter_lab)>2:
                    lab_cnts=defaultdict(int)
                    for ori,new in zip(changelabel_guys_lab,changelabel_guys_mapping_lab):
                        if ori>=source_class and new>=source_class:
                            lab_cnts['{} {}'.format(ori.item(),new.item())]+=1
                            lab_cnts['{} {}'.format(new.item(),ori.item())]+=1
                            if lab_cnts['{} {}'.format(ori.item(),new.item())]>=2:
                                new_mapping[ori.item()].append(new.item())
                                new_mapping[new.item()].append(ori.item())
            #cal confidence

            length=0
            old_inter=set([])
            while(length<len(merge_lab)):
                length=len(merge_lab)
                inter= (merge_lab & set(new_mapping.keys()))-(old_inter)
                if len(inter)>0:
                    for inter_lab in inter:
                        merge_lab=(merge_lab) | (set(new_mapping[inter_lab]))
                old_inter=(old_inter | inter)

            if len(merge_lab)>1:#merge
                merge_lab=list(merge_lab)
                merge_label_0=-1
                inter=set(merge_lab) & set(merge_map.keys())
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
                merge_label_0=merge_lab[0] if merge_label_0==-1 else merge_label_0
                for label in merge_lab:
                    merge_map[label]=merge_label_0

        for change_label,update_label in merge_map.items():
            #memory.labels[memory.labels==int(change_label)]=int(update_label)
            cur_label[cur_label==int(change_label)]=int(update_label)
        if len(merge_map)>0:
            print('merge:',len(merge_map))
        print('change num:',change_num)
        ##for debug
        #debug=[]
        # for i in range(5):
        #     mapping_label=max_idx[i,:len(inverse[i])]
        #     #print('mapping:',mapping_label[0].item(),inverse[i][0].item())
        #     debug.append([len(lab[i]),len(torch.unique(mapping_label))])
        # print(debug)

    def merge_v3(self,indexes,pred,cur_label,lab,inverse,batch_neighbor,next_label=None,source_class=0,merge_idxs=None,merge_thre=0):
        max_idx=torch.argmax(pred,dim=2)

        merge_map={}
        import pdb;pdb.set_trace()
        for i in range(len(indexes)):
            if self.merge_wo_outlier:
                if indexes[i] not in merge_idxs:
                    continue
            merge_lab=list(set(lab[i][pred_i[0]>=0.7*pred_i[0,inverse[i][0]]].tolist()))
            if next_label is not None:
                filter_lab=inverse[next_label[batch_neighbor[i,:len(inverse[i])]]==next_label[indexes[i]]]
                merge_lab=list(set(merge_lab) & set(filter_lab.tolist()))
            if cur_label[indexes[i]].item() not in merge_lab:
                continue
            if len(merge_lab)>1:#merge
                merge_label_0=-1
                inter=set(merge_lab) & set(merge_map.keys())
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
                merge_label_0=merge_lab[0] if merge_label_0==-1 else merge_label_0
                for label in merge_lab:
                    merge_map[label]=merge_label_0

        for change_label,update_label in merge_map.items():
            #memory.labels[memory.labels==int(change_label)]=int(update_label)
            cur_label[cur_label==int(change_label)]=int(update_label)
        print('merge:',len(merge_map))
        ##for debug
        debug=[]
        for i in range(5):
            mapping_label=max_idx[i,:len(inverse[i])]
            #print('mapping:',mapping_label[0].item(),inverse[i][0].item())
            debug.append([len(lab[i]),len(torch.unique(mapping_label))])
        print(debug)

    def merge_v4(self,indexes,pred,cur_label,lab,inverse,batch_neighbor,next_label=None,source_class=0,merge_idxs=None,merge_thre=0,bias=0.0):
        merge_map={}
        change_num=[]#for debug
        debug=1
        for i in range(len(indexes)):
            if self.merge_wo_outlier:
                if indexes[i] not in merge_idxs:
                    continue
            merge_lab=set(lab[i][pred[i,0,:len(lab[i])]>bias].tolist())
            if cur_label[indexes[i]].item() not in merge_lab:
                continue

            if len(merge_lab)>1:#merge
                merge_lab=list(merge_lab)
                merge_label_0=-1
                inter=set(merge_lab) & set(merge_map.keys())
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
                merge_label_0=merge_lab[0] if merge_label_0==-1 else merge_label_0
                for label in merge_lab:
                    merge_map[label]=merge_label_0

        for change_label,update_label in merge_map.items():
            #memory.labels[memory.labels==int(change_label)]=int(update_label)
            cur_label[cur_label==int(change_label)]=int(update_label)
        if len(merge_map)>0:
            print('merge:',len(merge_map))
        print('change num:',change_num)

    def merge_v5(self,indexes,pred,cur_label,lab,inverse,batch_neighbor,next_label=None,source_class=0,merge_idxs=None,nums=None,sims=None,memory=None):
        max_idx = torch.argmax(pred, dim=2)

        merge_map={}
        change_num={}#for debug
        debug=1
        for i in range(len(indexes)):
            if self.merge_wo_outlier:
                if indexes[i] not in merge_idxs:
                    continue
            mapping_label=max_idx[i,:len(inverse[i])]
            #core=batch_neighbor[i,:len(inverse[i])][(mapping_label==inverse[i][0])]
            #import pdb;pdb.set_trace()
            #find merge lab
            keep=pred[i][:len(inverse[i]),:len(lab[i])]
            keep=keep[(mapping_label==inverse[i][0]) & (inverse[i]==inverse[i][0])]#core

            #for debug
            change_num[nums[cur_label[indexes[i]]].item()]=len(keep)
            ###
            #merge_lab=lab[i][torch.any(keep>(self.method5_ratio*keep[:,inverse[i][0]]).unsqueeze(1).expand_as(keep),dim=0)].tolist()
            if len(keep)==0:
                continue
            merge_lab = lab[i][
                torch.any(keep > (self.method5_ratio * (keep[:, inverse[i][0]]).max()),dim=0)].tolist()


            if next_label is not None:
                same_label_nei=batch_neighbor[i,:len(inverse[i])][next_label[batch_neighbor[i,:len(inverse[i])]]==next_label[indexes[i]]]
                same_label_cur_label=cur_label[same_label_nei]
                merge_lab=list(set(same_label_cur_label.tolist()) & set(merge_lab))

            if len(merge_lab)>1:#merge
                if debug:
                    print('num: {} thre:{}'.format(nums[cur_label[indexes[i]]].item(),self.method5_ratio*keep[:,inverse[i][0]]))
                    print('merge num:', (nums.view(-1))[merge_lab])
                    debug = 0
                merge_label_0=-1
                inter=set(merge_lab) & set(merge_map.keys())
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
                merge_label_0=merge_lab[0] if merge_label_0==-1 else merge_label_0
                for label in merge_lab:
                    merge_map[label]=merge_label_0

        for change_label,update_label in merge_map.items():
            #memory.labels[memory.labels==int(change_label)]=int(update_label)
            cur_label[cur_label==int(change_label)]=int(update_label)
        if len(merge_map)>0:
            print('merge:',len(merge_map))
        print('change num:',change_num)
        ##for debug
        #debug=[]
        # for i in range(5):
        #     mapping_label=max_idx[i,:len(inverse[i])]
        #     #print('mapping:',mapping_label[0].item(),inverse[i][0].item())
        #     debug.append([len(lab[i]),len(torch.unique(mapping_label))])
        # print(debug)

    def merge_v6(self,indexes,pred,cur_label,lab,inverse,batch_neighbor,next_label=None,source_class=0,merge_idxs=None,nums=None,sims=None,memory=None,bias=-1.0):
        max_idx = torch.argmax(pred, dim=2)
        #import pdb;pdb.set_trace()

        merge_map={}
        change_num={}#for debug
        debug=1
        for i in range(len(indexes)):
            if self.merge_wo_outlier:
                if indexes[i] not in merge_idxs:
                    continue
            mapping_label=max_idx[i,:len(inverse[i])]
            #core=batch_neighbor[i,:len(inverse[i])][(mapping_label==inverse[i][0])]
            #import pdb;pdb.set_trace()
            #find merge lab
            keep=pred[i][:len(inverse[i]),:len(lab[i])]
            keep=keep[(mapping_label==inverse[i][0]) & (inverse[i]==inverse[i][0])]#core

            #for debug
            change_num[nums[cur_label[indexes[i]]].item()]=len(keep)
            ###
            #merge_lab=lab[i][torch.any(keep>(self.method5_ratio*keep[:,inverse[i][0]]).unsqueeze(1).expand_as(keep),dim=0)].tolist()
            if len(keep)==0:
                continue
            merge_lab = lab[i][torch.any(keep > bias,dim=0)].tolist()


            if next_label is not None:
                same_label_nei=batch_neighbor[i,:len(inverse[i])][next_label[batch_neighbor[i,:len(inverse[i])]]==next_label[indexes[i]]]
                same_label_cur_label=cur_label[same_label_nei]
                merge_lab=list(set(same_label_cur_label.tolist()) & set(merge_lab))

            if len(merge_lab)>1:#merge
                if debug:
                    print('num: {} thre:{}'.format(nums[cur_label[indexes[i]]].item(),keep[:,inverse[i][0]]))
                    print('merge num:', (nums.view(-1))[merge_lab])
                    debug = 0
                merge_label_0=-1
                inter=set(merge_lab) & set(merge_map.keys())
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
                merge_label_0=merge_lab[0] if merge_label_0==-1 else merge_label_0
                for label in merge_lab:
                    merge_map[label]=merge_label_0

        for change_label,update_label in merge_map.items():
            #memory.labels[memory.labels==int(change_label)]=int(update_label)
            cur_label[cur_label==int(change_label)]=int(update_label)
        if len(merge_map)>0:
            print('merge:',len(merge_map))
        print('change num:',change_num)
        ##for debug
        #debug=[]
        # for i in range(5):
        #     mapping_label=max_idx[i,:len(inverse[i])]
        #     #print('mapping:',mapping_label[0].item(),inverse[i][0].item())
        #     debug.append([len(lab[i]),len(torch.unique(mapping_label))])
        # print(debug)


    def empty_label(self,label,return_num):
        empty_label=set(torch.arange(label.max()+1).tolist())-set(label.tolist())
        if len(empty_label)<return_num:
            outliers_label=torch.arange(label.max()+1,label.max()+1+return_num).cuda()
        else:
            empty_label=list(empty_label)[-return_num:]
            outliers_label=torch.tensor(empty_label).cuda()
        return outliers_label

    def postprocess(self,s_indexes,memory):
        self.utils.update_sub_cluster_label(s_indexes,memory)
        #step1 merge&split
        #step2 update sub cluster label-->src


#others
def p_lp():
    model = Point_Level()
    model.cuda()
    #model = nn.DataParallel(model)
    return model

def merge_lp(alpha, topk_num,temp,**kwargs):
    model = Merge_LP(alpha=alpha,topk_num=topk_num,temp=temp)
    model.cuda()
    #model = nn.DataParallel(model)
    return model

def split_lp(alpha,split_num,thre,**kwargs):
    model=Split_LP(alpha=alpha,split_num=split_num,confidence_thre=thre)
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
