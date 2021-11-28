#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from examples.utils import GraphConv, MeanAggregator
from spcl.utils.faiss_rerank import compute_jaccard_distance,compute_jaccard_distance_step1,compute_jaccard_distance_inital_rank,compute_knn

class Point_Level_GCN(nn.Module):
    def __init__(self, feature_dim, nhid, feature_size,source_classes,nclass, momentum=0.2,dropout=0,cal_num=30):
        super(Point_Level_GCN, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        self.conv2 = GraphConv(nhid, 512, MeanAggregator, dropout)
        self.conv3 = GraphConv(512, 256, MeanAggregator, dropout)
        self.conv4 = GraphConv(256, 256, MeanAggregator, dropout)

        self.nclass = 2
        self.classifier = nn.Sequential(nn.Linear(256, 256), nn.PReLU(256),
                                        nn.Linear(256, self.nclass))
        self.loss=torch.nn.CrossEntropyLoss(reduction='none').cuda()
        self.cal_num=cal_num
        self.top5_avg=0

    def forward(self, indexes, features,labels,domain,ori_0,ori_knn_neighbor,neighbor_num,gt_conf=None,gt_weights=None,output_feat=False, return_loss=False):
        if domain:
            with torch.no_grad(): #compute A & x
                all_x=[]
                all_adj=[]
                all_neighbors=[]
                topk=10
                all_index_dict={}

                for i in range(len(indexes)):
                    all_neighbor=ori_knn_neighbor[i].tolist()
                    # if self.top5_avg:
                    #     nei_top=torch.from_numpy(ori_0[all_neighbor,:5]).view(-1)
                    #     feature=features[nei_top].type(torch.float)
                    #     neigh=torch.from_numpy(ori_0[nei_top,:topk])
                    #     tmp=torch.zeros(len(nei_top),len(features)).float().cuda()
                    #     A=feature.mm(features.t())
                    #     tmp[torch.arange(len(nei_top)).view(-1,1).repeat(1,topk),neigh]=A[torch.arange(len(nei_top)).view(-1,1).repeat(1,topk),neigh]
                    #     adj=tmp.mm(torch.gt(tmp.t(),0).float())
                    #     adj=F.softmax(adj,dim=1)
                    #     adj=torch.mean(adj.view(5,len(nei_top)//5,-1),dim=0)
                    #     adj=adj[:,0::5]
                    #     feature=features[all_neighbor].type(torch.float)
                    # else:
                    feature=features[all_neighbor].type(torch.float)
                    all_neighbors.append(all_neighbor)
                    neigh=torch.from_numpy(ori_0[all_neighbor,:topk]).cuda()
                    tmp=torch.zeros(neighbor_num,len(features)).float().cuda()
                    A=feature.mm(features.t())
                    tmp[torch.arange(neighbor_num).view(-1,1).repeat(1,topk),neigh]=A[torch.arange(neighbor_num).view(-1,1).repeat(1,topk),neigh]
                    adj=tmp.mm(torch.gt(tmp.t(),0).float())
                    adj=F.softmax(adj,dim=1)

                    feature-=feature[0].clone()

                    all_x.append(feature.unsqueeze(0))
                    all_adj.append(adj.unsqueeze(0))
            #TODO:concat
            all_x=torch.cat(all_x,dim=0)
            all_adj=torch.cat(all_adj,dim=0)
            # x=self.conv1(all_x, all_adj)
            all_pred = []
            x=self.conv1(all_x,all_adj)
            x= self.conv2(x,all_adj)
            x= self.conv3(x,all_adj)
            x= self.conv4(x,all_adj)
            dout=x.size(-1)
            x_0=x.view(-1,dout)
            all_pred=self.classifier(x_0) #B*topk*2
            #all_pred.append(pred.unsqueeze(0))
            #all_pred=torch.cat(all_pred,dim=0)

            if output_feat:
                all_pred=all_pred.view(len(indexes),-1,self.nclass)
                all_pred=F.softmax(all_pred,dim=2)
                return all_pred

            if return_loss:
                # for i in range(len(indexes)):
                #     loss += torch.mean((self.loss(all_pred[i,:self.cal_num], gt_conf[i,:self.cal_num])*gt_weights[i,:self.cal_num]))
                #all_pred0=all_pred.permute(1,2,0)
                with torch.no_grad():
                    gt_conf=gt_conf.view(-1)
                    gt_weights[:,self.cal_num:]=0
                    gt_weights=gt_weights.view(-1)
                    gt_weights_new=gt_weights[gt_weights>0]
                    cnt=torch.sum(gt_weights>0)
                    view_pred=all_pred.detach()
                loss=torch.sum(self.loss(all_pred[gt_weights>0,:],gt_conf[gt_weights>0])*gt_weights_new)/cnt
                return loss,view_pred.view(len(indexes),-1,self.nclass)
        else:
            return torch.tensor(0).cuda(),0

class Sub_Cluster_Level_GCN(nn.Module):
    def __init__(self, feature_dim, nhid, feature_size,source_classes,nclass, momentum=0.2,dropout=0,cal_num=30,A_thre=0.3):
        super(Sub_Cluster_Level_GCN, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)

        self.nclass = 2
        self.classifier = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(nhid),
                                        nn.Linear(nhid, self.nclass))
        self.loss=torch.nn.CrossEntropyLoss(reduction='none').cuda()

        self.source_classes=source_classes
        self.cal_num=cal_num
        self.eps = np.finfo(float).eps
        self.A_thre=A_thre

    def forward(self,indexes,features,labels,sub_label,domain,ori_0,ori_knn_neighbor,all_pred,gt_conf=None,gt_weights=None,output_feat=False, return_loss=False,min_point=3,num_penalty=0,num_penalty_ratio=1,pred_ratio=1):
        if domain:
            neighbor_num=len(ori_knn_neighbor[0])
            #sub feature-X
            all_x=[]
            all_adj=[]
            with torch.no_grad():
                #sub cluster sum & num
                sub_sum = torch.zeros(sub_label.max()+1, 2048).float().cuda()
                sub_sum.index_add_(0, sub_label, features)
                nums = torch.zeros(sub_label.max()+1, 1).float().cuda()
                nums.index_add_(0, sub_label, torch.ones(len(sub_label),1).float().cuda())
                mask = (nums>0).float()
                sub_sum /= (mask*nums+(1-mask)).clone().expand_as(sub_sum)
                if output_feat:
                    print('sub max:',nums[self.source_classes:].max())

                for i in range(len(indexes)):
                    #sub_lab=sub_label[torch.from_numpy(ori_knn_neighbor[i]).cuda()]
                    #sub_lab[0]=indexes[i]
                    #sub_feat=features[sub_lab].type(torch.float)
                    sub_lab=sub_label[ori_knn_neighbor[i].tolist()]
                    sub_feat=sub_sum[sub_lab].type(torch.float)
                    sub_feat[0]=features[indexes[i]] #index itself

                    #A
                    A=sub_feat.mm(sub_feat.t())
                    #A*=(torch.exp(pred_ratio*all_pred[i,:,1]).unsqueeze(0).expand_as(A))#addweights
                    # if num_penalty:
                    #     A_nums=nums[sub_lab].view(-1)
                    #     A_nums[0]=1
                    #     A_nums/=50#A_nums.max() 104-->use 50 to normalize
                    #     A-=(1-torch.exp(-num_penalty_ratio*A_nums).unsqueeze(0).expand_as(A))
                    # A=F.softmax(A,dim=1)
                    #A[A<self.A_thre]=0
                    #A=A/torch.sum(A+self.eps,dim=1,keepdim=True)
                    A/=5

                    topk, indices = torch.topk(A, 5)
                    mask = torch.zeros_like(A)
                    mask = mask.scatter(1, indices, 1)
                    mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32) #intersection
                    A=A*mask

                    sub_feat/=torch.norm(sub_feat,dim=1,keepdim=True)
                    sub_feat-=sub_feat[0].clone() #X
                    all_x.append(sub_feat.unsqueeze(0))
                    all_adj.append(A.unsqueeze(0))
                del A

            all_x=torch.cat(all_x,dim=0)
            all_adj=torch.cat(all_adj,dim=0)

            x=self.conv1(all_x, all_adj)
            dout=x.size(-1)
            x_0=x.view(-1,dout)
            all_pred=self.classifier(x_0) #B*topk*2

            if output_feat:
                #cal sub criterion
                #nums[:self.source_classes]=0
                #keep_sub_sum=sub_sum[nums.view(-1)>=min_point]#wo outliers
                #simm=torch.sum(keep_sub_sum*keep_sub_sum,dim=1)
                #sub_sim_mean=torch.mean(simm)
                #sub_sim_var=torch.var(simm)
                simm=torch.sum(sub_sum*sub_sum,dim=1)

                all_pred=all_pred.view(len(indexes),-1,self.nclass)
                all_pred=F.softmax(all_pred,dim=2)
                return all_pred,simm,sub_sum,nums#sub_sim_mean,sub_sim_var

            if return_loss: #TODO:add weights
                #all_pred0=all_pred.permute(1,2,0)
                #loss=torch.mean(self.loss(all_pred0[:,:,:self.cal_num],gt_conf[:,:self.cal_num])*gt_weights[:,:self.cal_num])
                with torch.no_grad():
                    gt_conf=gt_conf.view(-1)
                    gt_weights[:,self.cal_num:]=0
                    #104-->rm self
                    gt_weights[:,0]=0
                    gt_weights=gt_weights.view(-1)
                    gt_weights_new=gt_weights[gt_weights>0]
                    cnt=torch.sum(gt_weights>0)
                    view_pred=all_pred.detach()
                loss=torch.sum(self.loss(all_pred[gt_weights>0,:],gt_conf[gt_weights>0])*gt_weights_new)/cnt
                return loss,view_pred.view(len(indexes),-1,self.nclass)

        else:
            return torch.tensor(0).cuda(),0

class Cluster_Level_GCN(nn.Module):
    def __init__(self, feature_dim, nhid, feature_size,source_classes,nclass, momentum=0.2,dropout=0,cal_num=30,A_thre=0.3):
        super(Cluster_Level_GCN, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)

        self.nclass = 2
        self.classifier = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(nhid),
                                        nn.Linear(nhid, self.nclass))
        self.loss=torch.nn.CrossEntropyLoss(reduction='none').cuda()
        self.cal_num=cal_num
        self.source_classes=source_classes
        self.eps = np.finfo(float).eps
        self.A_thre=A_thre

    def forward(self, indexes, features,labels,domain,ori_0,ori_knn_neighbor,all_pred,gt_conf=None,gt_weights=None,output_feat=False, return_loss=False,min_point=3,num_penalty=0,num_penalty_ratio=1,pred_ratio=1):
        if domain:
            neighbor_num=len(ori_knn_neighbor[0])
            all_x=[]
            all_adj=[]
            with torch.no_grad():
                clu_sum = torch.zeros(labels.max()+1, 2048).float().cuda()
                clu_sum.index_add_(0, labels, features)
                nums = torch.zeros(labels.max()+1, 1).float().cuda()
                nums.index_add_(0, labels, torch.ones(len(labels),1).float().cuda())
                mask = (nums>0).float()
                clu_sum /= (mask*nums+(1-mask)).clone().expand_as(clu_sum)
                if output_feat:
                    print('clu num max:',nums.max())

                for i in range(len(indexes)):
                    clu_lab=labels[ori_knn_neighbor[i].tolist()]
                    clu_feat=clu_sum[clu_lab].type(torch.float)
                    clu_feat[0]=features[indexes[i]]

                    A=clu_feat.mm(clu_feat.t())
                    #A*=(torch.exp(pred_ratio*all_pred[i,:,1]).unsqueeze(0).expand_as(A))#addweights
                    # if num_penalty:
                    #     A_nums[0]=1
                    #     #104-->update use 50 to normalize
                    #     # if A_nums.max()>1000:
                    #     #     print('large number')
                    #     #     import pdb;pdb.set_trace()
                    #     A_nums/=50#A_nums.max()
                    #     #A*=torch.exp(-num_penalty_ratio*A_nums).unsqueeze(0).expand_as(A)
                    #     A-=(1-torch.exp(-num_penalty_ratio*A_nums).unsqueeze(0).expand_as(A))
                    #A=torch.exp(A)
                    #A[A<self.A_thre]=0
                    #A=A/torch.sum(A+self.eps,dim=1,keepdim=True)
                    A/=5 #wo normalization

                    filter_lab,inde=np.unique(clu_lab.cpu().numpy(),return_index=True)
                    del_list=list(set(np.arange(len(A)).tolist())-set(inde))
                    #k_nums=max(len(inde)//10,2)

                    topk, indices = torch.topk(A, 5)#k_nums)
                    mask = torch.zeros_like(A)
                    mask = mask.scatter(1, indices, 1)
                    mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32) #intersection
                    #keep only one for each cluster center
                    mask[del_list]=0
                    mask[:,del_list]=0

                    A=A*mask

                    clu_feat/=torch.norm(clu_feat,dim=1,keepdim=True) #cluster feat
                    clu_feat-=clu_feat[0].clone()
                    all_x.append(clu_feat.unsqueeze(0))
                    all_adj.append(A.unsqueeze(0))
                del A

            all_x=torch.cat(all_x,dim=0)
            all_adj=torch.cat(all_adj,dim=0)
            x=self.conv1(all_x, all_adj)
            dout=x.size(-1)
            x_0=x.view(-1,dout)
            all_pred=self.classifier(x_0)

            if output_feat:
                #cal cluster mean
                #nums[:self.source_classes]=0
                #keep_clu_sum=clu_sum[nums.view(-1)>=min_point]#wo outliers
                #simm=torch.sum(keep_clu_sum*keep_clu_sum,dim=1)
                #clu_sim_mean=torch.mean(simm)
                #clu_sim_var=torch.var(simm)
                simm=torch.sum(clu_sum*clu_sum,dim=1)

                all_pred=all_pred.view(len(indexes),-1,self.nclass)
                all_pred=F.softmax(all_pred,dim=2)
                return all_pred,simm,clu_sum,nums

            if return_loss: #TODO:add weights
                with torch.no_grad():
                    gt_conf=gt_conf.view(-1)
                    gt_weights[:,self.cal_num:]=0
                    #104-->rm self
                    gt_weights[:,0]=0
                    ###############
                    gt_weights=gt_weights.view(-1)
                    gt_weights_new=gt_weights[gt_weights>0]
                    cnt=torch.sum(gt_weights>0)
                    view_pred=all_pred.detach()
                loss=torch.sum(self.loss(all_pred[gt_weights>0,:],gt_conf[gt_weights>0])*gt_weights_new)/cnt
                return loss,view_pred.view(len(indexes),-1,self.nclass)
        else:
            return torch.tensor(0).cuda(),0

class Split_LP(nn.Module):#fps
    def __init__(self,alpha):
        super(Split_LP, self).__init__()
        self.alpha=alpha
        self.eps = np.finfo(float).eps
        self.method=1 #0-->2 cluster 1-->3 cluster
        self.split_num=8
    def forward(self,indexes,features,labels,sub_level=0,sub_labels=None,outliers_label=None):
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
                    if print_cnts==0:
                        print(pred)
                        print_cnts=1
            else:
                print_cnts=0
                for n,idx in enumerate(indexes):
                    batch_idx=all_idxs[labels==labels[idx]]
                    batch_sub_label=sub_labels[batch_idx]
                    split_idxs=list(set((batch_sub_label).tolist())) #sub label
                    if len(split_idxs)<=self.split_num:
                        continue
                    anchor_idxs=[]
                    anchor_indices=[]

                    split_feat=features[split_idxs]
                    split_sim=features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
                    anchor_idx=split_idxs[torch.argmin(split_sim)]
                    anchor_idxs.append(anchor_idx)
                    anchor_indices.append(torch.argmin(split_sim).item())

                    for sp in range(1,self.split_num):
                        split_sim_2=features[sub_labels[anchor_idx]].unsqueeze(0).mm(split_feat.t())
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
                    ori_label=labels[idx].item()
                    for sub,pre in zip(split_idxs,pred):
                        labels[batch_idx[batch_sub_label==sub]]=labs[pre]
                    labels[batch_idx[batch_sub_label==sub_labels[idx]]]=ori_label#outliers_label[(self.split_num-1)*len(indexes)+n]
                    if print_cnts==0:
                        print(pred)
                        print_cnts=1
    # def forward(self,indexes,features,labels,sub_level=0,sub_labels=None,outliers_label=None):
    #     all_idxs=torch.arange(len(labels)).cuda()
    #     split_nums=[]
    #     sub_nums=[]
    #     if self.method==0:
    #         if sub_level:
    #             for n,idx in enumerate(indexes):
    #                 split_idxs=all_idxs[labels==labels[idx]]
    #                 if len(split_idxs)<=2:
    #                     continue
    #                 split_feat=features[labels==labels[idx]]
    #                 split_sim=features[idx].unsqueeze(0).mm(split_feat.t())
    #                 anchor_idx=split_idxs[torch.argmin(split_sim)]
    #                 split_sim_2=features[anchor_idx].unsqueeze(0).mm(split_feat.t())
    #                 anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
    #
    #                 Y=torch.zeros((len(split_idxs),2)).cuda()
    #                 i_0,i_1=torch.argmax(split_sim_2),torch.argmin(split_sim_2)
    #                 Y[i_0,0]=1
    #                 Y[i_1,1]=1
    #
    #                 W=torch.exp(split_feat.mm(split_feat.t()))
    #                 # mask=torch.ones_like(W)
    #                 # mask[i_0,i_0]=0
    #                 # mask[i_1,i_1]=0
    #                 # W*=mask
    #                 D       = W.sum(0)
    #                 D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
    #                 D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,len(split_idxs))
    #                 D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(len(split_idxs),1)
    #                 S       = D1*W*D2
    #
    #                 pred  = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda()-self.alpha*S+self.eps), Y)
    #                 pred=torch.argmax(pred,dim=1)
    #                 lab=torch.tensor([anchor_idx.item(),anchor_idx_2.item()]).cuda()
    #                 labels[split_idxs]=lab[pred]
    #                 #for debug
    #                 split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
    #         else:
    #             for n,idx in enumerate(indexes):
    #                 batch_idx=all_idxs[labels==labels[idx]]
    #                 batch_sub_label=sub_labels[batch_idx]
    #                 split_idxs=list(set((batch_sub_label).tolist())) #sub label
    #                 if len(split_idxs)==1:
    #                     continue
    #                 split_feat=features[split_idxs]
    #                 split_sim=features[sub_labels[idx]].unsqueeze(0).mm(split_feat.t())
    #                 anchor_idx=split_idxs[torch.argmin(split_sim)]
    #                 split_sim_2=features[sub_labels[anchor_idx]].unsqueeze(0).mm(split_feat.t())
    #                 anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
    #
    #                 Y=torch.zeros((len(split_idxs),2)).cuda()
    #                 i_0,i_1=torch.argmax(split_sim_2),torch.argmin(split_sim_2)
    #                 Y[i_0,0]=1
    #                 Y[i_1,1]=1
    #
    #                 W=torch.exp(split_feat.mm(split_feat.t()))
    #                 #mask=torch.ones_like(W)
    #                 #mask[i_0,i_0]=0
    #                 #mask[i_1,i_1]=0
    #                 #W*=mask
    #                 D       = W.sum(0)
    #                 D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
    #                 D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,len(split_idxs))
    #                 D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(len(split_idxs),1)
    #                 S       = D1*W*D2
    #
    #                 pred  = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda()-self.alpha*S+self.eps), Y)
    #                 pred=torch.argmax(pred,dim=1)
    #
    #                 labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
    #                 for sub,pre in zip(split_idxs,pred):
    #                     labels[batch_idx[batch_sub_label==sub]]=labs[pre]
    #                 split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
    #     elif self.method==1:
    #         if sub_level:
    #             for n,idx in enumerate(indexes):
    #                 split_idxs=all_idxs[labels==labels[idx]]
    #                 if len(split_idxs)<=2:
    #                     continue
    #                 split_feat=features[split_idxs]
    #                 split_sim=features[idx].unsqueeze(0).mm(split_feat.t())
    #                 anchor_idx=split_idxs[torch.argmin(split_sim)]
    #                 split_sim_2=features[anchor_idx].unsqueeze(0).mm(split_feat.t())
    #                 #anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
    #                 # if anchor_idx_2==idx:
    #                 #     split_sim_2[0,torch.argmin(split_sim_2)]=1
    #                 #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
    #                 #fix bug 104
    #                 split_sim_2[split_sim_2<split_sim]=split_sim[split_sim_2<split_sim]
    #                 anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
    #                 assert (anchor_idx != idx ) and (anchor_idx_2 != idx)
    #
    #                 Y=torch.zeros((len(split_idxs),2)).cuda()
    #                 i_0,i_1=torch.argmin(split_sim),torch.argmin(split_sim_2)
    #                 Y[i_0,0]=1
    #                 Y[i_1,1]=1
    #
    #                 #104-->fix bug
    #                 W=torch.exp(split_feat.mm(split_feat.t()))
    #                 mask=(1-torch.eye(len(split_feat))).cuda()
    #                 W*=mask
    #
    #                 D       = W.sum(0)
    #                 D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
    #                 D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,len(split_idxs))
    #                 D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(len(split_idxs),1)
    #                 S       = D1*W*D2
    #
    #                 pred  = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda()-self.alpha*S+self.eps), Y)
    #                 pred=torch.argmax(pred,dim=1)
    #
    #                 lab=torch.tensor([anchor_idx.item(),anchor_idx_2.item()]).cuda()
    #                 labels[split_idxs]=lab[pred]
    #                 labels[idx]=idx
    #                 #for debug
    #                 # if n==0:
    #                 #     print('pred:',pred)
    #                 #     print('label:',labels[split_idxs])
    #                 split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
    #         else:
    #             for n,idx in enumerate(indexes):
    #                 batch_idx=all_idxs[labels==labels[idx]]
    #                 batch_sub_label=sub_labels[batch_idx]
    #                 split_idxs=list(set((batch_sub_label).tolist())) #sub label
    #                 if len(split_idxs)<=2:
    #                     continue
    #                 split_feat=features[split_idxs]
    #                 split_feat/=torch.norm(split_feat,dim=1,keepdim=True)
    #                 split_sim=(features[sub_labels[idx]]/features[sub_labels[idx]].norm()).unsqueeze(0).mm(split_feat.t())
    #                 anchor_idx=split_idxs[torch.argmin(split_sim)]
    #                 split_sim_2=(features[sub_labels[anchor_idx]]/features[sub_labels[anchor_idx]].norm()).unsqueeze(0).mm(split_feat.t())
    #                 # anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
    #                 # if anchor_idx_2==sub_labels[idx]:
    #                 #     split_sim_2[0,torch.argmin(split_sim_2)]=1
    #                 #     anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
    #                 split_sim_2[split_sim_2<split_sim]=split_sim[split_sim_2<split_sim]
    #                 anchor_idx_2=split_idxs[torch.argmin(split_sim_2)]
    #                 assert (anchor_idx!=sub_labels[idx]) and (anchor_idx_2 != sub_labels[idx])
    #
    #                 Y=torch.zeros((len(split_idxs),2)).cuda()
    #                 i_0,i_1=torch.argmin(split_sim),torch.argmin(split_sim_2)
    #                 Y[i_0,0]=1
    #                 Y[i_1,1]=1
    #
    #                 #104-->fix bug
    #                 W=torch.exp(split_feat.mm(split_feat.t()))
    #                 mask=(1-torch.eye(len(split_feat))).cuda()
    #                 W*=mask
    #
    #                 D       = W.sum(0)
    #                 D_sqrt_inv = torch.sqrt(1.0/(D+self.eps))
    #                 D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,len(split_idxs))
    #                 D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(len(split_idxs),1)
    #                 S       = D1*W*D2
    #
    #                 pred  = torch.matmul(torch.inverse(torch.eye(len(split_idxs)).cuda()-self.alpha*S+self.eps), Y)
    #                 pred=torch.argmax(pred,dim=1)
    #
    #                 labs=torch.tensor([labels[idx].item(),outliers_label[n].item()]).cuda()
    #                 for sub,pre in zip(split_idxs,pred):
    #                     labels[batch_idx[batch_sub_label==sub]]=labs[pre]
    #                 labels[batch_idx[batch_sub_label==sub_labels[idx]]]=outliers_label[len(indexes)+n]
    #                 split_nums.append([len(split_idxs)-torch.sum(pred).item(),torch.sum(pred).item()])
    #                 sub_nums.append(len(batch_idx[batch_sub_label==sub_labels[idx]]))
    #             print('sub nums:',sub_nums)
    #     print('split:',split_nums)

class Hierarchy_GCN(object):
    def __init__(self,point_level_gcn,sub_cluster_level_gcn,cluster_level_gcn,split_lp,utils,r=1,merge_label_func=0,point_add_dyn_thre=0,
                var_weights=2,mean_min_points=2,merge_dyn_ratio=1.1,cal_num=30,cluster_level_thre=0.5,tgt_only=0,dyn_thre_type=1,
                consider_cur_batch=0,num_penalty=1,num_penalty_ratio=1,fix_dyn_thre=0,pred_ratio=1,
                pred_dyn_type=0,neighbor_num=64,split_type=0,thre=0.8):
        self.point_level_gcn=point_level_gcn
        self.sub_cluster_level_gcn=sub_cluster_level_gcn
        self.cluster_level_gcn=cluster_level_gcn
        self.split_lp=split_lp
        self.utils=utils #utils


        self.thre=thre
        self.step=(1-self.thre)*0.5

        self.neighbor_num=neighbor_num #knn
        self.cal_num=cal_num
        self.r=r
        self.var_weights=var_weights
        self.mean_min_points=mean_min_points
        self.merge_label_func=merge_label_func
        self.point_add_dyn_thre=point_add_dyn_thre
        #######
        self.dyn_thre_type=dyn_thre_type
        self.merge_dyn_ratio=merge_dyn_ratio
        self.split_type=split_type

        self.train_cluster_outliers_label=None
        self.train_sub_outliers_label=None

        self.tgt_only=tgt_only
        self.cluster_level_thre=cluster_level_thre

        self.consider_cur_batch=consider_cur_batch

        self.num_penalty=num_penalty
        self.num_penalty_ratio=num_penalty_ratio
        self.pred_ratio=pred_ratio
        self.fix_dyn_thre=fix_dyn_thre

        self.pred_dyn_type=pred_dyn_type #for gcn
        self.sub_pred_dyn_thre=-1
        self.clu_pred_dyn_thre=-1
        if self.num_penalty:
            print('use num penalty...')

    def train(self,s_indexes,memory,train):
        if train:
            #knn neighbor
            ori_0=compute_knn(memory.s_features.clone(),k1=self.neighbor_num)
            ori_knn_neighbor=ori_0[s_indexes.cpu().numpy(),:]

            #cal gt
            all_gt=[]
            all_sub_gt=[]
            for i in range(len(s_indexes)):
                all_gt.append(memory.s_label[ori_knn_neighbor[i].tolist()].unsqueeze(0))
                all_sub_gt.append(memory.s_sub_label[ori_knn_neighbor[i].tolist()].unsqueeze(0))
            all_gt=torch.cat(all_gt,0)
            all_sub_gt=torch.cat(all_sub_gt,0)
            gt_template=all_gt[:,0].unsqueeze(1).expand_as(all_gt)
            gt_sub_template=all_sub_gt[:,0].unsqueeze(1).expand_as(all_sub_gt)
            gt_conf=(all_gt==gt_template).long()
            #gt weights-->sub 2 gt 1
            gt_weights=torch.ones_like(gt_conf).float()
            gt_weights+=self.r*(all_sub_gt==gt_sub_template).float()

            #add change label
            if self.train_cluster_outliers_label is None:
                self.train_cluster_outliers_label=torch.arange(len(memory.s_label)+1,len(memory.s_label)+1+len(s_indexes)).cuda()
            if self.train_sub_outliers_label is None:
                self.train_sub_outliers_label=torch.arange(len(memory.s_sub_label)+1,len(memory.s_sub_label)+1+len(s_indexes)).cuda()

            ori_cluster_label=memory.s_label[s_indexes]
            ori_sub_label=memory.s_sub_label[s_indexes]
            memory.s_label[s_indexes]=self.train_cluster_outliers_label
            memory.s_sub_label[s_indexes]=self.train_sub_outliers_label # for simplitfy
        else:
            gt_conf=None
            gt_weights=None
            ori_0=-1
            ori_knn_neighbor=-1

        loss_point_level,all_pred=self.point_level_gcn(s_indexes,memory.s_features,memory.s_label,train,ori_0,ori_knn_neighbor,self.neighbor_num,gt_conf=gt_conf,gt_weights=gt_weights,return_loss=True)
        loss_sub_level,all_pred_sub=self.sub_cluster_level_gcn(s_indexes,memory.s_features,memory.s_label,memory.s_sub_label,train,ori_0,ori_knn_neighbor,all_pred,gt_conf=gt_conf,gt_weights=gt_weights,return_loss=True,num_penalty=self.num_penalty,num_penalty_ratio=self.num_penalty_ratio,pred_ratio=self.pred_ratio)
        loss_cluster_level,all_pred_clu=self.cluster_level_gcn(s_indexes,memory.s_features,memory.s_label,train,ori_0,ori_knn_neighbor,all_pred_sub,gt_conf=gt_conf,gt_weights=gt_weights,return_loss=True,num_penalty=self.num_penalty,num_penalty_ratio=self.num_penalty_ratio,pred_ratio=self.pred_ratio)

        if train:
            memory.s_label[s_indexes]=ori_cluster_label
            memory.s_sub_label[s_indexes]=ori_sub_label

        return [loss_point_level,loss_sub_level,loss_cluster_level]

    def inference(self,t_indexes,memory,infer):
        if infer:
            if self.tgt_only: #memory.source_classes
                ori_0=compute_knn(memory.features[memory.source_classes:].clone(),k1=self.neighbor_num)
                ori_0+=memory.source_classes
                ori_0=np.concatenate((np.zeros((memory.source_classes,self.neighbor_num)),ori_0)).astype('int')
                ori_knn_neighbor=ori_0[t_indexes.cpu().numpy(),:]
            else:
                ori_0=compute_knn(memory.features.clone(),k1=self.neighbor_num)
                ori_knn_neighbor=ori_0[t_indexes.cpu().numpy(),:]

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

            #point level
            all_pred=self.point_level_gcn(t_indexes,memory.features,memory.labels,infer,ori_0,ori_knn_neighbor,self.neighbor_num,output_feat=True)
            neighbor_index,nearest_neighbor=self.point_level_merge_split(t_indexes,all_pred.detach(),torch.from_numpy(ori_knn_neighbor),memory)

            #sub level split
            self.split_lp(t_indexes,memory.features,memory.t_sub_label,sub_level=1)
            all_pred_sub,sub_inter_sim,self.sub_ori_sim,self.sub_nums=self.sub_cluster_level_gcn(t_indexes,memory.features,memory.labels,memory.t_sub_label,infer,ori_0,ori_knn_neighbor,all_pred,output_feat=True,min_point=self.mean_min_points,num_penalty=self.num_penalty,num_penalty_ratio=self.num_penalty_ratio,pred_ratio=self.pred_ratio)
            self.sub_cluster_level_merge_split(t_indexes,memory,neighbor_index,all_pred_sub.detach(),nearest_neighbor,torch.from_numpy(ori_knn_neighbor))

            #cluster level
            #outliers
            empty_label=set(torch.arange(memory.labels.max()+1).tolist())-set(memory.labels.tolist())
            if len(empty_label)<self.split_lp.split_num*len(t_indexes)+1:
                outliers_label=torch.arange(memory.labels.max()+1,memory.labels.max()+2+self.split_lp.split_num*len(t_indexes)).cuda()
            else:
                empty_label=list(empty_label)[-(self.split_lp.split_num*len(t_indexes)+1):]
                outliers_label=torch.tensor(empty_label).cuda()

            self.split_lp(t_indexes,self.sub_ori_sim,memory.labels,sub_level=0,sub_labels=memory.t_sub_label,outliers_label=outliers_label)
            all_pred_clu,clu_inter_sim,self.clu_sim,self.clu_nums=self.cluster_level_gcn(t_indexes,memory.features,memory.labels,infer,ori_0,ori_knn_neighbor,all_pred_sub,output_feat=True,min_point=self.mean_min_points,num_penalty=self.num_penalty,num_penalty_ratio=self.num_penalty_ratio,pred_ratio=self.pred_ratio)
            self.cluster_level_merge_split(t_indexes,memory,neighbor_index,self.clu_sim,all_pred_clu.detach(),nearest_neighbor,torch.from_numpy(ori_knn_neighbor),outliers_label,clu_inter_sim)

    def point_level_merge_split(self,indexes,all_pred,ori_knn_neighbor,memory):
        #print('point_level_merge_split')
        neighbor_thre=0.45

        max_conf,max_label=torch.max(all_pred[:,1:5,1],1) #only consider top5 neighbor as candidate
        nearest_neighbor=ori_knn_neighbor[torch.arange(len(indexes)),max_label+1].view(-1)

        #sim
        new_label=memory.t_sub_label[ori_knn_neighbor[torch.arange(len(indexes)),max_label+1].view(-1).cuda()]
        new_cluster_label=memory.labels[ori_knn_neighbor[torch.arange(len(indexes)),max_label+1].view(-1).cuda()]

        feature_sim=(memory.features[indexes]).mm(memory.features.t())
        neigh_sim=feature_sim[torch.arange(len(indexes)),nearest_neighbor]

        new_label[new_label<memory.source_classes]=0
        new_label[neigh_sim<neighbor_thre]=0
        print('outliers num:',torch.sum(new_label==0).item())

        #change sub cluster label & cluster label
        new_cluster_label[new_label==0]=memory.labels[indexes][new_label==0]
        memory.labels[indexes]=new_cluster_label
        new_label[new_label==0]=indexes[new_label==0] #outliers
        memory.t_sub_label[indexes]=new_label#new_label

        return max_label+1,nearest_neighbor

    def sub_cluster_level_merge_split(self,indexes,memory,neighbor_index,all_pred_sub,nearest_neighbor,ori_knn_neighbor):
        #print('sub mean:',torch.mean(all_pred_sub[:,:self.cal_num,1]))
        #print('sub mean+var:',torch.mean(all_pred_sub[:,:self.cal_num,1])+torch.var(all_pred_sub[:,:self.cal_num,1]))
        print('sub pred:',all_pred_sub[0,:5,1])
        merge_labels=(all_pred_sub[:,:self.cal_num,1]>self.thre).cpu()#+self.step).cpu()
        merge_label_sum=torch.sum(merge_labels,dim=1)

        merge_map={}
        for idx,lab in enumerate(merge_labels):
            if merge_label_sum[idx]<=1:
                continue
            merge_neighbors=ori_knn_neighbor[idx,:self.cal_num][(lab>0)]
            merge_idx=memory.t_sub_label[merge_neighbors]
            merge_idx_label=memory.labels[merge_neighbors]
            merge_idx=list(set(merge_idx[(merge_idx_label==merge_idx_label[0])].tolist()))

            if lab[0]==0 or lab[neighbor_index[idx]]==0: #not include itself/reliable neighbor
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

        ###for debug######
        merge_nums=[]
        for change_label,update_label in merge_map.items():
            if change_label != update_label:
                merge_nums.append([self.sub_nums[change_label].item(),self.sub_nums[update_label].item()])
        print('sub cluster merge nums:',merge_nums)
        ###################
        for change_label,update_label in merge_map.items():
            memory.t_sub_label[memory.t_sub_label==int(change_label)]=int(update_label)

    def cluster_level_merge_split(self,indexes,memory,neighbor_index,sim,all_pred_clu,nearest_neighbor,ori_knn_neighbor,outliers_label,clu_inter_sim):
        #print('clu mean:',torch.mean(all_pred_clu[:,:self.cal_num,1]))
        #print('clu mean+var:',torch.mean(all_pred_clu[:,:self.cal_num,1])+torch.var(all_pred_clu[:,:self.cal_num,1]))
        print('clu pred:',all_pred_clu[0,:5,1])
        merge_labels=(all_pred_clu[:,:self.cal_num,1]>self.thre).cpu()
        merge_label_sum=torch.sum(merge_labels,dim=1)

        merge_map={}
        for idx,lab in enumerate(merge_labels):
            if merge_label_sum[idx]<=1:
                continue
            merge_idx=list(set((memory.labels[ori_knn_neighbor[idx,:self.cal_num][lab>0]]).tolist()))
            if lab[0]==0 or lab[neighbor_index[idx]]==0: #not include itself/reliable neighbor
                continue
            #if len(merge_idx)>1 and self.clu_nums[merge_idx].max()>2000:
            #   import pdb;pdb.set_trace()
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

        ####for debug########
        merge_nums=[]
        for change_label,update_label in merge_map.items():
            if change_label!=update_label:
                merge_nums.append([self.clu_nums[change_label].item(),self.clu_nums[update_label].item()])
        print('cluster merge nums:',merge_nums)
        ####################
        for change_label,update_label in merge_map.items():
            memory.labels[memory.labels==int(change_label)]=int(update_label)
        #self.split_lp(split_batch,self.sub_ori_sim,memory.labels,sub_level=0,sub_labels=memory.t_sub_label,outliers_label=outliers_label)


    def postprocess(self,s_indexes,memory):
        self.utils.update_sub_cluster_label(s_indexes,memory)
        #step1 merge&split
        #step2 update sub cluster label-->src


#others
def p_gcn(feature_dim, nhid,feature_size, source_classes,nclass=1, dropout=0.,cal_num=30, **kwargs):
    model = Point_Level_GCN(feature_dim=feature_dim,
                  nhid=nhid,
                  feature_size=feature_size,
                  source_classes=source_classes,
                  nclass=nclass,
                  dropout=dropout,
                  cal_num=cal_num)
    model.cuda()
    #model = nn.DataParallel(model)
    return model

def s_gcn(feature_dim, nhid,feature_size, source_classes,nclass=1, dropout=0.,cal_num=30 ,**kwargs):
    model = Sub_Cluster_Level_GCN(feature_dim=feature_dim,
                  nhid=nhid,
                  feature_size=feature_size,
                  source_classes=source_classes,
                  nclass=nclass,
                  dropout=dropout,
                  cal_num=cal_num)
    model.cuda()
    #model = nn.DataParallel(model)
    return model

def c_gcn(feature_dim, nhid,feature_size, source_classes,nclass=1, dropout=0.,cal_num=30, **kwargs):
    model = Cluster_Level_GCN(feature_dim=feature_dim,
                  nhid=nhid,
                  feature_size=feature_size,
                  source_classes=source_classes,
                  nclass=nclass,
                  dropout=dropout,
                  cal_num=cal_num)
    model.cuda()
    #model = nn.DataParallel(model)
    return model

def split_lp(alpha=0.99, **kwargs):
    model=Split_LP(
        alpha=alpha
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
