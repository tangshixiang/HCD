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
    def __init__(self, feature_dim, nhid, feature_size,source_classes,nclass, momentum=0.2,dropout=0,cal_num=30):
        super(Sub_Cluster_Level_GCN, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        self.conv2 = GraphConv(nhid, 256, MeanAggregator, dropout)

        self.nclass = 2
        self.classifier = nn.Sequential(nn.Linear(256, 256), nn.PReLU(256),
                                        nn.Linear(256, self.nclass))
        self.loss=torch.nn.CrossEntropyLoss(reduction='none').cuda()

        self.source_classes=source_classes
        self.cal_num=cal_num

    def forward(self,indexes,features,labels,sub_label,domain,ori_0,ori_knn_neighbor,all_pred,gt_conf=None,gt_weights=None,output_feat=False, return_loss=False,min_point=3):
        if domain:
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

                for i in range(len(indexes)):
                    #sub_lab=sub_label[torch.from_numpy(ori_knn_neighbor[i]).cuda()]
                    #sub_lab[0]=indexes[i]
                    #sub_feat=features[sub_lab].type(torch.float)
                    sub_lab=sub_label[ori_knn_neighbor[i].tolist()]
                    sub_feat=sub_sum[sub_lab].type(torch.float)
                    sub_feat[0]=features[indexes[i]] #index itself

                    #A
                    A=sub_feat.mm(sub_feat.t())
                    A*=(torch.exp(all_pred[i,:,1]).unsqueeze(0).expand_as(A))#addweights
                    A=F.softmax(A,dim=1)

                    sub_feat/=torch.norm(sub_feat,dim=1,keepdim=True)
                    sub_feat-=sub_feat[0].clone() #X
                    all_x.append(sub_feat.unsqueeze(0))
                    all_adj.append(A.unsqueeze(0))

            all_x=torch.cat(all_x,dim=0)
            all_adj=torch.cat(all_adj,dim=0)

            x=self.conv1(all_x, all_adj)
            x=self.conv2(x, all_adj)
            dout=x.size(-1)
            x_0=x.view(-1,dout)
            all_pred=self.classifier(x_0) #B*topk*2

            if output_feat:
                simm=torch.sum(sub_sum*sub_sum,dim=1)

                all_pred=all_pred.view(len(indexes),-1,self.nclass)
                all_pred=F.softmax(all_pred,dim=2)
                return all_pred,simm#sub_sim_mean,sub_sim_var

            if return_loss: #TODO:add weights
                #all_pred0=all_pred.permute(1,2,0)
                #loss=torch.mean(self.loss(all_pred0[:,:,:self.cal_num],gt_conf[:,:self.cal_num])*gt_weights[:,:self.cal_num])
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

class Cluster_Level_GCN(nn.Module):
    def __init__(self, feature_dim, nhid, feature_size,source_classes,nclass, momentum=0.2,dropout=0,cal_num=30):
        super(Cluster_Level_GCN, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        self.conv2 = GraphConv(nhid, 256, MeanAggregator, dropout)

        self.nclass = 2
        self.classifier = nn.Sequential(nn.Linear(256, 256), nn.PReLU(256),
                                        nn.Linear(256, self.nclass))
        self.loss=torch.nn.CrossEntropyLoss(reduction='none').cuda()
        self.cal_num=cal_num
        self.source_classes=source_classes

    def forward(self, indexes, features,labels,domain,ori_0,ori_knn_neighbor,all_pred,gt_conf=None,gt_weights=None,output_feat=False, return_loss=False,min_point=3):
        if domain:
            all_x=[]
            all_adj=[]
            with torch.no_grad():
                clu_sum = torch.zeros(labels.max()+1, 2048).float().cuda()
                clu_sum.index_add_(0, labels, features)
                nums = torch.zeros(labels.max()+1, 1).float().cuda()
                nums.index_add_(0, labels, torch.ones(len(labels),1).float().cuda())
                mask = (nums>0).float()
                clu_sum /= (mask*nums+(1-mask)).clone().expand_as(clu_sum)

                for i in range(len(indexes)):
                    clu_lab=labels[ori_knn_neighbor[i].tolist()]
                    clu_feat=clu_sum[clu_lab].type(torch.float)
                    clu_feat[0]=features[indexes[i]]

                    A=clu_feat.mm(clu_feat.t())
                    A*=(torch.exp(all_pred[i,:,1]).unsqueeze(0).expand_as(A))#addweights
                    A=F.softmax(A,dim=1)

                    clu_feat/=torch.norm(clu_feat,dim=1,keepdim=True) #cluster feat
                    clu_feat-=clu_feat[0].clone()
                    all_x.append(clu_feat.unsqueeze(0))
                    all_adj.append(A.unsqueeze(0))

            all_x=torch.cat(all_x,dim=0)
            all_adj=torch.cat(all_adj,dim=0)
            x=self.conv1(all_x, all_adj)
            x= self.conv2(x,all_adj)
            dout=x.size(-1)
            x_0=x.view(-1,dout)
            all_pred=self.classifier(x_0)

            if output_feat:
                simm=torch.sum(clu_sum*clu_sum,dim=1)

                all_pred=all_pred.view(len(indexes),-1,self.nclass)
                all_pred=F.softmax(all_pred,dim=2)
                return all_pred,simm

            if return_loss: #TODO:add weights
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

class Hierarchy_GCN(object):
    def __init__(self,point_level_gcn,sub_cluster_level_gcn,cluster_level_gcn,utils,r=1,merge_label_func=0,point_add_dyn_thre=0,
                var_weights=2,mean_min_points=2,merge_dyn_ratio=1.1,cal_num=30,tgt_only=0,dyn_thre_type=0):
        self.point_level_gcn=point_level_gcn
        self.sub_cluster_level_gcn=sub_cluster_level_gcn
        self.cluster_level_gcn=cluster_level_gcn
        self.utils=utils #utils

        self.neighbor_num=64 #knn
        self.cal_num=cal_num
        self.r=r
        self.var_weights=var_weights
        self.mean_min_points=mean_min_points
        self.merge_label_func=merge_label_func
        self.point_add_dyn_thre=point_add_dyn_thre
        #######
        self.dyn_thre_type=dyn_thre_type
        self.merge_dyn_ratio=merge_dyn_ratio

        self.train_cluster_outliers_label=None
        self.train_sub_outliers_label=None

        self.src_only=1
        self.tgt_only=tgt_only

    def train(self,s_indexes,memory,train):
        if train:
            #knn neighbor
            if self.src_only:
                ori_0=compute_knn(memory.features[:memory.source_samples].clone(),k1=self.neighbor_num) #only consider src
            else:
                ori_0=compute_knn(memory.features.clone(),k1=self.neighbor_num)
            ori_knn_neighbor=ori_0[s_indexes.cpu().numpy(),:]

            #cal gt
            all_gt=[]
            all_sub_gt=[]
            for i in range(len(s_indexes)):
                all_gt.append(memory.labels[ori_knn_neighbor[i].tolist()].unsqueeze(0))
                all_sub_gt.append(memory.sub_label[ori_knn_neighbor[i].tolist()].unsqueeze(0))
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
                self.train_cluster_outliers_label=torch.arange(len(memory.labels)+1,len(memory.labels)+1+len(s_indexes)).cuda()
            if self.train_sub_outliers_label is None:
                self.train_sub_outliers_label=torch.arange(len(memory.sub_label)+1,len(memory.sub_label)+1+len(s_indexes)).cuda()

            ori_cluster_label=memory.labels[s_indexes]
            ori_sub_label=memory.sub_label[s_indexes]
            memory.labels[s_indexes]=self.train_cluster_outliers_label
            memory.sub_label[s_indexes]=self.train_sub_outliers_label
        else:
            gt_conf=None
            gt_weights=None
            ori_0=-1
            ori_knn_neighbor=-1

        if self.src_only:
            loss_point_level,all_pred=self.point_level_gcn(s_indexes,memory.features[:memory.source_samples],memory.labels[:memory.source_samples],train,ori_0,ori_knn_neighbor,self.neighbor_num,gt_conf=gt_conf,gt_weights=gt_weights,return_loss=True)
            loss_sub_level,all_pred_sub=self.sub_cluster_level_gcn(s_indexes,memory.features[:memory.source_samples],memory.labels[:memory.source_samples],memory.sub_label[:memory.source_samples],train,ori_0,ori_knn_neighbor,all_pred,gt_conf=gt_conf,gt_weights=gt_weights,return_loss=True)
            loss_cluster_level,all_pred_clu=self.cluster_level_gcn(s_indexes,memory.features[:memory.source_samples],memory.labels[:memory.source_samples],train,ori_0,ori_knn_neighbor,all_pred_sub,gt_conf=gt_conf,gt_weights=gt_weights,return_loss=True)
        else:
            loss_point_level,all_pred=self.point_level_gcn(s_indexes,memory.features,memory.labels,train,ori_0,ori_knn_neighbor,self.neighbor_num,gt_conf=gt_conf,gt_weights=gt_weights,return_loss=True)
            loss_sub_level,all_pred_sub=self.sub_cluster_level_gcn(s_indexes,memory.features,memory.labels,memory.sub_label,train,ori_0,ori_knn_neighbor,all_pred,gt_conf=gt_conf,gt_weights=gt_weights,return_loss=True)
            loss_cluster_level,all_pred_clu=self.cluster_level_gcn(s_indexes,memory.features,memory.labels,train,ori_0,ori_knn_neighbor,all_pred_sub,gt_conf=gt_conf,gt_weights=gt_weights,return_loss=True)

        if train:
            #change label back
            memory.labels[s_indexes]=ori_cluster_label
            memory.sub_label[s_indexes]=ori_sub_label

        loss=loss_point_level+loss_sub_level+loss_cluster_level
        return [loss_point_level,loss_sub_level,loss_cluster_level]

    def inference(self,t_indexes,memory,infer):
        if infer:
            if self.tgt_only:
                #print('tgt_only')
                #import pdb;pdb.set_trace()
                ori_0=compute_knn(memory.features[memory.source_samples:].clone(),k1=self.neighbor_num)
                ori_0+=memory.source_samples
                ori_0=np.concatenate((np.zeros((memory.source_samples,self.neighbor_num)),ori_0)).astype('int')
                ori_knn_neighbor=ori_0[t_indexes.cpu().numpy(),:]
            else:
                ori_0=compute_knn(memory.features.clone(),k1=self.neighbor_num)
                ori_knn_neighbor=ori_0[t_indexes.cpu().numpy(),:]

            #change sub_label include indexes
            ori_labels=memory.sub_label[t_indexes]
            change_sub_label=t_indexes[ori_labels==t_indexes]

            memory.sub_label[t_indexes]=0
            if len(change_sub_label)>0:
                all_label=torch.arange(len(memory.features)).cuda()
                for change_lab in change_sub_label:
                    change_idx=all_label[memory.sub_label==change_lab]
                    if len(change_idx)>0:
                        change_feat=memory.features[change_idx]
                        change_feat_sim=torch.mean(change_feat.mm(change_feat.t()),0)
                        memory.sub_label[change_idx]=change_idx[torch.argmax(change_feat_sim)]
            memory.sub_label[t_indexes]=t_indexes
            #change label
            empty_label=set(torch.arange(memory.source_samples,memory.labels.max()+1).tolist())-set(memory.labels.tolist())
            if len(empty_label)<len(t_indexes):
                outliers_label=torch.arange(memory.labels.max()+1,memory.labels.max()+1+len(t_indexes)).cuda()
            else:
                empty_label=list(empty_label)[-len(t_indexes):]
                outliers_label=torch.tensor(empty_label).cuda()
            memory.labels[t_indexes]=outliers_label

            all_pred=self.point_level_gcn(t_indexes,memory.features,memory.labels,infer,ori_0,ori_knn_neighbor,self.neighbor_num,output_feat=True)
            all_pred_sub,sub_inter_sim=self.sub_cluster_level_gcn(t_indexes,memory.features,memory.labels,memory.sub_label,infer,ori_0,ori_knn_neighbor,all_pred,output_feat=True,min_point=self.mean_min_points)
            all_pred_clu,clu_inter_sim=self.cluster_level_gcn(t_indexes,memory.features,memory.labels,infer,ori_0,ori_knn_neighbor,all_pred_sub,output_feat=True,min_point=self.mean_min_points)

            ori_knn_neighbor=torch.from_numpy(ori_knn_neighbor)
            all_pred=all_pred.detach()
            all_pred_sub=all_pred_sub.detach()
            all_pred_clu=all_pred_clu.detach()
            neighbor_index,nearest_neighbor,sim,sim_cluster=self.point_level_merge_split(t_indexes,all_pred,ori_knn_neighbor,memory)
            self.sub_cluster_level_merge_split(t_indexes,memory,neighbor_index,all_pred_sub,nearest_neighbor,ori_knn_neighbor,sim,sub_inter_sim)
            self.cluster_level_merge_split(t_indexes,memory,neighbor_index,sim_cluster,all_pred_clu,nearest_neighbor,ori_knn_neighbor,outliers_label,clu_inter_sim)

    def point_level_merge_split(self,indexes,all_pred,ori_knn_neighbor,memory):
        #print('point_level_merge_split')
        sim_thre=0.3
        neighbor_thre=0.45

        dyn_thre=torch.mean(all_pred[:,:self.cal_num,1])

        max_conf,max_label=torch.max(all_pred[:,1:5,1],1) #only consider top5 neighbor as candidate
        nearest_neighbor=ori_knn_neighbor[torch.arange(len(indexes)),max_label+1].view(-1)
        #print('max_conf:',max_conf.tolist())

        #sim
        new_label=memory.sub_label[ori_knn_neighbor[torch.arange(len(indexes)),max_label+1].view(-1).cuda()]
        new_cluster_label=memory.labels[ori_knn_neighbor[torch.arange(len(indexes)),max_label+1].view(-1).cuda()]

        sim = torch.zeros((memory.sub_label).max()+1, (memory.features[indexes]).size(0)).float().cuda()
        feature_sim=(memory.features[indexes]).mm(memory.features.t())
        sim.index_add_(0, memory.sub_label, feature_sim.t().contiguous())
        nums = torch.zeros((memory.sub_label).max()+1, 1).float().cuda()
        nums.index_add_(0, memory.sub_label, torch.ones(len(memory.sub_label),1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        sim[:memory.source_samples,:]=0
        self.sub_nums=nums

        sim_cluster = torch.zeros((memory.labels).max()+1, (memory.features[indexes]).size(0)).float().cuda()
        sim_cluster.index_add_(0, memory.labels, feature_sim.t().contiguous())
        nums_cluster = torch.zeros((memory.labels).max()+1, 1).float().cuda()
        nums_cluster.index_add_(0, memory.labels, torch.ones(len(memory.labels),1).float().cuda())
        mask = (nums_cluster>0).float()
        sim_cluster /= (mask*nums_cluster+(1-mask)).clone().expand_as(sim_cluster)
        sim_cluster[:memory.source_samples,:]=0
        self.clu_nums=nums_cluster

        label_sim=sim[new_label,torch.arange(len(indexes))]
        neigh_sim=feature_sim[torch.arange(len(indexes)),nearest_neighbor]

        new_label[label_sim<=sim_thre]=0
        new_label[new_label<memory.source_samples]=0
        new_label[neigh_sim<neighbor_thre]=0
        if self.point_add_dyn_thre:
            new_label[max_conf<dyn_thre]=0
        #print('new label:',new_label.tolist())
        #nei_sim=torch.sum(memory.features[indexes]*memory.features[nearest_neighbor.cuda()],1)#closest sim should larger than neighbor_thre
        #new_label[neigh_sim<neighbor_thre]=0

        #change sub cluster label & cluster label
        new_cluster_label[new_label==0]=memory.labels[indexes][new_label==0]
        memory.labels[indexes]=new_cluster_label
        new_label[new_label==0]=indexes[new_label==0] #outliers
        memory.sub_label[indexes]=new_label#new_label

        return max_label+1,nearest_neighbor,sim,sim_cluster

    def sub_cluster_level_merge_split(self,indexes,memory,neighbor_index,all_pred_sub,nearest_neighbor,ori_knn_neighbor,sim,sub_inter_sim):
        if self.dyn_thre_type==0:
            merge_sim_thre=0.65
            split_sim_thre=0.55
        if self.dyn_thre_type==1:
            self.sub_nums[:memory.source_samples]=1
            sub_inter_sim=sub_inter_sim[self.sub_nums.view(-1)>=self.mean_min_points]
            sub_sim_mean=torch.mean(sub_inter_sim)
            sub_sim_var=torch.var(sub_inter_sim)
            merge_sim_thre=sub_sim_mean+self.var_weights*sub_sim_var
            split_sim_thre=sub_sim_mean-self.var_weights*sub_sim_var
            #print('sub merge: {}, sub_split: {}'.format(merge_sim_thre,split_sim_thre))
            merge_sim_thre=max(merge_sim_thre,0.6)
            split_sim_thre=min(split_sim_thre,0.6)
        if self.dyn_thre_type==2:
            merge_dyn_ratio=1.1#sim/sub_inter_sim
            split_dyn_ratio=0.9
            merge_sim_thre=(sub_inter_sim*merge_dyn_ratio).unsqueeze(1).expand_as(sim)
            split_sim_thre=(sub_inter_sim*split_dyn_ratio).unsqueeze(1).expand_as(sim)
            merge_sim_thre=torch.clamp(merge_sim_thre,0.6,1.0)
            split_sim_thre=torch.clamp(split_sim_thre,0.0,0.6)
        # merge_sim_thre=0.65
        # split_sim_thre=0.55
        #dynamic thre
        dyn_thre=torch.mean(all_pred_sub[:,:self.cal_num,1])#+torch.var(all_pred_sub[:,:,1])
        neighbor_prediction=all_pred_sub[torch.arange(len(indexes)),neighbor_index,1]

        #>dyn_thre-->merge
        new_label=memory.sub_label[indexes]
        neighbor_prediction[neighbor_prediction<dyn_thre]=0
        neighbor_prediction[new_label!=memory.sub_label[nearest_neighbor]]=-1 #not equal-->outlier

        #debug
        # if self.dyn_thre_type==2:
        #     merge_debug=merge_sim_thre[memory.t_sub_label[indexes],torch.arange(len(indexes))][neighbor_prediction>=0]
        #     print('sub sim thre:',merge_debug.tolist())

        merge_labels=new_label[neighbor_prediction>0]
        if len(merge_labels)>0:
            merge_batch=indexes[neighbor_prediction>0]
            #add sim thre
            sim_lab=torch.gt(sim,merge_sim_thre)[:,neighbor_prediction>0]
            #wo calculation of current batch
            tmp=sim_lab[memory.sub_label[merge_batch],torch.arange(len(merge_batch)).cuda()]
            sim_lab[indexes,:]=0
            sim_lab[memory.sub_label[merge_batch],torch.arange(len(merge_batch)).cuda()]=tmp

            pred=torch.gt(all_pred_sub[neighbor_prediction>0,:,1],dyn_thre)
            merge_neighbors=ori_knn_neighbor[(neighbor_prediction>0).cpu()]
            merge_map={}

            if self.merge_label_func>0:
                merge_label_map={}
            for idx,lab in enumerate(merge_labels):
                merge_idx=memory.sub_label[merge_neighbors[idx]]
                sim_idx=sim_lab[memory.sub_label[merge_neighbors[idx]],idx]
                merge_idx=list(set(merge_idx[(pred[idx]>0) & (sim_idx>0)].tolist()))
                #print('sub sim_idx:',sim_idx.tolist())
                #print('sub pred_idx:',pred[idx].tolist())
                if memory.sub_label[merge_batch[idx]].item() not in merge_idx:
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
                    if self.merge_label_func==1:
                        merge_cluster_label0=memory.labels[merge_batch[idx]].item()
                        for label in merge_idx:
                            merge_label_map[label]=merge_cluster_label0


            print("sub cluster merge:",len(merge_map.keys()))
            for change_label,update_label in merge_map.items():
                memory.sub_label[memory.sub_label==int(change_label)]=int(update_label)
            ###TODO:merge sub label in different label
            if self.merge_label_func==1:
                for change_label,update_label in merge_label_map.items():
                    memory.labels[memory.sub_label==int(change_label)]=int(update_label)

        #<dyn_thre-->split
        split_idxs=torch.arange(len(new_label))[neighbor_prediction==0].cuda()
        #only consider label sim<=split sim thre
        sim_lab=torch.le(sim,split_sim_thre)[:,neighbor_prediction==0]
        if len(split_idxs)>0:
            split_labels=new_label[split_idxs]
            split_idxs=split_idxs[sim_lab[split_labels,torch.arange(len(split_labels)).cuda()]]

        if len(split_idxs)>0:
            split_labels=new_label[split_idxs]
            split_nums=[]
            split_nearest=nearest_neighbor[split_idxs]
            split_batch=indexes[split_idxs]
            all_idx=torch.arange(len(memory.sub_label))
            for idx,lab in enumerate(split_labels):
                split_indexes=all_idx[(memory.sub_label==lab).cpu()].cuda()
                #memory.t_sub_label[split_indexes]=split_indexes
                memory.sub_label[split_nearest[idx]]=split_batch[idx]
                memory.sub_label[split_batch[idx]]=split_batch[idx]
                split_nums.append(len(split_indexes))
            print('sub cluster split nums:',split_nums)
        del sim_lab

    def cluster_level_merge_split(self,indexes,memory,neighbor_index,sim,all_pred_clu,nearest_neighbor,ori_knn_neighbor,outliers_label,clu_inter_sim):
        if self.dyn_thre_type==0:
            merge_sim_thre=0.55
            split_sim_thre=0.45
        if self.dyn_thre_type==1:
            self.clu_nums[:memory.source_samples]=1
            clu_inter_sim=clu_inter_sim[self.clu_nums.view(-1)>=self.mean_min_points]
            clu_sim_mean=torch.mean(clu_inter_sim)
            clu_sim_var=torch.var(clu_inter_sim)
            merge_sim_thre=clu_sim_mean+self.var_weights*clu_sim_var
            split_sim_thre=clu_sim_mean-self.var_weights*clu_sim_var
            merge_sim_thre=max(merge_sim_thre,0.5)
            split_sim_thre=min(split_sim_thre,0.5)
            print('merge: {} split:{}'.format(merge_sim_thre,split_sim_thre))
        if self.dyn_thre_type==2:
            merge_dyn_ratio=1.1#sim/sub_inter_sim
            split_dyn_ratio=0.9
            merge_sim_thre=(clu_inter_sim*merge_dyn_ratio).unsqueeze(1).expand_as(sim)
            split_sim_thre=(clu_inter_sim*split_dyn_ratio).unsqueeze(1).expand_as(sim)
            merge_sim_thre=torch.clamp(merge_sim_thre,0.5,1.0)
            split_sim_thre=torch.clamp(split_sim_thre,0.0,0.5)
        #print('clu merge: {}, clu split: {}'.format(merge_sim_thre,split_sim_thre))

        # merge_sim_thre=0.55
        # split_sim_thre=0.45

        #dynamic thre
        #dyn_thre=all_pred_clu[:,0,1]-torch.var(all_pred_clu[:,:self.cal_num,1])
        dyn_thre=torch.mean(all_pred_clu[:,:self.cal_num,1])#+torch.var(all_pred_sub[:,:,1])
        neighbor_prediction=all_pred_clu[torch.arange(len(indexes)),neighbor_index,1]

        #>dyn_thre-->merge
        new_label=memory.labels[indexes]
        try:
            neighbor_prediction[neighbor_prediction<dyn_thre]=0
        except:
            print('error')
            import pdb;pdb.set_trace()
        neighbor_prediction[new_label!=memory.labels[nearest_neighbor]]=-1 #not equal-->outliers

        #debug
        if self.dyn_thre_type==2:
            merge_debug=merge_sim_thre[memory.labels[indexes],torch.arange(len(indexes))][neighbor_prediction>=0]
            split_debug=split_sim_thre[memory.labels[indexes],torch.arange(len(indexes))][neighbor_prediction>=0]
            print('clu sim merge thre:',merge_debug.tolist())
            print('clu sim split thre:',split_debug.tolist())

        merge_labels=new_label[neighbor_prediction>0]
        if len(merge_labels)>0:
            merge_batch=indexes[neighbor_prediction>0]
            #add sim thre
            sim_lab_clu=(torch.gt(sim,merge_sim_thre)[:,neighbor_prediction>0]).cpu()
            #wo current batch
            tmp=sim_lab_clu[(memory.labels[merge_batch]).cpu(),torch.arange(len(merge_batch))]
            #print('tmp.shape:',tmp.shape)
            try:
                sim_lab_clu[outliers_label.cpu(),:]=0
                sim_lab_clu[(memory.labels[merge_batch]).cpu(),torch.arange(len(merge_batch))]=tmp.clone()
            except:
                print('outofbound')
                import pdb;pdb.set_trace()
            sim_lab_clu=sim_lab_clu.cuda()

            #pred=torch.gt(all_pred_clu[:,:,1],dyn_thre.unsqueeze(1).expand_as(all_pred_clu[:,:,1]))[neighbor_prediction>0]
            pred=torch.gt(all_pred_clu[:,:,1],dyn_thre)[neighbor_prediction>0]
            merge_neighbors=ori_knn_neighbor[(neighbor_prediction>0).cpu()]

            merge_map={}

            for idx,lab in enumerate(merge_labels):
                merge_idx=memory.labels[merge_neighbors[idx]]
                sim_idx=sim_lab_clu[memory.labels[merge_neighbors[idx]],idx]
                #sim_idx[self.cal_num:]=0 #wo consider guys behind
                pred_idx=pred[idx]
                # print('cluster sim_idx:',sim_idx.tolist())
                # print('cluster pred_idx:',pred_idx.tolist())
                merge_idx=list(set(merge_idx[(sim_idx>0) & (pred_idx>0)].tolist()))
                #print('cluster merge idx:',merge_idx)
                if memory.labels[merge_batch[idx]].item() not in merge_idx:
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

            print("cluster merge:",len(merge_map.keys()))
            for change_label,update_label in merge_map.items():
                memory.labels[memory.labels==int(change_label)]=int(update_label)

        #<dyn_thre-->split by sub cluster
        split_idxs=torch.arange(len(indexes))[neighbor_prediction==0].cuda()
        sim_lab=torch.le(sim,split_sim_thre)[:,neighbor_prediction==0]

        if len(split_idxs)>0:
            split_labels=new_label[split_idxs]
            split_idxs=split_idxs[sim_lab[split_labels,torch.arange(len(split_labels)).cuda()]]

        if len(split_idxs)>0:
            split_nums=[]
            split_labels=new_label[split_idxs]
            split_nearest=nearest_neighbor[split_idxs]
            split_batch=indexes[split_idxs]
            split_nearest_sub_label=memory.sub_label[split_batch]
            all_idx=torch.arange(len(memory.labels))

            empty_labels=set(torch.arange(memory.labels.max()+1).tolist())-set(memory.labels.tolist())
            if len(empty_labels)<len(split_labels):
                outliers_label=torch.arange(memory.labels.max()+1,memory.labels.max()+1+len(split_labels)).cuda()
            else:
                empty_labels=list(empty_labels)[-len(split_labels):]
                outliers_label=torch.tensor(empty_labels).cuda()

            for idx,lab in enumerate(split_labels):
                split_indexes=all_idx[((memory.labels==lab).cpu()) & (memory.sub_label==split_nearest_sub_label[idx]).cpu()].cuda()
                memory.labels[split_indexes]=outliers_label[idx]
                split_nums.append(len(split_indexes))
            print('cluster split nums:',split_nums)

        #deal with different label in the same sub cluster-->merge_label_func=2
        # new_sub_label=list(set(memory.t_sub_label[indexes].tolist()))
        # for lab in new_sub_label:
        #     cluster_labs=set(memory.labels[memory.t_sub_label==lab].tolist())
        #     if len(cluster_labs)>1:
        #         cluster_labs=list(cluster_labs)
        #         print('cluster_labs:',cluster_labs)
        #         merge_lab=cluster_labs[0]
        #         for cluster_l in cluster_labs:
        #             memory.labels[memory.labels==cluster_l]=merge_lab

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


class Utils(object): #in order to update * in memory
    def __init__(self,k1,k2,thre):
        self.k1=k1
        self.k2=k2
        self.thre=thre

        self.density_sim=0.5
        self.density_core_thre=0.7 #point sim>thre-->the same core

    def initialize_sub_cluster_label(self,ori_label,ori_sub_cluster_label,ori_features,start=0,end=-1):
        print('initialize sub cluster bank...')
        features=ori_features[start:end]
        label=ori_label[start:end]
        sub_cluster_label=ori_sub_cluster_label[start:end]

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

    def initialize_sub_cluster_label_ori(self,ori_label,ori_sub_cluster_label,ori_features,start=0,end=-1): #initialize
        print('initialize sub cluster bank...')
        #compute density
        features=ori_features[start:end]
        label=ori_label[start:end]
        sub_cluster_label=ori_sub_cluster_label[start:end]

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

        ori_sub_cluster_label[start:end]=sub_cluster_label #update
        del sim,rerank_dist
        print('Done')

    def update_sub_cluster_label(self,indexes,memory): #update online
        index_label=list(set(memory.labels[indexes].tolist()))
        all_idx=torch.arange(len(memory.labels)).cuda()
        #update sub label for these lael
        core_nums=[]
        for label in index_label:
            i_idx=all_idx[(memory.labels==label)]
            feat=memory.features[i_idx]
            sim=feat.mm(feat.t())
            #density
            i_density=torch.sum(torch.gt(sim,self.density_sim),dim=1)
            if torch.sum(torch.ge(i_density,self.thre))>0:
                i_density_core=torch.gt(sim,self.density_core_thre)
                high_density_idx=i_idx[i_density>=self.thre]
                i_density_core=i_density_core[i_density>=self.thre][:,i_density>=self.thre]

                memory.sub_label[i_idx]=-1

                for left_id in range(len(high_density_idx)):
                    neighbor=high_density_idx[i_density_core[left_id]>0]
                    if len(neighbor)>1:
                        neighbor_label=memory.sub_label[neighbor]
                        neighbor_label=neighbor_label[neighbor_label>-1]
                        if len(neighbor_label)>0:
                            memory.sub_label[high_density_idx[left_id].item()]=neighbor_label[0].item()
                        else:
                            memory.sub_label[high_density_idx[left_id].item()]=high_density_idx[left_id].item()
                    else:
                        memory.sub_label[high_density_idx[left_id].item()]=high_density_idx[left_id].item()

                #others
                sim[:,i_density<self.thre]=-1
                match=torch.argmax(sim,dim=1)
                memory.sub_label[i_idx]=memory.sub_label[i_idx[match]]
                core_nums.append(len(set(memory.sub_label[i_idx].tolist())))
            else:
                match=torch.argmax(i_density)
                memory.sub_label[i_idx]=i_idx[match]
                core_nums.append(1)

        #print('core_nums:',core_nums)
