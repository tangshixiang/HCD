#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from examples.utils import GraphConv, MeanAggregator
from spcl.utils.faiss_rerank import compute_jaccard_distance,compute_knn


class GCN_V(nn.Module):
    def __init__(self, feature_dim, nhid, nclass, dropout=0):
        super(GCN_V, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)

        self.nclass = nclass
        self.classifier = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(nhid),
                                        nn.Linear(nhid, self.nclass))
        self.loss = torch.nn.MSELoss()

    def forward(self, data, output_feat=False, return_loss=False):
        assert not output_feat or not return_loss
        x, adj = data[0], data[1]
        x = self.conv1(x, adj)
        pred = self.classifier(x).view(-1)

        if output_feat:
            return pred, x

        if return_loss:
            label = data[2]
            loss = self.loss(pred, label)
            return pred, loss

        return pred

class GCN_V_UDA(nn.Module):
    def __init__(self, feature_dim, nhid, feature_size,source_classes,nclass, momentum=0.2,dropout=0):
        super(GCN_V_UDA, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)

        self.nclass = 1
        self.classifier = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(nhid),
                                        nn.Linear(nhid, self.nclass))
        #self.confidence_weights=torch.tensor([1.0,0.8,0.6,0.4,0.2]).cuda()
        self.momentum = momentum
        self.loss = torch.nn.MSELoss()
        #self.loss=torch.nn.BCELoss()
        #self.confidence=0.5*torch.ones(feature_size)
        #self.confidence[:source_classes]=1
        self.neighbor_num=64
        self.batchsize=64

    def forward(self, indexes, features,labels,domain,output_feat=False, return_loss=False):
        if domain:
            with torch.no_grad():
                adj=feature.mm(feature.t())
                adj=F.softmax(adj,dim=1)


                topk=10
                #knn_neighbor = ori_knn_neighbor.reshape(-1)
                #x=features[torch.from_numpy(knn_neighbor).cuda()]
                #adj=x.mm(x.t())
                for i in range(len(indexes)):
                    all_neighbor=ori_knn_neighbor[i].tolist()
                    feature=features[all_neighbor].type(torch.float)
                    all_neighbors.append(all_neighbor)


                    all_x.append(feature)
                    all_adj.append(adj)
            #x = self.conv1(x, adj)
            for i in range(len(indexes)):
                x = self.conv1(all_x[i], all_adj[i])
                pred = self.classifier(x)
                all_pred.append(pred)

            if output_feat:
                return all_neighbors,all_pred
                #return pred, ori_knn_neighbor,ori_0

            if return_loss:
                with torch.no_grad():
                    ori_knn_neighbor=compute_knn(features.clone(),k1=self.neighbor_num)
                    #import pdb;pdb.set_trace()
                    # x_label=labels[torch.from_numpy(knn_neighbor).cuda()].view(-1,1).repeat(1,len(knn_neighbor))
                    # #gcn_label=( x_label == x_label.t() ).float()
                    # gcn_label=torch.where(x_label == x_label.t(),torch.ones_like(x_label).float(),-1*torch.ones_like(x_label).float())
                    # gcn_label=gcn_label*adj
                    # #gcn_label=gcn_label[torch.arange(0,640,self.neighbor_num).cuda(),:]
                    # gt_conf=torch.sum(gcn_label,dim=1)/gcn_label.size(0)
                    #print("gt_conf:",gt_conf[:10])
                    gt_conf=[]
                    for i in range(len(indexes)):
                        try:
                            x_label=labels[all_neighbors[i]].view(-1,1).repeat(1,self.neighbor_num)
                            adj=all_ori_adj[i]
                            gcn_label=torch.where(x_label == x_label.t(),torch.ones_like(x_label).float(),-1*torch.ones_like(x_label).float())
                            gcn_label=gcn_label*adj
                            conf=torch.sum(gcn_label,dim=1)/gcn_label.size(0)
                            gt_conf.append(conf)
                        except:
                            print('label')
                            import pdb;pdb.set_trace()
                    print(gt_conf[0][:10])
                loss=0
                for i in range(len(indexes)):
                    loss+=self.loss(all_pred[i],gt_conf[i])
                #loss = self.loss(pred, gt_conf)
                loss/=len(indexes)
                return loss,all_neighbors,all_pred

            return pred
        else:
            return torch.tensor(0).cuda(),0,0

class GCN_estimator(nn.Module):
    def __init__(self, feature_dim, nhid, feature_size,source_classes,nclass, momentum=0.2,dropout=0):
        super(GCN_estimator, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        self.conv2 = GraphConv(nhid, 512, MeanAggregator, dropout)
        self.conv3 = GraphConv(512, 256, MeanAggregator, dropout)
        self.conv4 = GraphConv(256, 256, MeanAggregator, dropout)

        self.nclass = 2
        self.classifier = nn.Sequential(nn.Linear(256, 256), nn.PReLU(256),
                                        nn.Linear(256, self.nclass))
        self.momentum = momentum
        #self.loss = torch.nn.MSELoss()
        self.neighbor_num=64
        self.class_balance=0
        self.k_reciprocal=0
        self.neighbor_A=1
        self.softmax_norm=1
        self.baseline=0
        self.cal_num=30
        self.batchsize=64
        self.adddensity=1
        if self.baseline:
            self.loss=torch.nn.CrossEntropyLoss(reduction='none').cuda()
        else:
            self.loss=torch.nn.CrossEntropyLoss().cuda()

    def forward(self, indexes, features,labels,domain,output_feat=False, return_loss=False):
        #in order to select high confidence neighbors
        if domain:
            all_pred=[]

            with torch.no_grad():
                all_x=[]
                all_adj=[]
                all_neighbors=[]
                ori_0=compute_knn(features.clone(),k1=self.neighbor_num)
                ori_knn_neighbor=ori_0[indexes.cpu().numpy(),:]
                topk=10
                all_index_dict={}
                #knn_neighbor=torch.from_numpy(ori_knn_neighbor).cuda()
                if self.adddensity:
                    density_neighbor=10
                    #cal density top10sum
                    sim_neighbor=torch.from_numpy(ori_0[:,:density_neighbor])
                    all_sim=torch.zeros(len(features)).cuda()
                    for i in range(density_neighbor):
                        all_sim+=torch.sum(torch.mul(features,features[sim_neighbor[:,i]]),dim=1)
                    all_sim/=(4*density_neighbor)

                for i in range(len(indexes)):
                    all_neighbor=ori_knn_neighbor[i].tolist()
                    if self.baseline:
                        k1=30
                        k2=5
                        all_neighbor=np.unique(ori_0[ori_knn_neighbor[i][:k1],:k2].reshape(-1)).tolist()
                        index_dict={}
                        for idx,nei in enumerate(all_neighbor):
                            index_dict[nei]=idx
                        all_index_dict[i]=index_dict
                        feature=features[all_neighbor].type(torch.float)
                        all_neighbors.append(all_neighbor)

                        if self.softmax_norm:
                            #import pdb;pdb.set_trace()
                            adj=feature.mm(feature.t())
                            adj=F.softmax(adj,dim=1)
                        else:
                            adj=torch.zeros((len(all_neighbor),len(all_neighbor))).float().cuda()
                            for node in ori_knn_neighbor[i][:k1]:
                                for node2 in ori_0[node][:k2]:
                                    if not self.k_reciprocal:
                                        adj[index_dict[node]][index_dict[node2]]=1
                                        adj[index_dict[node2]][index_dict[node]]=1
                                    else:
                                        if node in ori_0[node2][:k2]:
                                            adj[index_dict[node]][index_dict[node2]]=1
                                            adj[index_dict[node2]][index_dict[node]]=1

                            D= adj.sum(1, keepdim=True)
                            adj = adj.div(D)
                    elif not self.neighbor_A:
                        feature=features[all_neighbor].type(torch.float)
                        all_neighbors.append(all_neighbor)
                        adj=feature.mm(feature.t())
                    else:
                        #import pdb;pdb.set_trace()
                        #print('-------hei----------')

                        feature=features[all_neighbor].type(torch.float)
                        all_neighbors.append(all_neighbor)
                        neigh=torch.from_numpy(ori_0[all_neighbor,:topk]).cuda()
                        tmp=torch.zeros(self.neighbor_num,len(features)).float().cuda()
                        #opt1
                        #tmp[torch.arange(self.neighbor_num).view(-1,1).repeat(1,topk),neigh]=1
                        #opt2
                        A=feature.mm(features.t())
                        if self.adddensity:
                            #density_matrix=all_sim[all_neighbor].mm(all_sim.t())
                            exp=1
                            if exp:
                                A=A*(torch.ones_like(all_sim[all_neighbor].unsqueeze(1)).mm(torch.exp(all_sim.unsqueeze(1).t())))
                            else:
                                A=A*(all_sim[all_neighbor].unsqueeze(1).mm(all_sim.unsqueeze(1).t()))
                        tmp[torch.arange(self.neighbor_num).view(-1,1).repeat(1,topk),neigh]=A[torch.arange(self.neighbor_num).view(-1,1).repeat(1,topk),neigh]
                        adj=tmp.mm(torch.gt(tmp.t(),0).float())
                        #import pdb;pdb.set_trace()

                        adj=F.softmax(adj,dim=1)
                        #adj/=adj[torch.arange(self.neighbor_num),torch.arange(self.neighbor_num)].view(-1,1)

                    #D = adj.sum(1, keepdim=True)
                    #adj = adj.div(D)
                    feature-=feature[0]
                    all_x.append(feature)
                    all_adj.append(adj)
            #import pdb;pdb.set_trace()
            for i in range(len(indexes)):
                x = self.conv1(all_x[i], all_adj[i])
                x= self.conv2(x,all_adj[i])
                x= self.conv3(x,all_adj[i])
                x= self.conv4(x,all_adj[i])
                #import pdb;pdb.set_trace()
                pred = self.classifier(x)
                all_pred.append(pred)

            if output_feat:
                all_pred=[F.softmax(pred, dim=1) for _,pred in enumerate(all_pred)]
                return all_neighbors,all_pred,ori_knn_neighbor

            if return_loss:
                with torch.no_grad():
                    gt_conf=[]
                    #gt_one=[]
                    for i in range(len(indexes)):
                        #x_label=labels[all_neighbors[i]].view(-1,1).repeat(1,len(all_neighbors[i]))
                        x_label=labels[all_neighbors[i]]
                        gt=torch.where(x_label==labels[indexes[i]],torch.ones_like(x_label),torch.zeros_like(x_label))
                        #gcn_label=torch.where(x_label == x_label.t(),torch.ones_like(x_label).float(),-1*torch.ones_like(x_label).float())
                        #gcn_label=gcn_label*all_adj[i]
                        #gt=torch.sum(gcn_label,dim=1)/gcn_label.size(0)
                        #gt_one.append(torch.sum(gt[:self.cal_num]).item())
                        gt_conf.append(gt.long())
                    #print('gt sum:',gt_one)
                    #print('gt:',gt_conf[0][:5])
                    #print('pred:',all_pred[0][:5])
                loss=0
                for i in range(len(indexes)):
                    if self.baseline:
                        try:
                            with torch.no_grad():
                                index_dict=all_index_dict[i]
                                mul=torch.zeros((len(all_pred[i]))).float()
                                for nei in ori_knn_neighbor[i][:k1]:
                                    mul[index_dict[nei]]=1
                                mul=mul.cuda()
                            loss+=torch.sum(self.loss(all_pred[i],gt_conf[i])*mul)/torch.sum(mul)
                        except:
                            print('loss error')
                            import pdb;pdb.set_trace()
                    elif not self.class_balance:
                        loss += self.loss(all_pred[i][:self.cal_num], gt_conf[i][:self.cal_num])
                    else:
                        #print('choose')
                        #import pdb;pdb.set_trace()
                        pick_list=[]
                        pos_list=torch.arange(self.neighbor_num)[gt==1]
                        neg_list=torch.arange(self.neighbor_num)[gt==0]
                        pos_num=min(len(pos_list),self.cal_num//2)
                        neg_num=self.cal_num-pos_num
                        choose_list=[]
                        choose_list.extend(pos_list[:pos_num].tolist())
                        choose_list.extend(neg_list[:neg_num].tolist())
                        loss+=self.loss(all_pred[i][choose_list],gt_conf[i][choose_list])
                    #import pdb;pdb.set_trace()
                loss/=len(indexes)
                #print('loss:',loss)
                return loss,all_neighbors,all_pred

            return all_pred
        else:
            return torch.tensor(0).cuda(),0,0

class GCN_L(nn.Module):#selector
    def __init__(self, feature_dim, nhid, feature_size,source_classes,nclass, momentum=0.2,dropout=0):
        super(GCN_L, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)

        self.nclass = 2
        self.classifier = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(nhid),
                                        nn.Linear(nhid, self.nclass))
        self.momentum = momentum
        self.loss = torch.nn.CrossEntropyLoss()
        self.classifi=torch.nn.Softmax(dim=1)
        self.batchsize=64
        self.topk=2
        self.max_cluster_num=100

    def forward(self, indexes, features,labels,domain,neighbors,all_pred,output_feat=False, return_loss=False):
        #return max label
        if domain:
            loss=0
            all_label=[]
            for i in range(len(indexes)):
               #import pdb;pdb.set_trace()
               neighbor=torch.tensor(neighbors[i][:self.topk]).cuda()
               pred=all_pred[i]
               neighbor_label=torch.unique(labels[neighbor[(pred[:self.topk]>=pred[0]).view(-1)]]) #keep high conf neighbors
               max_conf=-1
               max_label=0
               ori_label=labels[indexes[i]]
               labels[indexes[i]]=-1
               for lab in neighbor_label:
                   #print("sum:",torch.sum(labels==lab))
                   with torch.no_grad():
                       lab_feature=features[labels==lab]
                       if len(lab_feature)>self.max_cluster_num:
                           #import pdb;pdb.set_trace()
                           ll=torch.randperm(len(lab_feature))[:self.max_cluster_num]
                           lab_feature=lab_feature[ll]
                   if len(lab_feature)>0:
                       x=torch.cat((features[indexes[i]].unsqueeze(0),lab_feature))
                       adj=x.mm(x.t())
                       x = self.conv1(x, adj)
                       pred = self.classifier(x)
                       #pred=self.classifi(pred)
                       if output_feat:
                           if pred[0][1]>pred[0][0] and pred[0][1]>max_conf:
                               max_conf=pred[0][1]
                               max_label=lab.item()
                       else:
                           if labels[indexes[i]]==lab:
                               gt=torch.tensor([1]).cuda()
                           else:
                               gt=torch.tensor([0]).cuda()
                           loss+=self.loss(pred[0].unsqueeze(0),gt)
                           print('pred[0]:',pred[0],' gt:',gt)
               all_label.append(max_label)
               labels[indexes[i]]=ori_label
            if output_feat:
                return all_label
            else:
                loss/=len(indexes)
                #print('loss:',loss)
                return loss*10


        #else:
        return torch.tensor(0).cuda()

def graph_estimator(feature_dim, nhid,feature_size, source_classes,nclass=1, dropout=0., **kwargs):
    model = GCN_estimator(feature_dim=feature_dim,
                  nhid=nhid,
                  feature_size=feature_size,
                  source_classes=source_classes,
                  nclass=nclass,
                  dropout=dropout)
    model.cuda()
    #model = nn.DataParallel(model)
    return model

def graph_selection(feature_dim, nhid,feature_size, source_classes,nclass=1, dropout=0., **kwargs):
    model = GCN_L(feature_dim=feature_dim,
                  nhid=nhid,
                  feature_size=feature_size,
                  source_classes=source_classes,
                  nclass=nclass,
                  dropout=dropout)
    model.cuda()
    #model = nn.DataParallel(model)
    #model.to(device)
    return model


def gcn_n(feature_dim, nhid,feature_size, source_classes,nclass=1, dropout=0., **kwargs):
    model = GCN_V_UDA(feature_dim=feature_dim,
                  nhid=nhid,
                  feature_size=feature_size,
                  source_classes=source_classes,
                  nclass=nclass,
                  dropout=dropout)
    model.cuda()
    #model = nn.DataParallel(model)
    return model
