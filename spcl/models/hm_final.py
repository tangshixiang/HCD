import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd
from spcl.utils.faiss_rerank import compute_jaccard_distance_inital_rank,compute_jaccard_distance_inital_rank_index,compute_knn
from collections import defaultdict


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, domain,labels,source_classes,num_samples,momentum,changelabel_thre,k1,k2,change_cnt,label_cache,confidence,gcn_n,gcn_s):
        ctx.features = features
        ctx.momentum = momentum
        ctx.domain=domain
        ctx.change_cnt=change_cnt
        ctx.source_classes=source_classes
        ctx.save_for_backward(inputs, indexes)

        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        ctx.change_cnt[indexes]+=1
        return grad_inputs, None, None, None,None, None, None,None, None, None,None,None,None,None,None,None


def hm(inputs, indexes, features,domain,labels,source_classes,num_samples,momentum=0.5,changelabel_thre=0.3,k1=10,k2=1,change_cnt=None,label_cache=None,confidence=None,gcn_n=None,gcn_s=None):
    return HM.apply(inputs, indexes, features,domain, labels,source_classes,num_samples,torch.Tensor([momentum]).to(inputs.device),changelabel_thre,k1,k2,change_cnt,label_cache,confidence,gcn_n,gcn_s)


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, source_classes,source_samples,temp=0.05, momentum=0.2,changelabel_thre=0.3,cluster_k1=10,cluster_k2=1,iterative=2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.source_classes=source_classes
        self.source_samples=source_samples

        self.momentum = momentum
        self.temp = temp
        #for clustering
        self.changelabel_thre=changelabel_thre
        self.cluster_k1=cluster_k1
        self.cluster_k2=cluster_k2

        self.register_buffer('features', torch.zeros(num_samples, num_features))

        self.register_buffer('label_cache', torch.zeros(1).long()) #not use now
        self.register_buffer('change_cnt', torch.zeros(num_samples).long())
        self.iterative=iterative

        self.labels = []
        for i in range(iterative):
            self.labels.append(torch.zeros(num_samples).long().cuda())


    def forward(self, inputs, indexes,domain=0,gcn_n=None,gcn_s=None):#domain=0:source domain=1:target
        # inputs: B*2048, features: L*2048
        inputs= hm(inputs, indexes, self.features, domain,self.labels,self.source_classes,self.num_samples,self.momentum,self.changelabel_thre,self.cluster_k1,self.cluster_k2,self.change_cnt,self.label_cache,None,gcn_n,gcn_s)
        inputs /= self.temp#<f1,f2>/temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon#overflow?
            return (masked_exps/masked_sums)

        #print(self.labels[indexes])
        #import pdb;pdb.set_trace()
        targets=self.labels[-1][indexes].clone()
        labels=self.labels[-1].clone()

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())#sim for each label
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)#mean-->center
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())

        del sim,nums
        return F.nll_loss(torch.log(masked_sim+1e-6), targets)
