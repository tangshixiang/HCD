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
    def forward(ctx, inputs, indexes, features,labels,source_classes,num_samples,momentum):
        ctx.features = features
        ctx.momentum = momentum
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

        #if ctx.domain:#与原来label的距离&max sim的对比

        return grad_inputs, None, None, None,None,None,None


def hm(inputs, indexes, features,labels,source_classes,num_samples,momentum=0.5):
    return HM.apply(inputs, indexes, features,labels,source_classes,num_samples,torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, source_classes,source_samples,temp=0.05, momentum=0.2,iterative=2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.source_classes=source_classes
        self.source_samples=source_samples

        self.momentum = momentum
        self.temp = temp
        self.labels=[]

        for i in range(iterative):
            self.labels.append(torch.zeros(num_samples).long().cuda())
            #self.register_buffer('labels_{}'.format(i), torch.zeros(num_samples).long())

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, indexes,domain=0):#domain=0:source domain=1:target
        # inputs: B*2048, features: L*2048
        inputs= hm(inputs, indexes, self.features, self.labels[-1],self.source_classes,self.num_samples,self.momentum)
        inputs /= self.temp#<f1,f2>/temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon#overflow?
            return (masked_exps/masked_sums)

        #print(self.labels[indexes])
        #import pdb;pdb.set_trace()
        targets = self.labels[-1][indexes].clone()
        labels = self.labels[-1].clone()

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())#sim for each label
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)#mean-->center
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim+1e-6), targets)
