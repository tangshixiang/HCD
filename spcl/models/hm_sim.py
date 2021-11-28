import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd
from spcl.utils.faiss_rerank import compute_jaccard_distance,compute_jaccard_distance_inital_rank


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, domain,update_label_thre_max, update_label_thre_min,labels,source_classes,num_samples,momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.domain=domain
        if domain:
            #with torch.no_grad():
            ctx.update_label_thre_max=update_label_thre_max
            ctx.update_label_thre_min=update_label_thre_min
            ctx.labels=labels
            ctx.source_classes=source_classes
            ctx.num_samples=num_samples
            #max_thre=ctx.update_label_thre_max*torch.exp(-0.01*(nums[ctx.max_label]).view(-1).contiguous())
            #max_thre=torch.where(max_thre<0.65,torch.zeros_like(max_thre)+0.65,max_thre)
            # labels[ctx.max_indexes]=torch.where(ctx.max_sim[ctx.max_idx]>=ctx.update_label_thre_max,ctx.max_label[ctx.max_idx],labels[ctx.max_indexes])
            # #print("after:",labels[indexes][max_idx])
            # labels[ctx.min_indexes]=torch.where(ctx.max_sim[ctx.min_idx]<=ctx.update_label_thre_min,ctx.outliers_label,labels[ctx.min_indexes])
            #print("after:",labels[indexes])
        ctx.save_for_backward(inputs, indexes)

        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        #if ctx.domain:#与原来label的距离&max sim的对比
        if 0:
            mode="single"#cluster_center,single

            #import pdb;pdb.set_trace()

            sim = torch.zeros((ctx.labels).max()+1, (ctx.features[indexes]).size(0)).float().cuda()
            feature_sim=(ctx.features[indexes]).mm(ctx.features.t())
            sim.index_add_(0, ctx.labels, feature_sim.t().contiguous())
            nums = torch.zeros(ctx.labels.max()+1, 1).float().cuda()
            nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
            mask = (nums>0).float()
            sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
            sim=torch.where(sim>0.999,torch.zeros_like(sim),sim)#move iso guy

            #import pdb;pdb.set_trace()
            # if mode=="cluster_center":
            #     change_num=64
            #     keep_num=64
            #     assert (keep_num<=change_num)
            #     #import pdb;pdb.set_trace()
            #
            #     max_sim,max_label=torch.max(sim[ctx.source_classes:],dim=0)#max_sim for each img[64]
            #     max_label+=ctx.source_classes
            #
            #     max_idx=torch.argsort(max_sim)[-change_num:]
            #     max_idx=max_idx[torch.randperm(change_num)[:keep_num]]
            #
            #     min_idx=torch.argsort(max_sim)[:change_num]
            #     min_idx=min_idx[torch.randperm(change_num)[:keep_num]]
            #     max_indexes=indexes[max_idx]
            #     min_indexes=indexes[min_idx]
            #
            #     # for i in range(len(indexes)):
            #     #    print("ori:",sim[labels[indexes[i]]][i]," new:",max_sim[i])
            #     # print("ori:",sim[labels[indexes]]," new:",max_sim[max_idx])
            #     #import pdb;pdb.set_trace()
            #     empty_label=torch.eq(nums[ctx.source_classes:],0)
            #     if torch.sum(empty_label)>keep_num:
            #         outliers_label=torch.argsort(empty_label.view(-1).contiguous())[-keep_num:]+ctx.source_classes
            #     else:
            #         outliers_label=torch.arange(ctx.labels.max()+1,ctx.labels.max()+1+keep_num).cuda()
            #
            #     ctx.max_indexes=max_indexes
            #     ctx.min_indexes=min_indexes
            #     ctx.max_sim=max_sim
            #     ctx.max_label=max_label
            #     ctx.max_idx=max_idx
            #     ctx.min_idx=min_idx
            #     ctx.outliers_label=outliers_label
            #
            #     #print(labels[indexes])
            #     #import pdb;pdb.set_trace()
            #     #modify-->torch.max(max_sim)
            #     ctx.update_label_thre_max=1.2 if torch.max(max_sim)>0.92 else 0.62
            #     if ctx.update_label_thre_max<0.9:
            #         print("nums:",nums[ctx.max_label].view(1,-1))
            #         print("max sim:",ctx.max_sim[ctx.max_idx])
            #     ctx.labels[ctx.max_indexes]=torch.where(ctx.max_sim[ctx.max_idx]>=ctx.update_label_thre_max,ctx.max_label[ctx.max_idx],ctx.labels[ctx.max_indexes])
            #     #print("after:",labels[indexes][max_idx])
            #     ctx.labels[ctx.min_indexes]=torch.where(ctx.max_sim[ctx.min_idx]<=ctx.update_label_thre_min,ctx.outliers_label,ctx.labels[ctx.min_indexes])
            if mode=="single":
                #reranking-->single
                k1=10
                k2=6
                reranking_thre_single=0.55
                changelabel_thre=0.25
                feature_sim[:,:ctx.source_classes]=0
                #feature_sim=torch.where(feature_sim>0.999,torch.zeros_like(feature_sim),feature_sim)
                reranking_feature=torch.sum(torch.gt(feature_sim,reranking_thre_single),dim=0)
                keep_feature=ctx.features[reranking_feature>0]
                cal_feature=keep_feature.clone()
                rerank_dist = compute_jaccard_distance(cal_feature, k1=k1, k2=k2)
                #change label
                rerank_dist[np.where(rerank_dist<0.000001)]=1#remove self
                close_guy=np.argmin(rerank_dist,axis=1)
                close_sim=rerank_dist.min(1)
                keep_arg=torch.arange(ctx.features.size(0))[reranking_feature>0].numpy()

                #keep_index=np.where(keep_arg in indexes.cpu().numpy().tolist())
                all_index=np.array([np.argwhere(keep_arg==indexes[i].item())[0] for i in range(len(indexes))]).reshape(-1)
                close_sim=torch.from_numpy(close_sim[all_index]).cuda()
                close_guy=torch.from_numpy(keep_arg[close_guy[all_index]]).cuda()
                all_index=torch.from_numpy(all_index).cuda()
                print("close_sim:",close_sim)
                #import pdb;pdb.set_trace()
                #max_labels=torch.where((nums.view(-1)[ctx.labels[indexes]]-nums.view(-1)[ctx.labels[close_guy]])>0,ctx.labels[indexes],ctx.labels[close_guy])
                ctx.labels[indexes]=torch.where(close_sim<changelabel_thre,ctx.labels[close_guy],ctx.labels[indexes])

                del cal_feature
                #merge_cluster
                # reranking_thre_cluster=0.55
                # sim[:ctx.source_classes]=0
                # reranking_cluster=torch.gt(sim,reranking_thre_cluster)
                # for i in len(indexes):
                #     cal_feature=sim[:,i][reranking_cluster[:,i]>0]
                #     k1=cal_feature.size(0)
                #     k2=1
                #     rerank_dist = compute_jaccard_distance(cal_feature, k1=k1, k2=k2)
                #     rerank_dist[np.where(rerank_dist<0.000001)]=1
                #     import pdb;pdb.set_trace()
                #merge cluster
                # reranking_thre=0.55
                # sim[:ctx.source_classes]=0
                # merge_cluster=torch.sum(torch.gt(sim,reranking_thre),dim=1)
                #
                sim_thre_high=0.7
                merge_thre=0.65
                sim_thre_low=0.5
                feature_sim=torch.where(feature_sim>0.999,torch.zeros_like(feature_sim),feature_sim)
                max_sim,max_label=torch.max(feature_sim,dim=1)
                # ctx.labels[indexes]=torch.where(close_guy==max_label,ctx.labels[max_label],ctx.labels[indexes])
                # import pdb;pdb.set_trace()
                #
                # print((nums[ctx.labels[max_label]]).view(-1))
                print(max_sim)
                # ctx.labels[indexes]=torch.where(max_sim>sim_thre_high,ctx.labels[max_label],ctx.labels[indexes])
                #
                # #merge cluster
                # #import pdb;pdb.set_trace()

                sim[:ctx.source_classes]=0
                #cluster_max_sim,cluster_max_label=torch.max(sim,dim=0)
                merge_cluster=torch.gt(sim,merge_thre)
                print("torch.max(sim):",torch.max(sim))
                _,merge_label=torch.max(sim,dim=0)
                if torch.sum(merge_cluster)>0:
                    for i in range(merge_cluster.size(1)):
                        if torch.sum(merge_cluster[:,i])>1:
                            print("------merge-------")
                            merge_label_i=torch.argsort(merge_cluster[:,i])[-torch.sum(merge_cluster[:,i]):]
                            for num in merge_label_i:
                                ctx.labels=torch.where(ctx.labels==num,torch.zeros_like(ctx.labels)+merge_label[i],ctx.labels)
                #
                #
                empty_label=torch.eq(nums[ctx.source_classes:],0)
                if torch.sum(empty_label)>len(indexes):
                    outliers_label=torch.argsort(empty_label.view(-1).contiguous())[-len(indexes):]+ctx.source_classes
                else:
                    outliers_label=torch.arange(ctx.labels.max()+1,ctx.labels.max()+1+len(indexes)).cuda()
                ctx.labels[indexes]=torch.where(max_sim<sim_thre_low,outliers_label,ctx.labels[indexes])


        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()
        return grad_inputs, None, None, None,None, None, None,None, None, None


def hm(inputs, indexes, features,domain,update_label_thre_max, update_label_thre_min,labels,source_classes,num_samples,momentum=0.5):
    return HM.apply(inputs, indexes, features,domain,update_label_thre_max,
    update_label_thre_min, labels,source_classes,num_samples,torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, source_classes,temp=0.05, momentum=0.2,update_label_thre_max=0.97,update_label_thre_min=0.55):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.source_classes=source_classes

        self.momentum = momentum
        self.temp = temp
        self.update_label_thre_max=update_label_thre_max
        self.update_label_thre_min=update_label_thre_min

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes,domain=0):#domain=0:source domain=1:target
        # inputs: B*2048, features: L*2048
        inputs= hm(inputs, indexes, self.features, domain,self.update_label_thre_max,
            self.update_label_thre_min,self.labels,self.source_classes,self.num_samples,self.momentum)
        inputs /= self.temp#<f1,f2>/temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon#overflow?
            return (masked_exps/masked_sums)

        #print(self.labels[indexes])
        #import pdb;pdb.set_trace()
        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())#sim for each label
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)#mean-->center
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim+1e-6), targets)
