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
    def forward(ctx, inputs, indexes, features, domain,update_label_thre_max, update_label_thre_min,labels,source_classes,num_samples,momentum,changelabel_thre,k1,k2,change_cnt,label_cache,confidence,gcn_n,gcn_s):
        ctx.features = features
        ctx.momentum = momentum
        ctx.domain=domain
        ctx.change_cnt=change_cnt
        ctx.source_classes=source_classes
        if domain:
            #with torch.no_grad():
            ctx.changelabel_thre=changelabel_thre
            ctx.confidence=confidence
            ctx.update_label_thre_max=update_label_thre_max
            ctx.update_label_thre_min=update_label_thre_min
            ctx.labels=labels
            ctx.k1=k1
            ctx.k2=k2
            ctx.gcn_n=gcn_n
            ctx.gcn_s=gcn_s
            ctx.num_samples=num_samples
            ctx.label_cache=label_cache
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
        ctx.change_cnt[indexes]+=1
        knn_gcn=1
        #if ctx.domain:
        if 0:
            mode="sub_cluster"#cluster_center,single,sim
            if mode=='hierarchy_gcn':
                pass
            if mode=='sub_cluster':
                sim_thre=0.3
                neighbor_thre=0.45
                cluster_merge_thre=0.55
                baseline=0
                #step1-->calculating the sub-cluster[label_cache]
                ori_labels=ctx.label_cache[indexes]
                max_label=(ctx.label_cache).max()+1
                ctx.label_cache[indexes]=1
                sim = torch.zeros(max_label, (ctx.features[indexes]).size(0)).float().cuda()
                feature_sim=(ctx.features[indexes]).mm(ctx.features.t())
                sim.index_add_(0, ctx.label_cache, feature_sim.t().contiguous())
                nums = torch.zeros(max_label, 1).float().cuda()
                nums.index_add_(0, ctx.label_cache, torch.ones(ctx.num_samples,1).float().cuda())
                mask = (nums>0).float()
                sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
                sim[:ctx.source_classes,:]=0

                new_conf=[]
                new_label=[]
                new_id=[]
                if baseline:
                    ori_0=compute_knn(ctx.features.clone(),k1=15)
                    neighbor=ori_0[indexes.cpu().numpy(),:]
                    nearest_neighbor=neighbor[:,1]
                    topk=10
                    with_top5_avg=0
                    for i in range(len(indexes)):
                        try:
                            all_neighbor=neighbor[i].tolist()
                            feature=ctx.features[all_neighbor].type(torch.float)
                            neigh=torch.from_numpy(ori_0[all_neighbor,:topk]).cuda()
                            tmp=torch.zeros(15,len(ctx.features)).float().cuda()
                            A=feature.mm(ctx.features.t())
                            tmp[torch.arange(15).view(-1,1).repeat(1,topk),neigh]=A[torch.arange(15).view(-1,1).repeat(1,topk),neigh]
                            adj=tmp.mm(torch.gt(tmp.t(),0).float())
                            if with_top5_avg:
                                #import pdb;pdb.set_trace()
                                adj=F.softmax(adj,dim=1)
                                tmp[0]=torch.mean(tmp[:5],dim=0)

                            max_conf,max_label=torch.max(adj[0][1:4],0)

                            new_label.append(ctx.label_cache[neigh[0][max_label.item()+1]].item())
                            new_conf.append(max_conf.item())
                            new_id.append(max_label.item()+1)
                        except:
                            import pdb;pdb.set_trace()
                else:
                    neighbor,pred,ori_neighbor=ctx.gcn_n(indexes,ctx.features,ctx.label_cache,1,output_feat=True)
                    #nearest_neighbor=ori_neighbor[:,1]
                    nearest_neighbor=[]

                    for i in range(len(indexes)):
                        index_dict={}
                        for idx,nei in enumerate(neighbor[i]):
                            index_dict[nei]=idx
                        top5_pred=[]
                        top5_index=[]
                        #debug
                        # tmp_pred=[]
                        # tmp_index=[]
                        # for idx,nei in enumerate(ori_neighbor[i,:5]):
                        #     tmp_pred.append(pred[i][index_dict[nei],1])
                        #     tmp_index.append(nei)
                        # import pdb;pdb.set_trace()

                        for idx,nei in enumerate(ori_neighbor[i,1:5]):
                            top5_pred.append(pred[i][index_dict[nei],1])
                            top5_index.append(nei)
                        max_conf,max_label=torch.max(torch.tensor(top5_pred),0)

                        la=ctx.label_cache[top5_index[max_label.item()]].item()
                        new_label.append(la)
                        new_conf.append(max_conf.item())
                        new_id.append(max_label.item()+1)
                        nearest_neighbor.append(top5_index[max_label.item()])
                #print('new conf:',new_conf)
                new_label=torch.tensor(new_label).cuda()
                nearest_neighbor=np.array(nearest_neighbor)

                sim[1]=1
                label_sim=sim[new_label,torch.arange(len(indexes))]
                new_label=torch.where(label_sim<=sim_thre,torch.zeros_like(new_label),new_label)
                new_label=torch.where(new_label<ctx.source_classes,torch.zeros_like(new_label),new_label)
                close_neighbor=torch.tensor([x[1] for idx,x in enumerate(neighbor)]).cuda()
                #close_neighbor=torch.tensor([x[new_id[idx]] for idx,x in enumerate(neighbor)]).cuda()
                new_label=torch.where(close_neighbor<=neighbor_thre,torch.zeros_like(new_label),new_label)

                nums[:ctx.source_classes]=1
                empty_label=torch.eq(nums,0).view(-1)
                if torch.sum(empty_label>0)<64:
                    outliers_label=torch.arange(ctx.label_cache.max()+1,ctx.label_cache.max()+1+len(indexes)).cuda()
                else:
                    outliers_label=(torch.arange(len(nums))[empty_label>0])[-len(indexes):].cuda()
                new_label=torch.where(new_label>0,new_label,outliers_label)
                ctx.label_cache[indexes]=new_label

                if torch.sum(torch.eq(new_label,1))>0:
                    #import pdb;pdb.set_trace()
                    cnt=0
                    loop=torch.sum(torch.eq(new_label,1)).item()
                    new_generate_label=torch.where(new_label>1,new_label,outliers_label)
                    ctx.label_cache[indexes]=new_generate_label
                    while(cnt<loop):
                        for i in range(len(indexes)):
                            if (new_label[i]==1):
                                    ctx.label_cache[indexes[i]]=ctx.label_cache[nearest_neighbor[i]]
                        cnt+=1
                    if torch.sum(torch.eq(ctx.label_cache[indexes],1))>0:
                        print('---------------------')
                        import pdb;pdb.set_trace()
                #sub-cluster merge(high confidence)
                #sub_cluster_cen
                sim[1]=0
                high_sim_thre=0.55
                use_cluster=3
                if use_cluster==0:
                    max_label=(ctx.label_cache).max()+1
                    nums = torch.zeros(max_label, 1).float().cuda()
                    nums.index_add_(0, ctx.label_cache, torch.ones(ctx.num_samples,1).float().cuda())
                    mask = (nums>0).float()
                    cluster_cen_sim = torch.zeros(max_label, (ctx.features).size(1)).float().cuda()
                    cluster_cen_sim.index_add_(0, ctx.label_cache, ctx.features.contiguous())
                    cluster_cen_sim /= (mask*nums+(1-mask)).clone().expand_as(cluster_cen_sim)
                    cluster_cen_sim /= torch.norm(cluster_cen_sim,dim=1).view(-1,1)
                    cen_sim=cluster_cen_sim[ctx.label_cache[indexes]].mm(cluster_cen_sim.t())
                    cen_sim=torch.gt(cen_sim,high_sim_thre)
                    cen_sim_num=torch.sum(cen_sim,dim=1)
                    if torch.sum(cen_sim_num)>len(indexes):
                        merge_map={}
                        for i in range(len(indexes)):
                            if cen_sim_num[i]>1:
                                #import pdb;pdb.set_trace()
                                merge_label_0=-1
                                merge_label=torch.arange(len(cen_sim[i]))[cen_sim[i]>0].tolist()
                                inter=set(merge_label) & set(merge_map.keys())
                                if len(inter)>0:
                                    inter_label=list(inter)
                                    merge_label_0=merge_map[inter_label[0]]
                                    if len(inter_label)>1:
                                        change_guys=[]
                                        for label in inter_label:
                                            change_guys.append(merge_map[label])
                                        for change_label,update_label in merge_map.items():
                                            if update_label in change_guys:
                                                merge_map[change_label]=merge_label_0
                                merge_label_0=merge_label[0] if merge_label_0==-1 else merge_label_0
                                for label in merge_label:
                                    merge_map[label]=merge_label_0
                        print("high conf map:",len(merge_map.keys()))
                        for change_label,update_label in merge_map.items():
                            ctx.label_cache[ctx.label_cache==int(change_label)]=int(update_label)
                        #change label for merge cluster
                        label_merge_map={}
                        for _,update_label in merge_map.items():
                            label_merge_map[update_label]=torch.unique(ctx.labels[ctx.label_cache==int(update_label)])
                        for update_label,label_list in label_merge_map.items():
                            if len(label_list)>1:
                                #import pdb;pdb.set_trace()
                                for la in label_list:
                                    ctx.labels[ctx.labels==la]=label_list[0]
                                print('fix bug')
                if use_cluster==1:#use index & sub_cluster_center
                    merge_sim=torch.gt(sim,high_sim_thre)
                    merge_sim_num=torch.sum(merge_sim,dim=0)

                    if torch.sum(merge_sim_num>1)>0:
                        merge_map={}
                        for i in range(len(indexes)):
                            if merge_sim_num[i]>1:
                                merge_label=torch.arange(merge_sim.size(0))[(merge_sim[:,i]>0)]
                                if ctx.label_cache[indexes[i]].item() not in merge_label:
                                    continue
                                merge_label_0=-1
                                inter=set(merge_label.tolist()) & set(merge_map.keys())
                                if len(inter)>0:
                                    inter_label=list(inter)
                                    merge_label_0=merge_map[inter_label[0]]
                                    if len(inter_label)>1:
                                        change_guys=[]
                                        for label in inter_label:
                                            change_guys.append(merge_map[label])
                                        for change_label,update_label in merge_map.items():
                                            if update_label in change_guys:
                                                merge_map[change_label]=merge_label_0
                                merge_label_0=merge_label[0].item() if merge_label_0==-1 else merge_label_0
                                for label in merge_label:
                                    merge_map[label.item()]=merge_label_0

                        print("high conf map:",len(merge_map.keys()))
                        for change_label,update_label in merge_map.items():
                            ctx.label_cache[ctx.label_cache==int(change_label)]=int(update_label)
                if use_cluster==2: #index& neighbor
                    sim[1]=0
                    high_sim_thre=0.55
                    merge_sim=torch.gt(feature_sim,high_sim_thre)
                    merge_sim_num=torch.sum(merge_sim,dim=0)
                    all_index=torch.arange(len(ctx.features))
                    merge_guys=[]
                    if torch.sum(merge_sim_num>1)>0:
                        for i in range(len(indexes)):
                            if merge_sim_num[i]>1:
                                merge_label=all_index[merge_sim_num[i]>0]
                                merge_label0=ctx.label_cache[merge_label[0]]
                                for lab in merge_label:
                                    ctx.label_cache[ctx.label_cache==ctx.label_cache[lab]]=merge_label0
                                merge_guys.append(len(merge_label))
                            else:
                                merge_guys.append(0)
                        print('sub cluster:',merge_guys)
                if use_cluster==3: #average linkage
                    high_sim_thre=0.5
                    avg_link=0.45
                    merge_sim=torch.gt(sim,high_sim_thre)
                    merge_sim_num=torch.sum(merge_sim,dim=0)

                    if torch.sum(merge_sim_num>1)>0:
                        merge_map={}
                        all_avg=[]
                        for i in range(len(indexes)):
                            if merge_sim_num[i]>1:
                                ori_merge_label=torch.arange(merge_sim.size(0))[(merge_sim[:,i]>0)].tolist()
                                if ctx.label_cache[indexes[i]].item() not in ori_merge_label:
                                    continue
                                #import pdb;pdb.set_trace()
                                #avg linkage
                                ori_sub_feat=ctx.features[ctx.label_cache==ctx.label_cache[indexes[i]]]
                                merge_label=[]
                                for lab in ori_merge_label:
                                    merge_sub_feat=ctx.features[ctx.label_cache==lab]
                                    feat_sim=ori_sub_feat.mm(merge_sub_feat.t())
                                    all_avg.append(torch.mean(feat_sim).item())
                                    if torch.mean(feat_sim)>=avg_link:
                                        merge_label.append(lab)
                                if ctx.label_cache[indexes[i]].item() not in merge_label:
                                    continue
                                merge_label_0=-1
                                inter=set(merge_label) & set(merge_map.keys())
                                if len(inter)>0:
                                    inter_label=list(inter)
                                    merge_label_0=merge_map[inter_label[0]]
                                    if len(inter_label)>1:
                                        change_guys=[]
                                        for label in inter_label:
                                            change_guys.append(merge_map[label])
                                        for change_label,update_label in merge_map.items():
                                            if update_label in change_guys:
                                                merge_map[change_label]=merge_label_0
                                merge_label_0=merge_label[0] if merge_label_0==-1 else merge_label_0
                                for label in merge_label:
                                    merge_map[label]=merge_label_0
                        for change_label,update_label in merge_map.items():
                            ctx.label_cache[ctx.label_cache==int(change_label)]=int(update_label)
                        print('all_avg:',all_avg)
                #step2-->cluster merge&split
                method=2
                if method==0:
                    ori_labels=ctx.labels[indexes]
                    max_label=(ctx.label_cache).max()+1
                    sim = torch.zeros(max_label, (ctx.features[indexes]).size(0)).float().cuda()
                    feature_sim=(ctx.features[indexes]).mm(ctx.features.t())
                    sim.index_add_(0, ctx.label_cache, feature_sim.t().contiguous())
                    nums = torch.zeros(max_label, 1).float().cuda()
                    nums.index_add_(0, ctx.label_cache, torch.ones(ctx.num_samples,1).float().cuda())
                    mask = (nums>0).float()
                    sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
                    sim[:ctx.source_classes,:]=0

                    cluster_cen_sim = torch.zeros(max_label, (ctx.features).size(1)).float().cuda()
                    cluster_cen_sim.index_add_(0, ctx.label_cache, ctx.features.contiguous())
                    cluster_cen_sim /= (mask*nums+(1-mask)).clone().expand_as(cluster_cen_sim)
                    cluster_cen_sim /= torch.norm(cluster_cen_sim,dim=1).view(-1,1)

                    sub_cluster_neighbor=torch.gt(sim,cluster_merge_thre)
                    sub_cluster_sum=torch.sum(sub_cluster_neighbor,dim=0)
                    topk=5
                    adj_thre=1.6
                    adj_sim_thre=0.5
                    for i in range(len(indexes)):
                        if sub_cluster_sum[i]>1:
                            #find label
                            sub_id=torch.arange(max_label)[sub_cluster_neighbor[:,i]>0]
                            label=[]
                            for x in sub_id:
                                label.extend(ctx.labels[ctx.label_cache==x.item()].tolist())
                            label=np.unique(np.array(label))
                            #find all sub-cluster in this label
                            all_sub_id=[]
                            for x in label:
                                all_sub_id.extend(ctx.label_cache[ctx.labels==x].tolist())
                            all_sub_id=np.unique(np.array(all_sub_id)).tolist()
                            all_sub_feature=cluster_cen_sim[all_sub_id]

                            top=topk#min(len(all_sub_feature),topk)
                            ori_0=compute_knn(cluster_cen_sim.clone(),k1=top)
                            neigh=torch.from_numpy(ori_0[all_sub_id,:top]).cuda()
                            tmp=torch.zeros(len(neigh),len(cluster_cen_sim)).float().cuda()
                            A=all_sub_feature.mm(cluster_cen_sim.t())
                            tmp[torch.arange(len(all_sub_feature)).view(-1,1).repeat(1,top),neigh]=A[torch.arange(len(all_sub_feature)).view(-1,1).repeat(1,top),neigh]
                            adj=tmp.mm(torch.gt(tmp.t(),0).float())
                            #print('adj:',adj)

                            #generate new label
                            A_adj=all_sub_feature.mm(all_sub_feature.t())
                            adj=(adj>adj_thre) & (A_adj>adj_sim_thre)
                            nums = torch.zeros((ctx.labels).max()+1, 1).float().cuda()
                            nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                            empty_label=torch.eq(nums,0).view(-1)
                            if torch.sum(empty_label>0)<len(all_sub_feature):
                                new_label=torch.arange(ctx.labels.max()+1,ctx.labels.max()+1+len(all_sub_feature)).tolist()
                            else:
                                new_label=(torch.arange(len(nums))[empty_label>0])[-len(all_sub_feature):].tolist()
                            merge_map={}
                            all_sub_id=torch.tensor(all_sub_id)
                            for id in range(len(all_sub_feature)):
                                merge_label_0=-1
                                merge_label=all_sub_id[adj[id]>0].tolist()
                                inter=set(merge_label) & set(merge_map.keys())
                                if len(inter)>0:
                                    inter_label=list(inter)
                                    merge_label_0=merge_map[inter_label[0]]
                                    if len(inter_label)>1:
                                        change_guys=[]
                                        for label in inter_label:
                                            change_guys.append(merge_map[label])
                                        for change_label,update_label in merge_map.items():
                                            if update_label in change_guys:
                                                merge_map[change_label]=merge_label_0
                                merge_label_0=new_label[id] if merge_label_0==-1 else merge_label_0
                                for label in merge_label:
                                    merge_map[label]=merge_label_0

                            print("len(merge_map):",len(merge_map.keys()))
                            for change_label,update_label in merge_map.items():
                                ctx.labels[ctx.label_cache==int(change_label)]=int(update_label)

                if method==1: #only reconsider the relationship with indexes
                    adj_thre=1.6
                    sim_thre=0.5
                    max_label=(ctx.label_cache).max()+1
                    nums = torch.zeros(max_label, 1).float().cuda()
                    nums.index_add_(0, ctx.label_cache, torch.ones(ctx.num_samples,1).float().cuda())
                    mask = (nums>0).float()

                    cluster_cen_sim = torch.zeros(max_label, (ctx.features).size(1)).float().cuda()
                    cluster_cen_sim.index_add_(0, ctx.label_cache, ctx.features.contiguous())
                    cluster_cen_sim /= (mask*nums+(1-mask)).clone().expand_as(cluster_cen_sim)
                    cluster_cen_sim /= torch.norm(cluster_cen_sim,dim=1).view(-1,1)

                    sim = torch.zeros(max_label, (ctx.features[indexes]).size(0)).float().cuda()
                    feature_sim=(cluster_cen_sim[ctx.label_cache[indexes]]).mm(ctx.features.t())
                    sim.index_add_(0, ctx.label_cache, feature_sim.t().contiguous())
                    sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
                    sim[:ctx.source_classes,:]=0

                    consider_clu=torch.gt(sim,sim_thre)
                    merge_sim_num=torch.sum(consider_clu,dim=0)
                    top=10
                    #ori_0=compute_knn(cluster_cen_sim.clone(),k1=top)

                    nums = torch.zeros((ctx.labels).max()+1, 1).float().cuda()
                    nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                    empty_label=torch.eq(nums,0).view(-1)
                    if (torch.sum(empty_label>0)<len(indexes)):
                        new_label=torch.arange(ctx.labels.max()+1,ctx.labels.max()+1+len(indexes)).tolist()
                    else:
                        new_label=(torch.arange(len(nums))[empty_label>0])[-len(indexes):].tolist()

                    merge_guys=[]
                    for i in range(len(indexes)):
                        ctx.labels[ctx.label_cache==ctx.label_cache[indexes[i]]]=new_label[i]
                        if merge_sim_num[i]>1:
                            #import pdb;pdb.set_trace()
                            try:
                                nei=torch.arange(len(consider_clu[:,i]))[consider_clu[:,i]>0] #neighbor sub center
                                all_label=[]
                                for neighbor in nei:
                                    all_label.extend(ctx.labels[ctx.label_cache==neighbor.item()].tolist())
                                all_label=torch.unique(torch.tensor(all_label)) #merge labels
                                if len(all_label)>1:
                                    merge_label0=all_label[0]
                                    for lab in all_label:
                                        ctx.labels[ctx.labels==lab.item()]=merge_label0
                            except:
                                print('oh no!')
                                import pdb;pdb.set_trace()

                            # nei=ori_0[torch.arange(len(consider_clu[i]))[condider_clu[i]>0].tolist(),:]
                            # neigh=torch.from_numpy(nei).cuda()
                            # tmp=torch.zeros(len(nei),len(cluster_cen_sim)).float().cuda()
                            # all_sub_feature=cluster_cen_sim[condider_clu[i]>0]
                            # A=all_sub_feature.mm(cluster_cen_sim.t())
                            # tmp[torch.arange(len(all_sub_feature)).view(-1,1).repeat(1,top),neigh]=A[torch.arange(len(all_sub_feature)).view(-1,1).repeat(1,top),neigh]
                            # adj_id=torch.arange(len(nei))[nei==ctx.label_cache[indexes[i]]]
                            # adj=tmp[adj_id].mm(torch.gt(tmp.t(),0).float())
                            # adj=(adj>adj_thre)
                            # keep_cluster=set(torch.arange(len(consider_clu[i]))[condider_clu[i]>0][adj>0].tolist())
                            #
                            # ctx.labels[ctx.label_cache==ctx.label_cache[indexes[i]]]=new_label[i]
                            # all_label=torch.unique(ctx.labels[torch.arange(len(consider_clu[i]))[condider_clu[i]>0]])
                            #
                            # for lab in all_label:
                            #     inter=keep_cluster & set(ctx.label_cache[ctx.labels==lab].tolist())
                            #     if len(inter)>0:
                            #         #merge
                            #         ctx.label[ctx.label==lab]=new_label[adj_id]
                            merge_guys.append(len(all_label))
                        else:
                            #split
                            merge_guys.append(0)
                    print('merge:',merge_guys)

                if method==2: #sub cluster center & cluster center
                    merge_thre=1.0
                    if merge_thre>=1:
                        print('update label')
                        ctx.labels[ctx.source_classes:]=ctx.label_cache[ctx.source_classes:]
                    else:
                        #sub center-->new label
                        nums = torch.zeros((ctx.labels).max()+1, 1).float().cuda()
                        nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                        empty_label=torch.eq(nums,0).view(-1)
                        if (torch.sum(empty_label>0)<len(indexes)):
                            new_label=torch.arange(ctx.labels.max()+1,ctx.labels.max()+1+len(indexes)).tolist()
                        else:
                            new_label=(torch.arange(len(nums))[empty_label>0])[-len(indexes):].tolist()
                        ori_labels=ctx.labels[indexes]
                        for i in range(len(indexes)):
                            ctx.labels[ctx.label_cache==ctx.label_cache[indexes[i]]]=new_label[i]

                        #cal cluster center
                        nums = torch.zeros((ctx.label_cache).max()+1, 1).float().cuda()
                        nums.index_add_(0, ctx.label_cache, torch.ones(ctx.num_samples,1).float().cuda())
                        mask = (nums>0).float()
                        cluster_cen_sim = torch.zeros((ctx.label_cache).max()+1, (ctx.features).size(1)).float().cuda()
                        cluster_cen_sim.index_add_(0, ctx.label_cache, ctx.features.contiguous())
                        cluster_cen_sim /= (mask*nums+(1-mask)).clone().expand_as(cluster_cen_sim)
                        cluster_cen_sim /= torch.norm(cluster_cen_sim,dim=1).view(-1,1)

                        #sim sub cluster center& cluster center
                        sim = torch.zeros((ctx.labels).max()+1, (ctx.features[indexes]).size(0)).float().cuda()
                        feature_sim=(cluster_cen_sim[ctx.label_cache[indexes]]).mm(ctx.features.t())
                        sim.index_add_(0, ctx.labels, feature_sim.t().contiguous())
                        nums = torch.zeros((ctx.labels).max()+1, 1).float().cuda()
                        nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                        mask = (nums>0).float()
                        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
                        sim[:ctx.source_classes,:]=0

                        merge_num=torch.gt(sim,merge_thre)
                        merge_num_sum=torch.sum(merge_num,dim=0)
                        all_label=torch.arange((ctx.labels).max()+1)
                        if torch.sum(merge_num_sum>1)>0:
                            merge_map={}
                            for i in range(len(indexes)):
                                if merge_num_sum[i]>1:
                                    merge_label=all_label[merge_num[:,i]>0]
                                    if ctx.labels[indexes[i]].item() not in merge_label:
                                        continue
                                    merge_label_0=-1
                                    inter=set(merge_label.tolist()) & set(merge_map.keys())
                                    if len(inter)>0:
                                        inter_label=list(inter)
                                        merge_label_0=merge_map[inter_label[0]]
                                        if len(inter_label)>1:
                                            change_guys=[]
                                            for label in inter_label:
                                                change_guys.append(merge_map[label])
                                            for change_label,update_label in merge_map.items():
                                                if update_label in change_guys:
                                                    merge_map[change_label]=merge_label_0
                                    merge_label_0=merge_label[0].item() if merge_label_0==-1 else merge_label_0
                                    for label in merge_label:
                                        merge_map[label.item()]=merge_label_0
                            for change_label,update_label in merge_map.items():
                                ctx.labels[ctx.labels==int(change_label)]=int(update_label)
                            print('merge:',len(merge_map.keys()))

                if method==3:
                    merge_thre=0.5
                    avg_link=0.45
                    #sub center-->new label
                    nums = torch.zeros((ctx.labels).max()+1, 1).float().cuda()
                    nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                    empty_label=torch.eq(nums,0).view(-1)
                    if (torch.sum(empty_label>0)<len(indexes)):
                        new_label=torch.arange(ctx.labels.max()+1,ctx.labels.max()+1+len(indexes)).tolist()
                    else:
                        new_label=(torch.arange(len(nums))[empty_label>0])[-len(indexes):].tolist()
                    ori_labels=ctx.labels[indexes]
                    for i in range(len(indexes)):
                        ctx.labels[ctx.label_cache==ctx.label_cache[indexes[i]]]=new_label[i]

                    #index & cluster center
                    sim = torch.zeros((ctx.labels).max()+1, (ctx.features[indexes]).size(0)).float().cuda()
                    feature_sim=(ctx.features[indexes]).mm(ctx.features.t())
                    sim.index_add_(0, ctx.labels, feature_sim.t().contiguous())
                    nums = torch.zeros((ctx.labels).max()+1, 1).float().cuda()
                    nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                    mask = (nums>0).float()
                    sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
                    sim[:ctx.source_classes,:]=0
                    #except new guys
                    try:
                        sim[new_label]=0
                        sim[new_label,torch.arange(len(indexes))]=1
                    except:
                        import pdb;pdb.set_trace()

                    merge_num=torch.gt(sim,merge_thre)
                    merge_num_sum=torch.sum(merge_num,dim=0)
                    all_label=torch.arange((ctx.labels).max()+1)
                    if torch.sum(merge_num_sum>1)>0:
                        merge_map={}
                        label_avg=[]
                        for i in range(len(indexes)):
                            if merge_num_sum[i]>1:
                                ori_merge_label=all_label[merge_num[:,i]>0].tolist()
                                if ctx.labels[indexes[i]].item() not in ori_merge_label:
                                    continue
                                ori_cluster_feat=ctx.features[ctx.labels==ctx.labels[indexes[i]]]
                                merge_label=[]
                                for lab in ori_merge_label:
                                    merge_feat=ctx.features[ctx.labels==lab]
                                    link=torch.mean(ori_cluster_feat.mm(merge_feat.t()))
                                    label_avg.append(link.item())
                                    if link>=avg_link:
                                        merge_label.append(lab)
                                if ctx.labels[indexes[i]].item() not in merge_label:
                                    continue
                                merge_label_0=-1
                                inter=set(merge_label) & set(merge_map.keys())
                                if len(inter)>0:
                                    inter_label=list(inter)
                                    merge_label_0=merge_map[inter_label[0]]
                                    if len(inter_label)>1:
                                        change_guys=[]
                                        for label in inter_label:
                                            change_guys.append(merge_map[label])
                                        for change_label,update_label in merge_map.items():
                                            if update_label in change_guys:
                                                merge_map[change_label]=merge_label_0
                                merge_label_0=merge_label[0] if merge_label_0==-1 else merge_label_0
                                for label in merge_label:
                                    merge_map[label]=merge_label_0

                        for change_label,update_label in merge_map.items():
                            ctx.labels[ctx.labels==int(change_label)]=int(update_label)
                        print('label_avg:',label_avg)
                        print('merge:',len(merge_map.keys()))


            if mode=="gcn_only":
                baseline=0
                gcnv=0
                only_cal_first_guy=0
                sim_thre=0.3
                neighbor_thre=0.45
                cluster_merge_thre=0.55
                neighbor_num=30

                ori_labels=ctx.labels[indexes]
                max_label=(ctx.labels).max()+1
                ctx.labels[indexes]=1
                sim = torch.zeros(max_label, (ctx.features[indexes]).size(0)).float().cuda()
                feature_sim=(ctx.features[indexes]).mm(ctx.features.t())
                sim.index_add_(0, ctx.labels, feature_sim.t().contiguous())
                nums = torch.zeros(max_label, 1).float().cuda()
                nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                mask = (nums>0).float()
                sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
                sim[:ctx.source_classes,:]=0

                #1109-cluster_cen_sim
                cluster_cen_sim = torch.zeros(max_label, (ctx.features).size(1)).float().cuda()
                cluster_cen_sim.index_add_(0, ctx.labels, ctx.features.contiguous())
                cluster_cen_sim /= (mask*nums+(1-mask)).clone().expand_as(cluster_cen_sim)
                cluster_cen_sim /= torch.norm(cluster_cen_sim,dim=1).view(-1,1)


                new_label=[]
                if baseline:
                    ori_0=compute_knn(ctx.features.clone(),k1=15)
                    neighbor=ori_0[indexes.cpu().numpy(),:]
                    nearest_neighbor=neighbor[:,1]
                    topk=10
                    with_top5_avg=1
                    for i in range(len(indexes)):
                        try:
                            all_neighbor=neighbor[i].tolist()
                            feature=ctx.features[all_neighbor].type(torch.float)
                            neigh=torch.from_numpy(ori_0[all_neighbor,:topk]).cuda()
                            tmp=torch.zeros(15,len(ctx.features)).float().cuda()
                            A=feature.mm(ctx.features.t())
                            tmp[torch.arange(15).view(-1,1).repeat(1,topk),neigh]=A[torch.arange(15).view(-1,1).repeat(1,topk),neigh]
                            adj=tmp.mm(torch.gt(tmp.t(),0).float())
                            if with_top5_avg:
                                #import pdb;pdb.set_trace()
                                try:
                                    adj=F.softmax(adj,dim=1)
                                    tmp[0]=torch.mean(tmp[:5],dim=0)
                                except:
                                    import pdb;pdb.set_trace()

                            max_conf,max_label=torch.max(adj[0][1:5],0)


                            new_label.append(ctx.labels[neigh[0][max_label.item()+1]].item())
                        except:
                            import pdb;pdb.set_trace()
                        #tmp[torch.arange(800).view(-1,1).repeat(1,topk),neigh]=A[torch.arange(self.neighbor_num).view(-1,1).repeat(1,topk),neigh]
                        #adj=tmp.mm(torch.gt(tmp.t(),0).float())
                    # neighbor=ori_0[indexes.cpu().numpy(),:]
                    # for i in range(len(indexes)):
                    #     new_label.append(ctx.labels[neighbor[i][1]])
                elif gcnv:
                    ctx.labels[indexes]=ori_labels
                    new_conf=[]
                    neighbor,pred=ctx.gcn_n(indexes,ctx.features,ctx.labels,1,output_feat=True)
                    for i in range(len(indexes)):
                        if pred[i][0]>pred[i][1]:
                            new_label.append(ctx.labels[neighbor[i][0]].item())
                            new_conf.append(pred[i][0].item())
                        else:
                            new_label.append(ctx.labels[neighbor[i][1]].item())
                            new_conf.append(pred[i][1].item())
                    print('new conf:',new_conf)
                else:
                    with_one_variance=0
                    neighbor,pred,ori_neighbor=ctx.gcn_n(indexes,ctx.features,ctx.labels,1,output_feat=True)
                    new_conf=[]
                    nearest_neighbor=ori_neighbor[:,1]
                    if with_one_variance:
                        #import pdb;pdb.set_trace()
                        try:
                            all_pred=[]
                            for x in pred:
                                all_pred.extend(x[:,1].tolist())
                            all_pred=torch.tensor(all_pred)
                            var=torch.var(all_pred)
                            outlier_thre=min((torch.mean(all_pred)+2*var).item(),0.9)
                            print('outlier_thre:',outlier_thre)
                        except:
                            import pdb;pdb.set_trace()
                    for i in range(len(indexes)):
                        index_dict={}
                        for idx,nei in enumerate(neighbor[i]):
                            index_dict[nei]=idx
                        top5_pred=[]
                        top5_index=[]
                        try:
                            for idx,nei in enumerate(ori_neighbor[i,1:5]):
                                top5_pred.append(pred[i][index_dict[nei],1])
                                top5_index.append(nei)
                            max_conf,max_label=torch.max(torch.tensor(top5_pred),0)
                            #import pdb;pdb.set_trace()
                            #if max_conf>0.9:
                            if not only_cal_first_guy:
                                la=ctx.labels[top5_index[max_label.item()]].item()
                                #new_label.append(ctx.labels[neighbor[i][max_label.item()+1]].item())
                            else:
                                if max_label==0:
                                    la=ctx.labels[top5_index[max_label.item()]].item()
                                    #new_label.append(ctx.labels[neighbor[i][max_label.item()+1]].item())
                                else:
                                    la=0
                            if with_one_variance and max_conf.item()<outlier_thre:
                                la=0

                            new_label.append(la)
                        except:
                            print('hm error')
                            import pdb;pdb.set_trace()
                        #else:
                        #    new_label.append(0)
                        new_conf.append(max_conf.item())
                    #print('new conf:',new_conf)
                new_label=torch.tensor(new_label).cuda()
                #import pdb;pdb.set_trace()
                #new_label=ctx.gcn_s(indexes,ctx.features,ctx.labels,1,neighbor,pred,output_feat=True)
                #new_label=torch.tensor(new_label).cuda()
                #cal sim
                #sim[ori_labels,torch.arange(len(indexes)).cuda()]=torch.where(sim[ori_labels,torch.arange(len(indexes)).cuda()]>0,sim[ori_labels,torch.arange(len(indexes)).cuda()],torch.ones_like(sim[ori_labels,torch.arange(len(indexes)).cuda()]))
                sim[1]=1
                label_sim=sim[new_label,torch.arange(len(indexes))]
                new_label=torch.where(label_sim<=sim_thre,torch.zeros_like(new_label),new_label)
                new_label=torch.where(new_label<ctx.source_classes,torch.zeros_like(new_label),new_label)
                try:
                    close_neighbor=torch.tensor([x[1] for _,x in enumerate(neighbor)]).cuda()
                    new_label=torch.where(close_neighbor<=neighbor_thre,torch.zeros_like(new_label),new_label)
                except:
                    import pdb;pdb.set_trace()

                nums[:ctx.source_classes]=1
                #nums[ori_labels]=1
                empty_label=torch.eq(nums,0).view(-1)
                #outliers_label=(torch.arange(len(nums))[empty_label>0])
                if torch.sum(empty_label>0)<64:
                    outliers_label=torch.arange(ctx.labels.max()+1,ctx.labels.max()+1+len(indexes)).cuda()
                else:
                    outliers_label=(torch.arange(len(nums))[empty_label>0])[-len(indexes):].cuda()
                new_label=torch.where(new_label>0,new_label,outliers_label)
                ctx.labels[indexes]=new_label

                if torch.sum(torch.eq(new_label,1))>0:
                    #import pdb;pdb.set_trace()
                    cnt=0
                    loop=torch.sum(torch.eq(new_label,1)).item()
                    new_generate_label=torch.where(new_label>1,new_label,outliers_label)
                    ctx.labels[indexes]=new_generate_label
                    while(cnt<loop):
                        for i in range(len(indexes)):
                            if (new_label[i]==1):
                                    ctx.labels[indexes[i]]=ctx.labels[nearest_neighbor[i]]
                        cnt+=1
                    if torch.sum(torch.eq(ctx.labels[indexes],1))>0:
                        print('---------------------')
                        import pdb;pdb.set_trace()

                #merge cluster
                merge_method=0
                sim[1]=0
                #sim[ori_labels,torch.arange(len(indexes)).cuda()]=torch.where(sim[ori_labels,torch.arange(len(indexes)).cuda()]<0.9999,sim[ori_labels,torch.arange(len(indexes)).cuda()],torch.zeros_like(sim[ori_labels,torch.arange(len(indexes)).cuda()]))
                if merge_method==0:
                    #fix bug-1108,recompute
                    # sim = torch.zeros((ctx.labels).max()+1, (ctx.features[indexes]).size(0)).float().cuda()
                    # feature_sim=(ctx.features[indexes]).mm(ctx.features.t())
                    # sim.index_add_(0, ctx.labels, feature_sim.t().contiguous())
                    # nums = torch.zeros((ctx.labels).max()+1,1).float().cuda()
                    # nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                    # mask = (nums>0).float()
                    # sim/=(mask*nums+(1-mask)).clone().expand_as(sim)
                    # sim[:ctx.source_classes,:]=0

                    merge_sim=torch.gt(sim,cluster_merge_thre)
                    merge_sim_num=torch.sum(merge_sim,dim=0)

                    without_outlier=1


                    if torch.sum(merge_sim_num>1)>0:
                        merge_map={}
                        for i in range(len(indexes)):
                            if merge_sim_num[i]>1:
                                #import pdb;pdb.set_trace()
                                if not without_outlier:
                                    merge_label=torch.arange(merge_sim.size(0))[merge_sim[:,i]>0]
                                else:
                                    merge_label=torch.arange(merge_sim.size(0))[(merge_sim[:,i]>0) & (nums[:,0]>1)]
                                if ctx.labels[indexes[i]].item() not in merge_label:
                                    continue
                                #merge_label_0=merge_label[0].item()
                                merge_label_0=-1
                                inter=set(merge_label.tolist()) & set(merge_map.keys())
                                if len(inter)>0:
                                    inter_label=list(inter)
                                    merge_label_0=merge_map[inter_label[0]]
                                    if len(inter_label)>1:
                                        change_guys=[]
                                        for label in inter_label:
                                            change_guys.append(merge_map[label])
                                        for change_label,update_label in merge_map.items():
                                            if update_label in change_guys:
                                                merge_map[change_label]=merge_label_0
                                merge_label_0=merge_label[0].item() if merge_label_0==-1 else merge_label_0
                                for label in merge_label:
                                    merge_map[label.item()]=merge_label_0
                        print("len(merge_map):",len(merge_map.keys()))
                        for change_label,update_label in merge_map.items():
                            ctx.labels[ctx.labels==int(change_label)]=int(update_label)
                #new merge method
                if merge_method==1:
                    #fix bug-1108-->include current batch(not good)
                    # sim = torch.zeros((ctx.labels).max()+1, (ctx.features[indexes]).size(0)).float().cuda()
                    # feature_sim=(ctx.features[indexes]).mm(ctx.features.t())
                    # sim.index_add_(0, ctx.labels, feature_sim.t().contiguous())

                    without_outlier=0
                    #cen_nums=nums
                    # cluster_cen_sim = torch.zeros((ctx.labels).max()+1, (ctx.features).size(1)).float().cuda()
                    # cluster_cen_sim.index_add_(0, ctx.labels, ctx.features.contiguous())
                    # cen_nums = torch.zeros((ctx.labels).max()+1,1).float().cuda()
                    # cen_nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                    # mask = (cen_nums>0).float()
                    # cluster_cen_sim /= (mask*cen_nums+(1-mask)).clone().expand_as(cluster_cen_sim)
                    # cluster_cen_sim /= torch.norm(cluster_cen_sim,dim=1).view(-1,1)
                    # sim/=(mask*nums+(1-mask)).clone().expand_as(sim)
                    # sim[:ctx.source_classes,:]=0


                    cen_sim_thre=0.5
                    merge_sim=torch.gt(sim,cluster_merge_thre)
                    merge_sim_num=torch.sum(merge_sim,dim=0)
                    if torch.sum(merge_sim_num>1)>0:
                        merge_map={}
                        for i in range(len(indexes)):
                            if merge_sim_num[i]>1:
                                #import pdb;pdb.set_trace()
                                if not without_outlier:
                                    merge_label=torch.arange(merge_sim.size(0))[merge_sim[:,i]>0]
                                else:
                                    merge_label=torch.arange(merge_sim.size(0))[(merge_sim[:,i]>0) & (cen_nums[:,0]>1)]
                                if ctx.labels[indexes[i]].item() not in merge_label:
                                    continue
                                #import pdb;pdb.set_trace()
                                merge_center_feature=cluster_cen_sim[merge_label]
                                cen_sim=merge_center_feature.mm(merge_center_feature.t())
                                print('cen_sim:',cen_sim)
                                cen_sim=(cen_sim>cen_sim_thre)
                                cen_sim=torch.gt(cen_sim,cen_sim_thre)
                                cen_sum=torch.sum(cen_sim,dim=1)
                                for id in range(len(merge_label)):
                                    if cen_sum[id]>1:
                                        all_label=torch.arange(len(merge_label))[cen_sim[id]==1]
                                        inter=set(all_label.tolist()) & set(merge_map.keys())
                                        merge_label_0=-1
                                        if len(inter)>0:
                                            inter_label=list(inter)
                                            merge_label_0=merge_map[inter_label[0]]
                                            if len(inter_label)>1:
                                                change_guys=[]
                                                for label in inter_label:
                                                    change_guys.append(merge_map[label])
                                                for change_label,update_label in merge_map.items():
                                                    if update_label in change_guys:
                                                        merge_map[change_label]=merge_label_0
                                        merge_label_0=all_label[0].item() if merge_label_0==-1 else merge_label_0
                                        for label in all_label:
                                            merge_map[label.item()]=merge_label_0

                        print("len(merge_map):",len(merge_map.keys()))
                        for change_label,update_label in merge_map.items():
                            ctx.labels[ctx.labels==int(change_label)]=int(update_label)
                if merge_method==2: #cluster merge by neighbor
                    #opt1-->baseline
                    # sim = torch.zeros((ctx.labels).max()+1, (ctx.features[indexes]).size(0)).float().cuda()
                    # feature_sim=(ctx.features[indexes]).mm(ctx.features.t())
                    # sim.index_add_(0, ctx.labels, feature_sim.t().contiguous())
                    #
                    # without_outlier=0
                    # cluster_cen_sim = torch.zeros((ctx.labels).max()+1, (ctx.features).size(1)).float().cuda()
                    # cluster_cen_sim.index_add_(0, ctx.labels, ctx.features.contiguous())
                    # cen_nums = torch.zeros((ctx.labels).max()+1,1).float().cuda()
                    # cen_nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                    # mask = (cen_nums>0).float()
                    # cluster_cen_sim /= (mask*cen_nums+(1-mask)).clone().expand_as(cluster_cen_sim)
                    # cluster_cen_sim /= torch.norm(cluster_cen_sim,dim=1).view(-1,1)
                    # sim/=(mask*cen_nums+(1-mask)).clone().expand_as(sim)
                    # sim[:ctx.source_classes,:]=0

                    cluster_merge_thre_first=0.55
                    cluster_merge_thre_second=0.5
                    ori_0=compute_knn(cluster_cen_sim.clone(),k1=10)
                    merge_sim=torch.gt(sim,cluster_merge_thre_first)
                    merge_sim_num=torch.sum(merge_sim,dim=0)
                    topk=5
                    if torch.sum(merge_sim_num>1)>0:
                        merge_map={}
                        for i in range(len(indexes)):
                            #import pdb;pdb.set_trace()
                            merge_label=torch.arange(merge_sim.size(0))[merge_sim[:,i]>0].tolist()
                            if len(merge_label)>1:
                                feature=cluster_cen_sim[merge_label].type(torch.float)
                                neigh=torch.from_numpy(ori_0[merge_label,:topk]).cuda()

                                tmp=torch.zeros(len(merge_label),len(cluster_cen_sim)).float().cuda()
                                A=feature.mm(cluster_cen_sim.t())
                                tmp[torch.arange(len(merge_label)).view(-1,1).repeat(1,topk),neigh]=A[torch.arange(len(merge_label)).view(-1,1).repeat(1,topk),neigh]
                                adj=tmp.mm(torch.gt(tmp.t(),0).float())
                                adj[torch.arange(len(feature)),torch.arange(len(feature))]=0#self-->0

                                max_conf,max_label=torch.max(adj,0)
                                #<cluster_merge_thre_second to be 0
                                A_ori=feature.mm(feature.t())
                                merge_label_new=torch.zeros_like(A_ori)
                                merge_label_new[torch.arange(len(feature)),max_label]=1
                                print('A_ori:',A_ori)
                                # print('merge_label_new:',merge_label_new)
                                # if len(feature)>2:
                                #     import pdb;pdb.set_trace()
                                merge_label_new[A_ori<cluster_merge_thre_second]=0

                                cen_sum=torch.sum(merge_label_new,dim=1)
                                for id in range(len(merge_label)):
                                    if cen_sum[id]>0:
                                        ll=(torch.arange(len(merge_label_new))[merge_label_new[id]==1]).item()
                                        #print('ll:',ll)
                                        merge_label_0=-1
                                        change_guys=[]
                                        if merge_label[id] in merge_map:
                                            merge_label_0=merge_map[merge_label[id]]
                                            change_guys.append(merge_map[merge_label[id]])
                                        if ll in merge_map:
                                            merge_label_0=merge_map[ll]
                                            change_guys.append(merge_map[ll])
                                        if merge_label_0!=-1:
                                            for change_label,update_label in merge_map.items():
                                                if update_label in change_guys:
                                                    merge_map[change_label]=merge_label_0
                                        merge_label_0=merge_label[id] if merge_label_0==-1 else merge_label_0
                                        merge_map[ll]=merge_label_0
                                        merge_map[merge_label[id]]=merge_label_0
                        print("len(merge_map):",len(merge_map.keys()))
                        for change_label,update_label in merge_map.items():
                            ctx.labels[ctx.labels==int(change_label)]=int(update_label)
                #split cluster
                # dist_thre=1
                # for i in range(len(indexes)):
                #     label=ctx.labels[indexes[i]]
                #     feature=ctx.features[ctx.labels==label]
                #     if len(feature)>1:
                #         import pdb;pdb.set_trace()
                #         sim=feature.mm(feature.t())
                #         dist=2-2*sim
                #         if torch.mean(dist)>dist_thre:
                #             print('split-->2')
                            #self.gcn_s()
                ctx.label_cache[ctx.source_classes:]=ctx.labels[ctx.source_classes:]


            if mode=="cluster_sim":
                #cluster sim
                sim_thre=0.3
                neighbor_thre=0.45
                cluster_merge_thre=0.55
                #neighbor_weights=torch.tensor([0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05]).cuda()
                neighbor_num=30
                #import pdb;pdb.set_trace()
                ori_labels=ctx.labels[indexes]
                max_label=(ctx.labels).max()+1
                ctx.labels[indexes]=1
                sim = torch.zeros(max_label, (ctx.features[indexes]).size(0)).float().cuda()
                feature_sim=(ctx.features[indexes]).mm(ctx.features.t())
                sim.index_add_(0, ctx.labels, feature_sim.t().contiguous())
                nums = torch.zeros(max_label, 1).float().cuda()
                nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                mask = (nums>0).float()
                sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
                sim[:ctx.source_classes,:]=0

                if not knn_gcn:
                    feature_sim[:,:ctx.source_classes]=0
                    #import pdb;pdb.set_trace()
                    feature_sim[:,1]=1
                    sim[1]=1 #keep indexes in the same batch
                    #debug
                    knn_neighbor = compute_knn(ctx.features.clone(),k1=neighbor_num+1)
                    rerank_dist_6 = compute_jaccard_distance_inital_rank_index(ctx.features.clone(), indexes.cpu().numpy().tolist(),k1=10, k2=6,print_flag=False)
                    rerank_dist_1 = compute_jaccard_distance_inital_rank_index(ctx.features.clone(), indexes.cpu().numpy().tolist(),k1=10, k2=1,print_flag=False)
                    import pdb;pdb.set_trace()

                    index_neighbor=knn_neighbor[indexes.cpu().numpy(),1:]
                    index_neighbor_label=ctx.labels[torch.from_numpy(index_neighbor).cuda()]
                    index_neighbor_sim=feature_sim[torch.arange(len(indexes)).view(-1,1).repeat(1,neighbor_num).cuda(),torch.from_numpy(index_neighbor).cuda()]

                    keep_cluster=torch.gt(sim,sim_thre)
                    keep_cluster_label=keep_cluster[index_neighbor_label,torch.arange(len(indexes)).view(-1,1).repeat(1,neighbor_num)]
                    index_neighbor_label=torch.where(keep_cluster_label>0,index_neighbor_label,torch.zeros_like(index_neighbor_label))
                    index_neighbor_label=torch.where(index_neighbor_sim>neighbor_thre,index_neighbor_label,torch.zeros_like(index_neighbor_label))
                else:
                    pred, knn_neighbor,ori_knn_neighbor=ctx.gcn_n(indexes,ctx.features,ctx.labels,1,output_feat=True)
                    topk=10

                    new_method=1
                    if not new_method:
                        knn_neighbor=torch.from_numpy(knn_neighbor[:,1:topk]).cuda()

                        pred=pred.view(-1,neighbor_num)[:,:topk]
                        ori_pred=pred[:,0]
                        pred=pred[:,1:]
                        import pdb;pdb.set_trace()
                        index_neighbor_label=ctx.labels[knn_neighbor]

                        #import pdb;pdb.set_trace()
                        feature_sim[:,:ctx.source_classes]=0
                        #import pdb;pdb.set_trace()
                        feature_sim[:,1]=1
                        sim[1]=1 #keep indexes in the same batch
                        # for i in range(len(indexes)):
                        #        sim[ori_labels[i]][i]=sim[ori_labels[i]][i] if sim[ori_labels[i]][i]>0 else 1
                        sim[ori_labels,torch.arange(len(indexes)).cuda()]=torch.where(sim[ori_labels,torch.arange(len(indexes)).cuda()]>0,sim[ori_labels,torch.arange(len(indexes)).cuda()],torch.ones_like(sim[ori_labels,torch.arange(len(indexes)).cuda()]))

                        #step2 filter neighbor with low sim
                        neighbor_sim=feature_sim[torch.arange(len(indexes)).view(-1,1).repeat(1,topk-1).cuda(),knn_neighbor]
                        index_neighbor_label=torch.where(neighbor_sim>neighbor_thre,index_neighbor_label,torch.zeros_like(index_neighbor_label))
                        #index_neighbor_label=torch.where((torch.sum(neighbor_sim[:,:5],1)>neighbor_sim_sum_thre).view(-1,1).repeat(1,topk-1),index_neighbor_label,torch.zeros_like(index_neighbor_label))
                        #print("neighbor_sim:",neighbor_sim)
                        #index_neighbor_label=torch.where(torch.sum(neighbor_sim>0.45,dim=1).float().view(-1,1).repeat(1,topk-1)>4,index_neighbor_label,torch.zeros_like(index_neighbor_label))

                        #step3 filter neighbor with conf lower than self
                        pred_index_neighbor_label=torch.where(pred>=ori_pred.view(-1,1).repeat(1,topk-1).cuda(),index_neighbor_label,ori_labels.view(-1,1).repeat(1,topk-1))
                        index_neighbor_label=torch.where(index_neighbor_label>0,pred_index_neighbor_label,torch.zeros_like(index_neighbor_label))

                        #step1 filter clusters lower than thre
                        keep_cluster=torch.gt(sim,sim_thre)
                        keep_cluster_label=keep_cluster[index_neighbor_label,torch.arange(len(indexes)).view(-1,1).repeat(1,topk-1).cuda()]
                        index_neighbor_label=torch.where(keep_cluster_label>0,index_neighbor_label,torch.zeros_like(index_neighbor_label))

                        top2=0
                        if top2:
                            for i in range(len(indexes)):
                                ss=torch.sum(index_neighbor_label[i][:2]>0)
                                if ss>0:
                                    index_neighbor_label[i][0]=index_neighbor_label[i][index_neighbor_label[i]>0][0]
                        index_neighbor=knn_neighbor
                    else:
                        topk=3
                        update_topk=3
                        knn_neighbor0=knn_neighbor
                        index_neighbor=torch.from_numpy(knn_neighbor[:,1:topk]).cuda()
                        knn_neighbor=(knn_neighbor[:,:topk]).reshape(-1)
                        knn_neighbor,knn_index=np.unique(knn_neighbor,return_index=True)
                        pred=(pred.view(-1,neighbor_num)[:,:topk]).contiguous().view(-1)[knn_index.tolist()]

                        index_set=set(indexes.cpu().tolist())
                        neighbor_rank=(torch.argsort(-pred)).cpu().numpy()
                        map={}
                        cnt=0
                        for _,idx in enumerate(indexes):
                            map[idx.item()]=cnt
                            cnt+=1
                        sim[ori_labels,torch.arange(len(indexes)).cuda()]=torch.where(sim[ori_labels,torch.arange(len(indexes)).cuda()]>0,sim[ori_labels,torch.arange(len(indexes)).cuda()],torch.ones_like(sim[ori_labels,torch.arange(len(indexes)).cuda()]))


                        # try:
                        #     label_cache=torch.zeros_like(ctx.labels)
                        #     label_cnt=torch.zeros_like(ctx.labels)
                        #     label_tmp=torch.zeros(update_topk,dtype=torch.int64).cuda()
                        #     #update for everyone
                        #     for _,neighbor in enumerate(neighbor_rank):
                        #         # if label_cache[knn_neighbor[neighbor]]==0:
                        #         #     label_tmp=ctx.labels[knn_neighbor[neighbor]]
                        #         # else:
                        #         #     label_tmp=label_cache[knn_neighbor[neighbor]]
                        #         keep_guys=(ori_knn_neighbor[knn_neighbor[neighbor],:update_topk]).tolist()
                        #         for guy in keep_guys:
                        #             if (knn_neighbor[neighbor] in ori_knn_neighbor[guy,:topk]):
                        #                 if label_cache[guy]==0 and (guy in indexes) and feature_sim[map[guy]][knn_neighbor[neighbor]]>0.5:
                        #                     label_cache[guy]=ctx.labels[knn_neighbor[neighbor]] #if label_cache[knn_neighbor[neighbor]]==0 else label_cache[knn_neighbor[neighbor]]
                        #                 label_cnt[guy]+=1
                        #         #label_cache[keep_guys]=torch.where(label_cache[keep_guys]==0,label_tmp,label_cache[keep_guys])
                        #         #label_cnt[keep_guys]+=1
                        #     index_cnt=torch.from_numpy(knn_neighbor0[:,:topk]).cuda()
                        #     index_cnt=torch.sum(label_cnt[index_cnt],1)
                        #     ctx.labels[indexes]=label_cache[indexes]
                        #     outliers=indexes[index_cnt<=10]
                        #     print("outliers:",len(outliers))
                        #     if len(outliers)>0:
                        #         ctx.labels[outliers]=0
                        #     #print("Done")
                        #     #import pdb;pdb.set_trace()
                        # except:
                        #     import pdb;pdb.set_trace()
                        for _,neighbor in enumerate(neighbor_rank):
                            union=(set((ori_knn_neighbor[knn_neighbor[neighbor],1:update_topk]).tolist()) & index_set)
                            if knn_neighbor[neighbor] in index_set: #add self
                                ctx.labels[knn_neighbor[neighbor]]=ori_labels[map[knn_neighbor[neighbor]]]
                                index_set=index_set-set([knn_neighbor[neighbor]])
                            if len(union)>0:
                                for x in union:
                                    if (knn_neighbor[neighbor] in ori_knn_neighbor[x,1:topk]) and feature_sim[map[x]][knn_neighbor[neighbor]]>0.45:
                                        ctx.labels[x]=ctx.labels[int(knn_neighbor[neighbor])]
                                        index_set=index_set-set([x])

                        print("len(index_set):",len(index_set))
                        #others-->outliers
                        #import pdb;pdb.set_trace()
                        all_neighbor=ori_knn_neighbor[knn_neighbor0[:,1:topk].reshape(-1),1:update_topk].reshape(-1,2*(update_topk-1))
                        ori_indexes=np.repeat((indexes.cpu().numpy()).reshape(-1,1),2*(update_topk-1),axis=1)
                        neighbor_sum=np.sum(all_neighbor==ori_indexes,axis=1)
                        outliers=np.where(neighbor_sum<1)[0]
                        if len(outliers)>0:
                            ctx.labels[indexes[outliers.tolist()]]=0
                        print("len(outliers):",len(outliers))
                        if (len(index_set)>0):
                            ctx.labels[list(index_set)]=0
                        index_neighbor_label=ctx.labels[indexes]

                        sim[1]=1

                        keep_cluster=torch.gt(sim,sim_thre)
                        keep_cluster_label=keep_cluster[index_neighbor_label,torch.arange(len(indexes)).cuda()]
                        index_neighbor_label=torch.where(keep_cluster_label>0,index_neighbor_label,torch.zeros_like(index_neighbor_label))

                        index_neighbor_label=index_neighbor_label.view(-1,1)

                        #import pdb;pdb.set_trace()


                    # for i in range(len(indexes)):
                    #     ss=torch.sum(index_neighbor_label[i]>0)
                    #     if ss>0:
                    #         #if len(torch.unique(index_neighbor_label[i]))==ss:
                    #         index_neighbor_label[i][0]=index_neighbor_label[i][index_neighbor_label[i]>0][0]
                            # else:
                            #     mm=np.zeros(torch.max(index_neighbor_label[i]).item()+1)
                            #     for x in index_neighbor_label[i]:
                            #         mm[x.item()]+=1
                            #     index_neighbor_label[i][0]=int(np.argmax(mm))

                    #print("conf:",conf)

                    ##debug
                    # try:
                    #     k1=10
                    #     k2=6
                    #     neighbor_thre=0.55
                    #     rerank_dist = compute_jaccard_distance_inital_rank_index(ctx.features.clone(), indexes.cpu().numpy().tolist(),k1=k1, k2=k2,print_flag=False)
                    #     rerank_dist[np.where(rerank_dist<0.000001)]=1
                    #     close_neighbor=np.argsort(rerank_dist,axis=1)[:,:5]
                    #     close_neighbor_sim=np.sort(rerank_dist,axis=1)[:,:5]
                    #     sim[1]=1 #keep indexes in the same batch
                    #     keep_cluster=torch.gt(sim,sim_thre)
                    #     close_neighbor_label=ctx.labels[torch.from_numpy(close_neighbor).cuda()]
                    #     close_neighbor_label=torch.where(torch.from_numpy(close_neighbor_sim).cuda()<=neighbor_thre,close_neighbor_label,torch.zeros_like(close_neighbor_label))
                    #     keep_cluster_label=keep_cluster[close_neighbor_label,torch.arange(len(indexes)).view(-1,1).repeat(1,5)]
                    #
                    #     close_neighbor_label=torch.where(keep_cluster_label>0,close_neighbor_label,torch.zeros_like(close_neighbor_label))
                    #     print("index_neighbor_label[:,0]:",index_neighbor_label[:,0])
                    #     print(close_neighbor_label.shape)
                    #     ##
                    #     #ori_knn_neighbor[]
                    # except:
                    #     import pdb;pdb.set_trace()
                    # import pdb;pdb.set_trace()

                #calculate
                nums[:ctx.source_classes]=1
                nums[ori_labels]=1
                empty_label=torch.eq(nums,0).view(-1)
                #outliers_label=(torch.arange(len(nums))[empty_label>0])
                if torch.sum(empty_label>0)<64:
                    outliers_label=torch.arange(ctx.labels.max()+1,ctx.labels.max()+1+len(indexes)).cuda()
                else:
                    outliers_label=(torch.arange(len(nums))[empty_label>0])[-len(indexes):].cuda()

                try:
                    generate_label=torch.where(index_neighbor_label[:,0].view(-1)>0,index_neighbor_label[:,0].view(-1),outliers_label)
                except:
                    import pdb;pdb.set_trace()

                ctx.labels[indexes]=generate_label
                if torch.sum(torch.eq(generate_label,1))>0:
                    #import pdb;pdb.set_trace()
                    cnt=0
                    loop=torch.sum(torch.eq(generate_label,1)).item()
                    new_generate_label=torch.where(index_neighbor_label[:,0].view(-1)>1,index_neighbor_label[:,0].view(-1),outliers_label)
                    ctx.labels[indexes]=new_generate_label
                    while(cnt<loop):
                        for i in range(len(indexes)):
                            if (generate_label[i]==1):
                                    ctx.labels[indexes[i]]=ctx.labels[index_neighbor[i,0]]
                        cnt+=1
                    if torch.sum(torch.eq(ctx.labels[indexes],1))>0:
                        print('---------------------')
                        import pdb;pdb.set_trace()

                #merge cluster
                sim[1]=0
                sim[ori_labels,torch.arange(len(indexes)).cuda()]=torch.where(sim[ori_labels,torch.arange(len(indexes)).cuda()]<0.9999,sim[ori_labels,torch.arange(len(indexes)).cuda()],torch.zeros_like(sim[ori_labels,torch.arange(len(indexes)).cuda()]))

                merge_sim=torch.gt(sim,cluster_merge_thre)
                merge_sim_num=torch.sum(merge_sim,dim=0)
                if torch.sum(merge_sim_num>1)>0:
                    merge_map={}
                    for i in range(len(indexes)):
                        if merge_sim_num[i]>1:
                            #import pdb;pdb.set_trace()
                            merge_label=torch.arange(merge_sim.size(0))[merge_sim[:,i]>0]
                            if ctx.labels[indexes[i]].item() not in merge_label:
                                continue
                            #merge_label_0=merge_label[0].item()
                            merge_label_0=-1
                            inter=set(merge_label.tolist()) & set(merge_map.keys())
                            if len(inter)>0:
                                inter_label=list(inter)
                                merge_label_0=merge_map[inter_label[0]]
                                if len(inter_label)>1:
                                    change_guys=[]
                                    for label in inter_label:
                                        change_guys.append(merge_map[label])
                                    for change_label,update_label in merge_map.items():
                                        if update_label in change_guys:
                                            merge_map[change_label]=merge_label_0
                            merge_label_0=merge_label[0].item() if merge_label_0==-1 else merge_label_0
                            for label in merge_label:
                                merge_map[label.item()]=merge_label_0
                    print("len(merge_map):",len(merge_map.keys()))
                    for change_label,update_label in merge_map.items():
                        ctx.labels[ctx.labels==int(change_label)]=int(update_label)

                # confidence=torch.zeros(len(indexes)).cuda()
                # generate_label=torch.zeros(len(indexes)).cuda()
                # base=torch.zeros(neighbor_num).cuda()
                # for i in range(len(indexes)):
                #     max_label=torch.bincount(index_neighbor_label[i])
                #     if torch.max(max_label)==1:
                #         confidence[i]=0
                #         generate_label[i]=outliers_label[i]
                #     else:
                #         belong_label=torch.argmax(max_label)
                #         if belong_label != 1:
                #             base=torch.where(index_neighbor_label[i]==belong_label,torch.ones_like(base),-1*torch.ones_like(base))
                #             base=torch.where(index_neighbor_label[i]<=ctx.source_classes,-1*torch.ones_like(base),base) #source domain
                #             confidence[i]=torch.sum(base*index_neighbor_sim[i])/neighbor_num

                #index_confidence=


            if mode=="cluster_center":
                #cluster sim
                sim_thre=0.3
                neighbor_thre=0.55
                cluster_merge_thre=0.55
                #import pdb;pdb.set_trace()
                ori_labels=ctx.labels[indexes]
                ctx.labels[indexes]=1
                sim = torch.zeros((ctx.labels).max()+1, (ctx.features[indexes]).size(0)).float().cuda()
                feature_sim=(ctx.features[indexes]).mm(ctx.features.t())
                sim.index_add_(0, ctx.labels, feature_sim.t().contiguous())
                nums = torch.zeros(ctx.labels.max()+1, 1).float().cuda()
                nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                mask = (nums>0).float()
                sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
                sim[:ctx.source_classes,:]=0

                k1=ctx.k1
                k2=ctx.k2
                feature_sim[:,:ctx.source_classes]=0
                rerank_dist = compute_jaccard_distance_inital_rank_index(ctx.features.clone(), indexes.cpu().numpy().tolist(),k1=k1, k2=k2,print_flag=False)
                rerank_dist[np.where(rerank_dist<0.000001)]=1
                close_neighbor=np.argsort(rerank_dist,axis=1)[:,:5]
                close_neighbor_sim=np.sort(rerank_dist,axis=1)[:,:5]

                #import pdb;pdb.set_trace()
                sim[1]=1 #keep indexes in the same batch
                keep_cluster=torch.gt(sim,sim_thre)
                close_neighbor_label=ctx.labels[torch.from_numpy(close_neighbor).cuda()]
                close_neighbor_label=torch.where(torch.from_numpy(close_neighbor_sim).cuda()<=neighbor_thre,close_neighbor_label,torch.zeros_like(close_neighbor_label))
                keep_cluster_label=keep_cluster[close_neighbor_label,torch.arange(len(indexes)).view(-1,1).repeat(1,5)]

                close_neighbor_label=torch.where(keep_cluster_label>0,close_neighbor_label,torch.zeros_like(close_neighbor_label))

                nums[:ctx.source_classes]=1
                empty_label=torch.eq(nums,0).view(-1)
                outliers_label=(torch.arange(len(nums))[empty_label>0][-len(indexes):]).cuda()
                if len(outliers_label)<64:
                    outliers_label=torch.arange(ctx.labels.max()+1,ctx.labels.max()+1+len(indexes)).cuda()
                #outliers_label=torch.argsort(empty_label.view(-1).contiguous())[-len(indexes):]+ctx.source_classes
                #close_neighbor_set=set(close_neighbor[:,0].tolist()) & set(indexes.cpu().numpy().tolist())
                #for idx in close_neighbor_set:
                #import pdb;pdb.set_trace()
                generate_label=torch.where(close_neighbor_label[:,0].view(-1)>0,close_neighbor_label[:,0].view(-1),outliers_label)
                #change label only when cnt%2==1
                #generate_label=torch.where(ctx.change_cnt[indexes]%2==1,generate_label,ori_labels)
                # for i in range(len(indexes)):
                #     max_cnt=torch.bincount(close_neighbor_label[i][close_neighbor_label[i]>0])
                #     if len(max_cnt)>0 and torch.max(max_cnt)>1:
                #         generate_label[i]=torch.argmax(max_cnt)
                ctx.labels[indexes]=generate_label
                if torch.sum(torch.eq(generate_label,1))>0:
                    #import pdb;pdb.set_trace()
                    cnt=0
                    loop=torch.sum(torch.eq(generate_label,1)).item()
                    new_generate_label=torch.where(close_neighbor_label[:,0].view(-1)>1,close_neighbor_label[:,0].view(-1),outliers_label)
                    ctx.labels[indexes]=new_generate_label
                    while(cnt<loop):
                        for i in range(len(indexes)):
                            if (generate_label[i]==1):
                                    ctx.labels[indexes[i]]=ctx.labels[close_neighbor[i,0]]
                        cnt+=1
                    if torch.sum(torch.eq(ctx.labels[indexes],1))>0:
                        print('---------------------')
                        import pdb;pdb.set_trace()

                #merge cluster
                sim[1]=0
                merge_sim=torch.gt(sim,cluster_merge_thre)
                merge_sim_num=torch.sum(merge_sim,dim=0)
                if torch.sum(merge_sim_num>1)>0:
                    merge_map={}
                    for i in range(len(indexes)):
                        if merge_sim_num[i]>1:
                            #import pdb;pdb.set_trace()
                            merge_label=torch.arange(merge_sim.size(0))[merge_sim[:,i]>0]
                            if ctx.labels[indexes[i]].item() not in merge_label:
                                continue
                            #merge_label_0=merge_label[0].item()
                            merge_label_0=-1
                            inter=set(merge_label.tolist()) & set(merge_map.keys())
                            if len(inter)>0:
                                inter_label=list(inter)
                                merge_label_0=merge_map[inter_label[0]]
                                if len(inter_label)>1:
                                    change_guys=[]
                                    for label in inter_label:
                                        change_guys.append(merge_map[label])
                                    for change_label,update_label in merge_map.items():
                                        if update_label in change_guys:
                                            merge_map[change_label]=merge_label_0
                            merge_label_0=merge_label[0].item() if merge_label_0==-1 else merge_label_0
                            for label in merge_label:
                                merge_map[label.item()]=merge_label_0
                    print("len(merge_map):",len(merge_map.keys()))
                    for change_label,update_label in merge_map.items():
                        ctx.labels[ctx.labels==int(change_label)]=int(update_label)

            if mode=="single":
                use_all_knn=0
                ori_label=ctx.labels[indexes]
                ctx.labels[indexes]=1

                sim = torch.zeros((ctx.labels).max()+1, (ctx.features[indexes]).size(0)).float().cuda()
                feature_sim=(ctx.features[indexes]).mm(ctx.features.t())
                sim.index_add_(0, ctx.labels, feature_sim.t().contiguous())
                nums = torch.zeros(ctx.labels.max()+1, 1).float().cuda()
                nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                mask = (nums>0).float()
                sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
                #reranking-->single
                k1=ctx.k1
                k2=ctx.k2
                changelabel_thre=ctx.changelabel_thre
                reranking_thre_single=0.4
                feature_sim[:,:ctx.source_classes]=0
                sim_thre=0.4

                if not use_all_knn:
                    reranking_feature=torch.sum(torch.gt(feature_sim,reranking_thre_single),dim=0)
                    keep_feature=ctx.features[reranking_feature>0]
                    cal_feature=keep_feature.clone()
                    rerank_dist = compute_jaccard_distance_inital_rank(cal_feature, k1=k1, k2=k2)
                else:
                    cal_feature=ctx.features.clone()
                    rerank_dist = compute_jaccard_distance_inital_rank_index(cal_feature, indexes.cpu().numpy().tolist(),k1=k1, k2=k2)


                #feature_sim=torch.where(feature_sim>0.999,torch.zeros_like(feature_sim),feature_sim)
                #top_k=30
                ####debug#########
                # cal_feature=ctx.features.clone()
                # all_rerank_dist = compute_jaccard_distance_inital_rank(cal_feature,k1=k1, k2=k2)
                # all_neighbor_dist=np.sort(all_rerank_dist,axis=1)[:,1:6]
                # all_neighbor_dist=all_neighbor_dist[indexes.cpu().numpy()]
                #
                # rerank_dist = compute_jaccard_distance_inital_rank_index(cal_feature, indexes.cpu().numpy().tolist(),k1=k1, k2=k2)
                # #keep_arg=keep_id.cpu().numpy()
                # import pdb;pdb.set_trace()
                # #all_index0=np.array([np.argwhere(keep_arg==indexes[i].item())[0] for i in range(len(indexes))]).reshape(-1)
                # neighbor_dist=np.sort(rerank_dist,axis=1)[:,1:6]
                # print(all_neighbor_dist[:10])
                # print(neighbor_dist[:10])
                ####################

                #calculate outliers:guys that are not in others dist low confidence
                #neighbor=np.argsort(rerank_dist,axis=1)[:,1:6]

                #neighbor_dist=np.sort(rerank_dist,axis=1)[:,1:6]
                #sim_sum=np.sum(neighbor_dist,axis=1)
                #import pdb;pdb.set_trace()
                ################################################

                #change label
                #rerank_dist=rerank_dist[indexes.cpu().numpy()]
                rerank_dist[np.where(rerank_dist<0.000001)]=1#remove self
                #import pdb;pdb.set_trace()
                #close_neighbor=np.argsort(rerank_dist,axis=1)[:,:5]
                #close_neighbor_sim=np.sort(rerank_dist,axis=1)[:,:5]
                close_guy=np.argmin(rerank_dist,axis=1)
                close_sim=rerank_dist.min(1)
                if not use_all_knn:
                    keep_arg=torch.arange(ctx.features.size(0))[reranking_feature>0].numpy()

                    #keep_index=np.where(keep_arg in indexes.cpu().numpy().tolist())
                    all_index0=np.array([np.argwhere(keep_arg==indexes[i].item())[0] for i in range(len(indexes))]).reshape(-1)
                    close_sim=torch.from_numpy(close_sim[all_index0]).cuda()
                    close_guy=torch.from_numpy(keep_arg[close_guy[all_index0]]).cuda()
                    all_index=torch.from_numpy(all_index0).cuda()
                else:
                    close_sim=torch.from_numpy(close_sim).cuda()
                    close_guy=torch.from_numpy(close_guy).cuda()
                #print("close_sim:",close_sim)
                #import pdb;pdb.set_trace()
                #max_labels=torch.where((nums.view(-1)[ctx.labels[indexes]]-nums.view(-1)[ctx.labels[close_guy]])>0,ctx.labels[indexes],ctx.labels[close_guy])
                #nums.index_add_(0,ctx.label_cache, torch.ones(ctx.num_samples,1).float().cuda())
                nums[:ctx.source_classes]=1
                empty_label=torch.eq(nums,0).view(-1)
                outliers_label=torch.arange(len(nums))[empty_label>0][-len(indexes):].cuda()
                if len(outliers_label)<64:
                    outliers_label=torch.arange(ctx.labels.max()+1,ctx.labels.max()+1+len(indexes)).cuda()

                #ori_label=ctx.labels.clone()
                #print("nums:",(nums[ctx.labels[indexes]]).view(-1))
                # for idx,index in enumerate(indexes):
                #      if close_sim[idx]<changelabel_thre:
                #         #import pdb;pdb.set_trace()
                #         ori_indexes=torch.nonzero(torch.eq(ctx.labels,ctx.labels[index].item())).view(-1)
                #         ctx.labels[ori_indexes]=ori_label[close_guy[idx]]
                #import pdb;pdb.set_trace()
                # cnt=0
                # while 1:
                #     #change to the maximum num label
                #     belong_label=torch.where(sim[ctx.labels[close_guy]]>sim_thre,ctx.labels[close_guy],ctx.labels[indexes])
                #     ctx.labels[indexes]=torch.where(close_sim<changelabel_thre,belong_label,ctx.labels[indexes])
                #     ctx.labels[close_guy]=torch.where(close_sim<changelabel_thre,belong_label,ctx.labels[close_guy])
                    # if (torch.equal(ori_label,ctx.labels[indexes])):
                    #     break
                    # cnt+=1
                    # if (cnt>1):
                    #     #import pdb;pdb.set_trace()
                    #     remain_indexes=indexes[torch.ne(ori_label,ctx.labels[indexes])>0]
                    #     remain_close_guy=close_guy[torch.ne(ori_label,ctx.labels[indexes])>0]
                    #     for idx in range(len(remain_indexes)):
                    #         nums_index=nums[ctx.labels[remain_indexes[idx]]]
                    #         nums_close_guy=nums[ctx.labels[remain_close_guy[idx]]]
                    #     if nums_index>=nums_close_guy:
                    #         ctx.labels[remain_close_guy[idx]]=ctx.labels[remain_indexes[idx]]
                    #     else:
                    #         ctx.labels[remain_indexes[idx]]=ctx.labels[remain_close_guy[idx]]
                    #     print('--------')
                    # if (cnt>20):
                #import pdb;pdb.set_trace()
                #belong_label=torch.where(sim[ctx.labels[close_guy],np.arange(len(indexes))].view(-1)>sim_thre,ctx.labels[indexes],ctx.labels[close_guy])
                #print(sim[ctx.labels[close_guy],np.arange(len(indexes))])
                #ctx.labels[indexes]=torch.where(close_sim<changelabel_thre,belong_label,ctx.labels[indexes])
                #ctx.labels[close_guy]=torch.where(close_sim<changelabel_thre,belong_label,ctx.labels[close_guy])
                ctx.labels[indexes]=torch.where(close_sim<changelabel_thre,ctx.labels[close_guy],ctx.labels[indexes])
                ctx.labels[indexes]=torch.where(close_sim>=changelabel_thre,outliers_label,ctx.labels[indexes])
                #ctx.change_cnt[indexes]=1
                #su=torch.sum(torch.eq(ctx.change_cnt[ctx.source_classes:],0))
                #print("sum:",su)
                # if su<2000:#update
                #     print('-----------------------update-------------------------')
                #     #others remained
                #     ctx.change_cnt[:ctx.source_classes]=1
                #     #remain_features=ctx.features[ctx.change_cnt==0]
                #     indexes=torch.nonzero(torch.eq(ctx.change_cnt,0)).view(-1).contiguous()
                #     rerank_dist = compute_jaccard_distance_inital_rank_index(cal_feature, indexes.cpu().numpy().tolist(),k1=k1, k2=k2)
                #     rerank_dist[np.where(rerank_dist<0.000001)]=1#remove self
                #     close_guy=np.argmin(rerank_dist,axis=1)
                #     close_sim=rerank_dist.min(1)
                #     close_sim=torch.from_numpy(close_sim).cuda()
                #     close_guy=torch.from_numpy(close_guy).cuda()
                #     nums = torch.zeros(ctx.labels.max()+1, 1).float().cuda()
                #     nums.index_add_(0, ctx.labels, torch.ones(ctx.num_samples,1).float().cuda())
                #     nums.index_add_(0,ctx.label_cache, torch.ones(ctx.num_samples,1).float().cuda())
                #     empty_label=torch.eq(nums[ctx.source_classes:],0)
                #     outliers_label=torch.argsort(empty_label.view(-1).contiguous())[-len(indexes):]+ctx.source_classes
                #
                #     ctx.label_cache[indexes]=torch.where(close_sim<changelabel_thre,ctx.labels[close_guy],ctx.labels[indexes])
                #     ctx.label_cache[indexes]=torch.where(close_sim>=changelabel_thre,outliers_label,ctx.labels[indexes])
                #
                #     ctx.labels[ctx.source_classes:]=ctx.label_cache[ctx.source_classes:]
                #ctx.labels[indexes]=torch.where(close_sim<changelabel_thre,ctx.labels[close_guy],ctx.labels[indexes])
                #ctx.labels[indexes]=torch.where(close_sim>=changelabel_thre,outliers_label,ctx.labels[indexes])
                #noisy guys
                # neighbor=torch.from_numpy(keep_arg[neighbor[all_index0]]).cuda()
                # neighbor_dist=torch.from_numpy(neighbor_dist[all_index0]).cuda()
                # neighbor_label=[ctx.labels[neighbor[i]] for i in range(len(neighbor))]
                # neighbor_unique_label_num=torch.tensor([len(torch.unique(neighbor_label[i])) for i in range(len(neighbor_label))]).cuda()
                # ctx.labels[indexes]=torch.where(neighbor_unique_label_num>=3,outliers_label,ctx.labels[indexes])


                del cal_feature


        return grad_inputs, None, None, None,None, None, None,None, None, None,None,None,None,None,None,None,None,None


def hm(inputs, indexes, features,domain,update_label_thre_max, update_label_thre_min,labels,source_classes,num_samples,momentum=0.5,changelabel_thre=0.3,k1=10,k2=1,change_cnt=None,label_cache=None,confidence=None,gcn_n=None,gcn_s=None):
    return HM.apply(inputs, indexes, features,domain,update_label_thre_max,
    update_label_thre_min, labels,source_classes,num_samples,torch.Tensor([momentum]).to(inputs.device),changelabel_thre,k1,k2,change_cnt,label_cache,confidence,gcn_n,gcn_s)


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, source_classes,source_samples,temp=0.05, momentum=0.2,update_label_thre_max=0.97,update_label_thre_min=0.55,changelabel_thre=0.3,cluster_k1=10,cluster_k2=1,src_feat=0):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.source_classes=source_classes
        self.source_samples=source_samples

        self.momentum = momentum
        self.temp = temp
        self.update_label_thre_max=update_label_thre_max
        self.update_label_thre_min=update_label_thre_min
        #for clustering
        self.changelabel_thre=changelabel_thre
        self.cluster_k1=cluster_k1
        self.cluster_k2=cluster_k2

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())
        self.register_buffer('label_cache', torch.zeros(1).long()) #not use now
        self.register_buffer('change_cnt', torch.zeros(num_samples).long())
        if not src_feat:
            #self.register_buffer('s_features', torch.zeros(source_samples, num_features))
            #self.register_buffer('s_label', torch.zeros(source_samples).long())
            #self.register_buffer('s_sub_label', torch.zeros(source_samples).long())
            self.register_buffer('t_sub_label', torch.zeros(num_samples).long())
        else:
            self.register_buffer('sub_label', torch.zeros(num_samples).long())
        #self.register_buffer('empty_label', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes,domain=0,gcn_n=None,gcn_s=None):#domain=0:source domain=1:target
        # inputs: B*2048, features: L*2048
        inputs= hm(inputs, indexes, self.features, domain,self.update_label_thre_max,
            self.update_label_thre_min,self.labels,self.source_classes,self.num_samples,self.momentum,self.changelabel_thre,self.cluster_k1,self.cluster_k2,self.change_cnt,self.label_cache,None,gcn_n,gcn_s)
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

        del sim,nums
        return F.nll_loss(torch.log(masked_sim+1e-6), targets)
