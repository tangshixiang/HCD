from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
sys.path.append("..")
import collections
import copy
import time
import os
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from spcl import datasets
from spcl import models
from spcl.models.dsbn import convert_dsbn, convert_bn
from spcl.models.hm import HybridMemory
from spcl.trainers import SpCLTrainer_UDA
from spcl.evaluators import Evaluator, extract_features
from spcl.utils.data import IterLoader
from spcl.utils.data import transforms as T
from spcl.utils.data.sampler import RandomMultipleGallerySampler,RandomMultipleGallerySampler_Target,RandomMultipleGallerySampler_Target_sub_cluster
from spcl.utils.data.preprocessor import Preprocessor
from spcl.utils.logging import Logger
from spcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from spcl.utils.faiss_rerank import compute_jaccard_distance,compute_jaccard_distance_step1,compute_jaccard_distance_inital_rank

from examples.gcn_v import gcn_n,graph_estimator,graph_selection

from tensorboardX import SummaryWriter
writer_dir='./logs_initial_rank_k1_10'
os.makedirs(writer_dir,exist_ok=True)
writer = SummaryWriter(writer_dir)
start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None,target_cnt=None,sub_label=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if trainset is None:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
        else:
            if sub_label is None:
                sampler=RandomMultipleGallerySampler_Target(train_set, num_instances,target_cnt)
            else:
                sampler=RandomMultipleGallerySampler_Target_sub_cluster(train_set, num_instances,target_cnt,sub_label)
    else:
        sampler = None
    #import pdb;pdb.set_trace()
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args,model=None):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0,model=model)
    # adopt domain-specific BN
    convert_dsbn(model)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)
    #pretrain(args)

def pretrain(args):
    print("---------pretrain-------------")
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load source-domain dataset")
    dataset_source = get_data(args.dataset_source, args.data_dir)
    print("==> Load target-domain dataset")
    dataset_target = get_data(args.dataset_target, args.data_dir)
    testset=dataset_source.query[:]
    testset.extend(dataset_source.gallery)
    test_loader_source = get_test_loader(dataset_source, args.height, args.width, args.batch_size, args.workers,testset=testset)
    train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                        args.batch_size, args.workers, 4, iters)
    source_classes = dataset_source.num_train_pids

    # Create model
    model = create_model(args)

    # Create hybrid memory
    # memory = HybridMemory(model.module.num_features, source_classes+len(dataset_target.train),source_classes,
    #                         temp=args.temp, momentum=args.momentum).cuda()
    # gcnv
    #GCN_N = gcn_n(2048,4096,source_classes+len(dataset_target.train),source_classes)
    GCN_N=graph_estimator(2048,2048,source_classes,source_classes)
    memory = HybridMemory(model.module.num_features, source_classes,source_classes,
                            temp=args.temp, momentum=args.momentum,changelabel_thre=args.changelabel_thre,cluster_k1=args.cluster_k1,cluster_k2=args.cluster_k2).cuda()

    # Initialize source-domain class centroids
    print("==> Initialize source-domain class centroids in the hybrid memory")
    sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset_source.train))
    source_features, _ = extract_features(model, sour_cluster_loader, print_freq=50)
    sour_fea_dict = collections.defaultdict(list)
    for f, pid, _ in sorted(dataset_source.train):
        sour_fea_dict[pid].append(source_features[f].unsqueeze(0))
    source_centers = [torch.cat(sour_fea_dict[pid],0).mean(0) for pid in sorted(sour_fea_dict.keys())]
    source_centers = torch.stack(source_centers,0)
    source_centers = F.normalize(source_centers, dim=1)

    ###draw target
    draw=0
    if draw:
        draw_feature=np.zeros((0,2048),dtype=np.float32)
        draw_label=[]
        for f, pid, _ in sorted(dataset_source.train):
            draw_feature=np.concatenate((draw_feature,source_features[f].unsqueeze(0).numpy()))
            draw_label.append(pid)
        draw_feature.tofile('spcl/models/features.bin')
        with open('spcl/models/label.txt','w') as f:
            for lab in draw_label:
                f.write(str(lab)+'\n')
        print("Done")
        import pdb;pdb.set_trace()

    # Initialize target-domain instance features
    print("==> Initialize target-domain instance features in the hybrid memory")
    #tgt_cluster_loader = get_test_loader(dataset_target, args.height, args.width,
    #                                args.batch_size, args.workers, testset=sorted(dataset_target.train))
    #target_features, _ = extract_features(model, tgt_cluster_loader, print_freq=50)
    #target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in sorted(dataset_target.train)], 0)
    #memory.features = torch.cat((source_centers, F.normalize(target_features, dim=1)), dim=0).cuda()
    memory.features=source_centers.cuda()
    #del tgt_cluster_loader, source_centers, target_features, sour_cluster_loader, sour_fea_dict

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = SpCLTrainer_UDA(model, memory, source_classes,writer,GCN_N)

    #fc layer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for epoch in range(args.epochs):

        if (epoch==0):
            memory.labels=torch.arange(source_classes).cuda()

        iters=400
        #print("iters:",iters)
        train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                            args.batch_size, args.workers, 4, iters)
        #import pdb;pdb.set_trace()

        train_loader_source.new_epoch()

        trainer.train(epoch, train_loader_source, train_loader_source, optimizer,
                    print_freq=args.print_freq)

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP = evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery, cmc_flag=False)
            is_best = (mAP>best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print ('==> Test with the best model on the target domain:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))



def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load source-domain dataset")
    dataset_source = get_data(args.dataset_source, args.data_dir)
    print("==> Load target-domain dataset")
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                        args.batch_size, args.workers, 4, iters)
    source_classes = dataset_source.num_train_pids

    # Create model
    #model_path='/mnt/lustre/zhengyi1/paper/resnet50_duke_xent.pth.tar'
    model = create_model(args)

    # Create hybrid memory
    # memory = HybridMemory(model.module.num_features, source_classes+len(dataset_target.train),source_classes,
    #                         temp=args.temp, momentum=args.momentum).cuda()
    # gcnv
    #GCN_N = gcn_n(2048,4096,source_classes+len(dataset_target.train),source_classes)
    if args.use_gcnv:
        GCN_N=gcn_n(2048,2048,source_classes+len(dataset_target.train),source_classes)
    else:
        GCN_N=graph_estimator(2048,2048,source_classes+len(dataset_target.train),source_classes)
    GCN_S=None
    #GCN_S=graph_selection(2048,4096,source_classes+len(dataset_target.train),source_classes)
    if args.gcn_tgt:
        source_sam=len(dataset_source.train)+len(dataset_target.train)
    else:
        source_sam=len(dataset_source.train)
    memory = HybridMemory(model.module.num_features, source_classes+len(dataset_target.train),source_classes,source_sam,
                            temp=args.temp, momentum=args.momentum,changelabel_thre=args.changelabel_thre,cluster_k1=args.cluster_k1,cluster_k2=args.cluster_k2).cuda()

    # Initialize source-domain class centroids
    print("==> Initialize source-domain class centroids in the hybrid memory")
    sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset_source.train))
    source_features, _ = extract_features(model, sour_cluster_loader, print_freq=50)
    #import pdb;pdb.set_trace()
    sour_fea_dict = collections.defaultdict(list)
    source_label=[]
    all_source_feature=[]
    for f, pid, _ in sorted(dataset_source.train):
        sour_fea_dict[pid].append(source_features[f].unsqueeze(0))
        source_label.append(pid)
        all_source_feature.append(source_features[f])
    memory.s_label[:len(dataset_source.train)]=torch.tensor(source_label).cuda()
    memory.s_label[len(dataset_source.train):]=memory.s_label.max()+2 #target domain
    memory.s_features[:len(dataset_source.train)]=torch.stack(all_source_feature,0).cuda()
    source_centers = [torch.cat(sour_fea_dict[pid],0).mean(0) for pid in sorted(sour_fea_dict.keys())]
    source_centers = torch.stack(source_centers,0)
    source_centers = F.normalize(source_centers, dim=1)

    ###draw target
    draw=0
    if draw:
        draw_feature=np.zeros((0,2048),dtype=np.float32)
        draw_label=[]
        for f, pid, _ in sorted(dataset_source.train):
            draw_feature=np.concatenate((draw_feature,source_features[f].unsqueeze(0).numpy()))
            draw_label.append(pid)
        draw_feature.tofile('spcl/models/features.bin')
        with open('spcl/models/label.txt','w') as f:
            for lab in draw_label:
                f.write(str(lab)+'\n')
        print("Done")
        import pdb;pdb.set_trace()

    # Initialize target-domain instance features
    print("==> Initialize target-domain instance features in the hybrid memory")
    tgt_cluster_loader = get_test_loader(dataset_target, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset_target.train))
    target_features, _ = extract_features(model, tgt_cluster_loader, print_freq=50)
    target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in sorted(dataset_target.train)], 0)
    memory.features = torch.cat((source_centers, F.normalize(target_features, dim=1)), dim=0).cuda()
    del tgt_cluster_loader, source_centers, target_features, sour_cluster_loader, sour_fea_dict

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    params_gcn=[{"params": [value]} for _, value in GCN_N.named_parameters() if value.requires_grad]
    optimizer_gcn=torch.optim.SGD(params_gcn, lr=args.lr_gcn, weight_decay=args.weight_decay)
    lr_gcn=torch.optim.lr_scheduler.StepLR(optimizer_gcn, step_size=args.step_size_gcn, gamma=0.1)

    # Trainer
    trainer = SpCLTrainer_UDA(model, memory, source_classes,writer,GCN_N,GCN_S,len(dataset_target.train),len(dataset_source.train))

    target_cnt=np.zeros(len(memory.features[source_classes:])).astype('int')
    all_iters=0
    pretrain=0
    if pretrain:
        #'prepare for gcn'
        for epoch in range(10):
            iters=400
            memory.labels = torch.cat((torch.arange(source_classes), torch.zeros(len(dataset_target.train)).long()+source_classes)).cuda()
            train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                                args.batch_size, args.workers, 4, iters)
            train_loader_source.new_epoch()
            train_gcn=1 if epoch>=5 else 0
            trainer.train(epoch, train_loader_source, None, optimizer,
                        print_freq=args.print_freq, train_iters=iters,all_iters=all_iters,lr_scheduler=lr_scheduler,lr_scheduler_gcn=lr_gcn,optimizer_gcn=optimizer_gcn,
                        pretrain=1,train_gcn=train_gcn)
    epoch=-1
    eval_cnt=0
    while(all_iters<24000):
    #for epoch in range(args.epochs):
        epoch+=1
        target_features = memory.features[source_classes:].clone()

        if (epoch==0):
            # Calculate distance
            rerank_dist = compute_jaccard_distance_inital_rank(target_features, k1=args.k1, k2=args.k2)
            # DBSCAN cluster
            eps = args.eps
            eps_tight = eps-args.eps_gap
            eps_loose = eps+args.eps_gap
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
            cluster = DBSCAN(eps=eps, min_samples=3, metric='precomputed', n_jobs=-1)
            cluster_tight = DBSCAN(eps=eps_tight, min_samples=3, metric='precomputed', n_jobs=-1)
            cluster_loose = DBSCAN(eps=eps_loose, min_samples=3, metric='precomputed', n_jobs=-1)
            # if (epoch==1):
            #     eps = args.eps
            #     eps_tight = eps-args.eps_gap
            #     eps_loose = eps+args.eps_gap
            #     print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
            #     cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            #     cluster_tight = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)
            #     cluster_loose = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)
            #     #import pdb;pdb.set_trace()
            del target_features

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
            pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)
            num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
            num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)

            # generate new dataset and calculate cluster centers
            def generate_pseudo_labels(cluster_id, num):
                labels = []
                outliers = 0
                for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset_target.train), cluster_id)):
                    if id!=-1:
                        labels.append(source_classes+id)
                    else:
                        labels.append(source_classes+num+outliers)
                        outliers += 1
                return torch.Tensor(labels).long()

            pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
            pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
            pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)

            # compute R_indep and R_comp
            N = pseudo_labels.size(0)
            #import pdb;pdb.set_trace()
            label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
            label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
            label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()

            R_comp = 1-torch.min(label_sim, label_sim_tight).sum(-1)/torch.max(label_sim, label_sim_tight).sum(-1)
            R_indep = 1-torch.min(label_sim, label_sim_loose).sum(-1)/torch.max(label_sim, label_sim_loose).sum(-1)
            assert((R_comp.min()>=0) and (R_comp.max()<=1))
            assert((R_indep.min()>=0) and (R_indep.max()<=1))

            cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
            cluster_img_num = collections.defaultdict(int)
            for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
                cluster_R_comp[label.item()-source_classes].append(comp.item())
                cluster_R_indep[label.item()-source_classes].append(indep.item())
                cluster_img_num[label.item()-source_classes]+=1

            cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
            cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
            cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if cluster_img_num[num]>1]
            if (epoch==0):
                indep_thres = np.sort(cluster_R_indep_noins)[min(len(cluster_R_indep_noins)-1,np.round(len(cluster_R_indep_noins)*0.9).astype('int'))]

                pseudo_labeled_dataset = []
                outliers = 0
                for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), pseudo_labels)):
                    indep_score = cluster_R_indep[label.item()-source_classes]
                    comp_score = R_comp[i]
                    if ((indep_score<=indep_thres) and (comp_score.item()<=cluster_R_comp[label.item()-source_classes])):
                        pseudo_labeled_dataset.append((fname,label.item(),cid))
                    else:
                        pseudo_labeled_dataset.append((fname,source_classes+len(cluster_R_indep)+outliers,cid))
                        pseudo_labels[i] = source_classes+len(cluster_R_indep)+outliers
                        outliers+=1
            memory.labels = torch.cat((torch.arange(source_classes), pseudo_labels)).cuda()
            memory.label_cache=torch.cat((torch.arange(source_classes), pseudo_labels)).cuda()
        else:
            su=torch.sum(torch.eq(memory.change_cnt[memory.source_classes:],0))
            print("no use sum:",su)
            if args.update_all_label:
                memory.change_cnt[:memory.source_classes]=1
                all_indexes=torch.arange(len(memory.change_cnt))[memory.change_cnt==0]

                all_batch=int(np.ceil(len(all_indexes)/64))
                pseudo_labeled_dataset = []
                pseudo_labels=memory.labels[source_classes:].cpu()
                tmp_memory_cnt=memory.change_cnt[memory.source_classes:]
                for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), pseudo_labels)):
                    if tmp_memory_cnt[i]==0:
                        pseudo_labeled_dataset.append((fname,label.item(),cid))
                update_loader_target = get_train_loader(args, dataset_target, args.height, args.width,
                                                    args.batch_size, args.workers, 0, all_batch,
                                                    trainset=pseudo_labeled_dataset,target_cnt=target_cnt)
                trainer.train(epoch, None, update_loader_target, None,train_iters=all_batch,only_update_label=1)
                for b in range(all_batch):
                    indexes=all_indexes[b*64:min((b+1)*64,len(all_indexes))]
                    if len(indexes)>0:
                        sim_thre=0.3
                        neighbor_thre=0.45
                        cluster_merge_thre=0.55
                        neighbor_num=30

                        ori_labels=memory.labels[indexes]
                        max_label=(memory.labels).max()+1
                        memory.labels[indexes]=1
                        sim = torch.zeros(max_label, (memory.features[indexes]).size(0)).float().cuda()
                        feature_sim=(memory.features[indexes]).mm(memory.features.t())
                        sim.index_add_(0, memory.labels, feature_sim.t().contiguous())
                        nums = torch.zeros(max_label, 1).float().cuda()
                        nums.index_add_(0, memory.labels, torch.ones(memory.num_samples,1).float().cuda())
                        mask = (nums>0).float()
                        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
                        sim[:memory.source_classes,:]=0

                        new_label=[]
                        neighbor,pred,ori_neighbor=GCN_N(indexes,memory.features,memory.labels,1,output_feat=True)
                        new_conf=[]
                        nearest_neighbor=ori_neighbor[:,1]

                        for i in range(len(indexes)):
                            index_dict={}
                            for idx,nei in enumerate(neighbor[i]):
                                index_dict[nei]=idx
                            top5_pred=[]
                            top5_index=[]
                            for idx,nei in enumerate(ori_neighbor[i,1:5]):
                                top5_pred.append(pred[i][index_dict[nei],1])
                                top5_index.append(nei)
                            max_conf,max_label=torch.max(torch.tensor(top5_pred),0)
                            la=memory.labels[top5_index[max_label.item()]].item()
                            new_label.append(la)
                            new_conf.append(max_conf.item())
                        new_label=torch.tensor(new_label).cuda()
                        sim[1]=1
                        label_sim=sim[new_label,torch.arange(len(indexes))]
                        new_label=torch.where(label_sim<=sim_thre,torch.zeros_like(new_label),new_label)
                        new_label=torch.where(new_label<memory.source_classes,torch.zeros_like(new_label),new_label)
                        close_neighbor=torch.tensor([x[1] for _,x in enumerate(neighbor)]).cuda()
                        new_label=torch.where(close_neighbor<=neighbor_thre,torch.zeros_like(new_label),new_label)
                        nums[:memory.source_classes]=1
                        #nums[ori_labels]=1
                        empty_label=torch.eq(nums,0).view(-1)
                        #outliers_label=(torch.arange(len(nums))[empty_label>0])
                        if torch.sum(empty_label>0)<64:
                            outliers_label=torch.arange(memory.labels.max()+1,memory.labels.max()+1+len(indexes)).cuda()
                        else:
                            outliers_label=(torch.arange(len(nums))[empty_label>0])[-len(indexes):].cuda()
                        new_label=torch.where(new_label>0,new_label,outliers_label)
                        memory.labels[indexes]=new_label

                        if torch.sum(torch.eq(new_label,1))>0:
                            #import pdb;pdb.set_trace()
                            cnt=0
                            loop=torch.sum(torch.eq(new_label,1)).item()
                            new_generate_label=torch.where(new_label>1,new_label,outliers_label)
                            memory.labels[indexes]=new_generate_label
                            while(cnt<loop):
                                for i in range(len(indexes)):
                                    if (new_label[i]==1):
                                            memory.labels[indexes[i]]=memory.labels[nearest_neighbor[i]]
                                cnt+=1
                            if torch.sum(torch.eq(memory.labels[indexes],1))>0:
                                print('---------------------')
                                import pdb;pdb.set_trace()

                        #merge cluster
                        #sim[1]=0
                        #sim[ori_labels,torch.arange(len(indexes)).cuda()]=torch.where(sim[ori_labels,torch.arange(len(indexes)).cuda()]<0.9999,sim[ori_labels,torch.arange(len(indexes)).cuda()],torch.zeros_like(sim[ori_labels,torch.arange(len(indexes)).cuda()]))

                        # merge_sim=torch.gt(sim,cluster_merge_thre)
                        # merge_sim_num=torch.sum(merge_sim,dim=0)
                        # if torch.sum(merge_sim_num>1)>0:
                        #     merge_map={}
                        #     for i in range(len(indexes)):
                        #         if merge_sim_num[i]>1:
                        #             #import pdb;pdb.set_trace()
                        #             merge_label=torch.arange(merge_sim.size(0))[merge_sim[:,i]>0]
                        #             if memory.labels[indexes[i]].item() not in merge_label:
                        #                 continue
                        #             #merge_label_0=merge_label[0].item()
                        #             merge_label_0=-1
                        #             inter=set(merge_label.tolist()) & set(merge_map.keys())
                        #             if len(inter)>0:
                        #                 inter_label=list(inter)
                        #                 merge_label_0=merge_map[inter_label[0]]
                        #                 if len(inter_label)>1:
                        #                     change_guys=[]
                        #                     for label in inter_label:
                        #                         change_guys.append(merge_map[label])
                        #                     for change_label,update_label in merge_map.items():
                        #                         if update_label in change_guys:
                        #                             merge_map[change_label]=merge_label_0
                        #             merge_label_0=merge_label[0].item() if merge_label_0==-1 else merge_label_0
                        #             for label in merge_label:
                        #                 merge_map[label.item()]=merge_label_0
                        #     print("len(merge_map):",len(merge_map.keys()))
                        #     for change_label,update_label in merge_map.items():
                        #         memory.labels[memory.labels==int(change_label)]=int(update_label)
                # pseudo_labeled_dataset = []
                # pseudo_labels=memory.labels[source_classes:].cpu()
                # tmp_memory_cnt=memory.change_cnt[memory.source_classes:]
                # for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), pseudo_labels)):
                #     if tmp_memory_cnt[i]==0:
                #         pseudo_labeled_dataset.append((fname,label.item(),cid))
                # iters=int(np.round(len(pseudo_labeled_dataset)/64))
                # update_loader_target = get_train_loader(args, dataset_target, args.height, args.width,
                #                                     args.batch_size, args.workers, 0, iters,
                #                                     trainset=pseudo_labeled_dataset,target_cnt=target_cnt)
                # trainer.train(epoch, None, update_loader_target, None,train_iters=iters,only_update_label=1)
                memory.change_cnt[memory.source_classes:]=0
            else:
                if su==0:
                    memory.change_cnt[memory.source_classes:]=0

            pseudo_labeled_dataset = []
            pseudo_labels=memory.labels[source_classes:].cpu()
            #pseudo_labels=memory.label_cache[source_classes:].cpu()
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), pseudo_labels)):
                pseudo_labeled_dataset.append((fname,label.item(),cid))

        pseudo_labels=memory.labels[source_classes:]
        # statistics of clusters and un-clustered instances
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
                    .format(epoch, (index2label>1).sum(), (index2label==1).sum(), 1-indep_thres))
        sub_cluster_label=memory.label_cache[source_classes:].tolist()
        print('sub cluster num:',len(set(sub_cluster_label)))
        if args.num_instances>0:
            #iters=int(((index2label>1).sum()*4+(index2label==1).sum()+(index2label[(index2label>4)]-4).sum())//(args.batch_size))+10
            iters=int(((index2label>1).sum()*4+(index2label==1).sum())//(args.batch_size))+1
        else:
            iters=len(dataset_target.train)//args.batch_size
        if epoch==0:
            iters=400
        print("iters:",iters)
        train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                            args.batch_size, args.workers, 4, iters)
        train_loader_target = get_train_loader(args, dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset,target_cnt=target_cnt,sub_label=sub_cluster_label)
        #import pdb;pdb.set_trace()

        train_loader_source.new_epoch()
        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_source, train_loader_target, optimizer,
                    print_freq=args.print_freq, train_iters=len(train_loader_target),all_iters=all_iters,lr_scheduler=lr_scheduler,lr_scheduler_gcn=lr_gcn,optimizer_gcn=optimizer_gcn)
        all_iters+=iters
        if (all_iters//1000-eval_cnt>0) or (all_iters>=24000):
            eval_cnt+=1
        #if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
            is_best = (mAP>best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        #lr_scheduler.step()

    print ('==> Test with the best model on the target domain:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

    # for epoch in range(args.epochs):
    #     print('==> Create pseudo labels for unlabeled target domain with self-paced policy')
    #     target_features = memory.features[source_classes:].clone()
    #     #change to compute sim
    #     #sim=torch.zeros((target_features.size(0),target_features.size(0)))
    #     #rerank_dist=(1-target_features.mm(target_features.t())).cpu().numpy()
    #     #import pdb;pdb.set_trace()
    #     #print("")
    #     #rerank_dist=np.maximum(0,rerank_dist)
    #     #cnt=target_features.size(0)/3000
    #     #for i in range(cnt):
    #
    #     #rerank_dist = compute_jaccard_distance_step1(target_features, k1=args.k1, k2=args.k2)
    #     #rerank_dist=compute_jaccard_distance_inital_rank(target_features,k1=args.k1, k2=args.k2)
    #
    #     if (epoch==0):
    #         # Calculate distance
    #         rerank_dist = compute_jaccard_distance_inital_rank(target_features, k1=args.k1, k2=args.k2)
    #         # DBSCAN cluster
    #         eps = args.eps
    #         eps_tight = eps-args.eps_gap
    #         eps_loose = eps+args.eps_gap
    #         print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
    #         cluster = DBSCAN(eps=eps, min_samples=3, metric='precomputed', n_jobs=-1)
    #         cluster_tight = DBSCAN(eps=eps_tight, min_samples=3, metric='precomputed', n_jobs=-1)
    #         cluster_loose = DBSCAN(eps=eps_loose, min_samples=3, metric='precomputed', n_jobs=-1)
    #         # if (epoch==1):
    #         #     eps = args.eps
    #         #     eps_tight = eps-args.eps_gap
    #         #     eps_loose = eps+args.eps_gap
    #         #     print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
    #         #     cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
    #         #     cluster_tight = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)
    #         #     cluster_loose = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)
    #         #     #import pdb;pdb.set_trace()
    #         del target_features
    #
    #         # select & cluster images as training set of this epochs
    #         pseudo_labels = cluster.fit_predict(rerank_dist)
    #         pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
    #         pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)
    #         num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    #         num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
    #         num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)
    #
    #         # generate new dataset and calculate cluster centers
    #         def generate_pseudo_labels(cluster_id, num):
    #             labels = []
    #             outliers = 0
    #             for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset_target.train), cluster_id)):
    #                 if id!=-1:
    #                     labels.append(source_classes+id)
    #                 else:
    #                     labels.append(source_classes+num+outliers)
    #                     outliers += 1
    #             return torch.Tensor(labels).long()
    #
    #         pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
    #         pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
    #         pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)
    #
    #         # compute R_indep and R_comp
    #         N = pseudo_labels.size(0)
    #         #import pdb;pdb.set_trace()
    #         label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
    #         label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
    #         label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()
    #
    #         R_comp = 1-torch.min(label_sim, label_sim_tight).sum(-1)/torch.max(label_sim, label_sim_tight).sum(-1)
    #         R_indep = 1-torch.min(label_sim, label_sim_loose).sum(-1)/torch.max(label_sim, label_sim_loose).sum(-1)
    #         assert((R_comp.min()>=0) and (R_comp.max()<=1))
    #         assert((R_indep.min()>=0) and (R_indep.max()<=1))
    #
    #         cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
    #         cluster_img_num = collections.defaultdict(int)
    #         for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
    #             cluster_R_comp[label.item()-source_classes].append(comp.item())
    #             cluster_R_indep[label.item()-source_classes].append(indep.item())
    #             cluster_img_num[label.item()-source_classes]+=1
    #
    #         cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
    #         cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
    #         cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if cluster_img_num[num]>1]
    #         if (epoch==0):
    #             indep_thres = np.sort(cluster_R_indep_noins)[min(len(cluster_R_indep_noins)-1,np.round(len(cluster_R_indep_noins)*0.9).astype('int'))]
    #
    #             pseudo_labeled_dataset = []
    #             outliers = 0
    #             for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), pseudo_labels)):
    #                 indep_score = cluster_R_indep[label.item()-source_classes]
    #                 comp_score = R_comp[i]
    #                 if ((indep_score<=indep_thres) and (comp_score.item()<=cluster_R_comp[label.item()-source_classes])):
    #                     pseudo_labeled_dataset.append((fname,label.item(),cid))
    #                 else:
    #                     pseudo_labeled_dataset.append((fname,source_classes+len(cluster_R_indep)+outliers,cid))
    #                     pseudo_labels[i] = source_classes+len(cluster_R_indep)+outliers
    #                     outliers+=1
    #         memory.labels = torch.cat((torch.arange(source_classes), pseudo_labels)).cuda()
    #         memory.label_cache=torch.cat((torch.arange(source_classes), pseudo_labels)).cuda()
    #     else:
    #         pseudo_labeled_dataset = []
    #         pseudo_labels=memory.labels[source_classes:].cpu()
    #         for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), pseudo_labels)):
    #             pseudo_labeled_dataset.append((fname,label.item(),cid))
    #     # elif (epoch>1):
    #     #     #update label for guys are not include in this epoch
    #     #     k1=10
    #     #     k2=6
    #     #     reranking_thre_single=0.4
    #     #     changelabel_thre=args.changelabel_thre
    #     #     memory.change_cnt[:memory.source_classes]=1
    #     #     indexes=torch.nonzero(torch.eq(memory.change_cnt,0)).view(-1).contiguous()
    #     #     print("remain guys:",len(indexes))
    #     #     feature_sim=(memory.features[indexes]).mm(memory.features.t())
    #     #     feature_sim[:,:memory.source_classes]=0
    #     #
    #     #     reranking_feature=torch.sum(torch.gt(feature_sim,reranking_thre_single),dim=0)
    #     #     keep_feature=memory.features[reranking_feature>0]
    #     #     cal_feature=keep_feature.clone()
    #     #     rerank_dist = compute_jaccard_distance_inital_rank(cal_feature, k1=k1, k2=k2)
    #     #
    #     #     rerank_dist[np.where(rerank_dist<0.000001)]=1#remove self
    #     #     close_guy=np.argmin(rerank_dist,axis=1)
    #     #     close_sim=rerank_dist.min(1)
    #     #     keep_arg=torch.arange(memory.features.size(0))[reranking_feature>0].numpy()
    #     #
    #     #     all_index0=np.array([np.argwhere(keep_arg==indexes[i].item())[0] for i in range(len(indexes))]).reshape(-1)
    #     #     close_sim=torch.from_numpy(close_sim[all_index0]).cuda()
    #     #     close_guy=torch.from_numpy(keep_arg[close_guy[all_index0]]).cuda()
    #     #     all_index=torch.from_numpy(all_index0).cuda()
    #     #
    #     #     nums = torch.zeros(memory.labels.max()+1, 1).float().cuda()
    #     #     nums.index_add_(0, memory.labels, torch.ones(memory.num_samples,1).float().cuda())
    #     #
    #     #     empty_label=torch.eq(nums[memory.source_classes:],0)
    #     #     outliers_label=torch.argsort(empty_label.view(-1).contiguous())[-len(indexes):]+memory.source_classes
    #     #
    #     #     memory.labels[indexes]=torch.where(close_sim<changelabel_thre,memory.labels[close_guy],memory.labels[indexes])
    #     #     memory.labels[indexes]=torch.where(close_sim>=changelabel_thre,outliers_label,memory.labels[indexes])
    #     #
    #     #     memory.change_cnt[memory.source_classes:]=0
    #
    #
    #     pseudo_labels=memory.labels[source_classes:]
    #     # statistics of clusters and un-clustered instances
    #     index2label = collections.defaultdict(int)
    #     for label in pseudo_labels:
    #         index2label[label.item()]+=1
    #     index2label = np.fromiter(index2label.values(), dtype=float)
    #     print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
    #                 .format(epoch, (index2label>1).sum(), (index2label==1).sum(), 1-indep_thres))
    #         #import pdb;pdb.set_trace()
    #     su=torch.sum(torch.eq(memory.change_cnt[memory.source_classes:],0))
    #     print("no use sum:",su)
    #     if su==0:
    #         memory.change_cnt[memory.source_classes:]=0
    #     train_loader_target = get_train_loader(args, dataset_target, args.height, args.width,
    #                                         args.batch_size, args.workers, args.num_instances, iters,
    #                                         trainset=pseudo_labeled_dataset,target_cnt=target_cnt)
    #
    #     train_loader_source.new_epoch()
    #     train_loader_target.new_epoch()
    #
    #     trainer.train(epoch, train_loader_source, train_loader_target, optimizer,
    #                 print_freq=args.print_freq, train_iters=len(train_loader_target))
    #
    #     if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
    #         mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
    #         is_best = (mAP>best_mAP)
    #         best_mAP = max(mAP, best_mAP)
    #         save_checkpoint({
    #             'state_dict': model.state_dict(),
    #             'epoch': epoch + 1,
    #             'best_mAP': best_mAP,
    #         }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
    #
    #         print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
    #               format(epoch, mAP, best_mAP, ' *' if is_best else ''))
    #
    #     lr_scheduler.step()
    #
    # print ('==> Test with the best model on the target domain:')
    # checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    # model.load_state_dict(checkpoint['state_dict'])
    # evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
    #
    # end_time = time.monotonic()
    # print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on UDA re-ID")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--use-gcnv', type=int, default=0, help="gcnv")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.00,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=10,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--lr-gcn', type=float, default=0.2,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--step-size-gcn', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--changelabel_thre', type=float, default=0.4)
    parser.add_argument('--update_all_label', type=int, default=0)
    parser.add_argument('--cluster_k1', type=int, default=10)
    parser.add_argument('--cluster_k2', type=int, default=6)
    parser.add_argument('--gcn-tgt', type=int, default=0)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
