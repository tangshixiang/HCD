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
from spcl.trainers_online_label_propagation import SpCLTrainer_UDA
from spcl.evaluators import Evaluator, extract_features
from spcl.utils.data import IterLoader
from spcl.utils.data import transforms as T
from spcl.utils.data.sampler import RandomMultipleGallerySampler,RandomMultipleGallerySampler_Target,RandomMultipleGallerySampler_Target_sub_cluster
from spcl.utils.data.preprocessor import Preprocessor
from spcl.utils.logging import Logger
from spcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from spcl.utils.faiss_rerank import compute_jaccard_distance,compute_jaccard_distance_step1,compute_jaccard_distance_inital_rank

from spcl.models.hierarchy_gcn_label_propagation import Hierarchy_GCN,Utils,p_lp,s_lp,c_lp,split_lp

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
    if args.add_autoaug:
        train_transformer = T.Compose([
                 #T.RandomApply([T.AutoAugment()], p=0.1),
                 #T.RandomApply([T.ColorJitter(0.15, 0.15, 0.1, 0.1)], p=0.5),
                 T.Resize((height, width), interpolation=3),
                 T.RandomHorizontalFlip(p=0.5),
                 T.Pad(10),
                 T.RandomCrop((height, width)),
                 T.ToTensor(),
                 normalizer,
                 #T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
             ])
    else:
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

def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    #if args.dataset_target=="msmt17":
    #    assert args.iters==800
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

    source_sam=len(dataset_source.train)
    num_samp=source_classes+len(dataset_target.train) if not args.src_feat else len(dataset_source.train)+len(dataset_target.train)
    memory = HybridMemory(model.module.num_features, num_samp,source_classes,source_sam,
                            temp=args.temp, momentum=args.momentum,changelabel_thre=args.changelabel_thre,cluster_k1=args.cluster_k1,
                            cluster_k2=args.cluster_k2,src_feat=args.src_feat).cuda()

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

    #memory.s_label[:len(dataset_source.train)]=torch.tensor(source_label).cuda()
    #memory.s_label[len(dataset_source.train):]=memory.s_label.max()+2 #target domain
    #memory.s_sub_label[:len(dataset_source.train)]=torch.tensor(source_label).cuda()
    #memory.s_features[:len(dataset_source.train)]=torch.stack(all_source_feature,0).cuda()
    source_centers = [torch.cat(sour_fea_dict[pid],0).mean(0) for pid in sorted(sour_fea_dict.keys())]
    source_centers = torch.stack(source_centers,0)
    source_centers = F.normalize(source_centers, dim=1)

    # Initialize target-domain instance features
    print("==> Initialize target-domain instance features in the hybrid memory")
    tgt_cluster_loader = get_test_loader(dataset_target, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset_target.train))
    target_features, _ = extract_features(model, tgt_cluster_loader, print_freq=50)
    target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in sorted(dataset_target.train)], 0)

    memory.features = torch.cat((source_centers, F.normalize(target_features, dim=1)), dim=0).cuda()
    src_nums=source_classes
    del tgt_cluster_loader, source_centers, target_features, sour_cluster_loader, sour_fea_dict

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    #hierarchy_gcn
    # uti=Utils(args.k1,args.k2,args.sub_clustre_thre)
    # P_LP=p_lp(alpha=0.99)
    # S_LP=s_lp(alpha=0.99,topk_num=args.topk_num,method=args.method)
    # C_LP=c_lp(alpha=0.99,topk_num=args.topk_num,method=args.method)
    # Split_LP=split_lp(args.split_alpha,args.split_num,args.split_anchor_thre)
    # hier_gcn=Hierarchy_GCN(P_LP,S_LP,C_LP,Split_LP,uti,thre=args.thre,neighbor_num=args.neighbor_num,merge_wo_outlier=args.merge_wo_outlier)

    # params_p_lp=[{"params":[value]} for _, value in model.named_parameters() if value.requires_grad]
    # params_s_lp=[{"params":[value]} for _, value in model.named_parameters() if value.requires_grad]
    # params_c_lp=[{"params":[value]} for _, value in model.named_parameters() if value.requires_grad]
    # #params_split_gcn=[{"params":[value]} for _, value in Split_GCN.named_parameters() if value.requires_grad]
    # optimizer_p_lp=torch.optim.Adam(params_p_lp, lr=args.lr_lp, weight_decay=args.weight_decay)
    # optimizer_s_lp=torch.optim.Adam(params_s_lp, lr=args.lr_lp, weight_decay=args.weight_decay)
    # optimizer_c_lp=torch.optim.Adam(params_c_lp, lr=args.lr_lp, weight_decay=args.weight_decay)
    # #optimizer_split_gcn=torch.optim.SGD(params_split_gcn, lr=args.lr_gcn, weight_decay=args.weight_decay)
    # lr_p_lp=torch.optim.lr_scheduler.StepLR(optimizer_p_lp, step_size=args.step_size_gcn, gamma=0.1)
    # lr_s_lp=torch.optim.lr_scheduler.StepLR(optimizer_s_lp, step_size=args.step_size_gcn, gamma=0.1)
    # lr_c_lp=torch.optim.lr_scheduler.StepLR(optimizer_c_lp, step_size=args.step_size_gcn, gamma=0.1)
    # #lr_split_gcn=torch.optim.lr_scheduler.StepLR(optimizer_split_gcn, step_size=args.step_size_gcn, gamma=0.1)
    # optimizer_lp=[optimizer_p_lp,optimizer_s_lp,optimizer_c_lp]#,optimizer_split_gcn]
    # lr_lp=[lr_p_lp,lr_s_lp,lr_c_lp]#,lr_split_gcn]

    # Trainer
    trainer = SpCLTrainer_UDA(model, memory, src_nums,writer,None,len(dataset_target.train),len(dataset_source.train),update_iters=args.iters)

    target_cnt=np.zeros(len(memory.features[src_nums:])).astype('int')
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
    tgt_cnt=int(np.ceil(1.0*len(dataset_target.train)/64))
    while(all_iters<60*args.iters):
    #for epoch in range(args.epochs):
        epoch+=1
        target_features = memory.features[src_nums:].clone()

        #if (epoch==0):#initialize
        #if (epoch==0):
        if 1:
            # Calculate distance
            #rerank_dist = compute_jaccard_distance_inital_rank(target_features, k1=args.k1, k2=args.k2)
            rerank_dist = compute_jaccard_distance(target_features, k1=args.k1, k2=args.k2)
            # DBSCAN cluster
            eps = args.eps
            eps_tight = eps-args.eps_gap
            eps_loose = eps+args.eps_gap
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
            cluster = DBSCAN(eps=eps, min_samples=3, metric='precomputed', n_jobs=-1)
            cluster_tight = DBSCAN(eps=eps_tight, min_samples=3, metric='precomputed', n_jobs=-1)
            cluster_loose = DBSCAN(eps=eps_loose, min_samples=3, metric='precomputed', n_jobs=-1)

            del target_features

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
            pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)
            num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
            num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)

            # generate new dataset and calculate cluster centers
            def generate_pseudo_labels(cluster_id, num,debug=0):
                labels = []
                outliers = 0
                if debug:
                    print('-----debug----------')
                    for i, ((fname, pid, cid), id) in enumerate(zip(sorted(dataset_target.train), cluster_id)):
                        if id!=-1:
                            labels.append(src_nums+id)
                        else:
                            labels.append(src_nums+num+outliers)
                            outliers += 1
                    #     hier_gcn.debug_label.append(pid)
                    # hier_gcn.debug_label=torch.tensor(hier_gcn.debug_label).cuda()
                    #
                    # # for debug
                    # hier_gcn.debug_label_num = torch.zeros(hier_gcn.debug_label.max() + 1, 1).float().cuda()
                    # hier_gcn.debug_label_num.index_add_(0, hier_gcn.debug_label,
                    #                                 torch.ones(len(hier_gcn.debug_label), 1).float().cuda())
                    # hier_gcn.debug_label_num = hier_gcn.debug_label_num.cpu().view(-1).tolist()
                else:
                    for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset_target.train), cluster_id)):
                        if id!=-1:
                            labels.append(src_nums+id)
                        else:
                            labels.append(src_nums+num+outliers)
                            outliers += 1
                return torch.Tensor(labels).long()

            pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids,debug=0)
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
                cluster_R_comp[label.item()-src_nums].append(comp.item())
                cluster_R_indep[label.item()-src_nums].append(indep.item())
                cluster_img_num[label.item()-src_nums]+=1

            cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
            cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
            cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if cluster_img_num[num]>1]

            indep_thres = np.sort(cluster_R_indep_noins)[min(len(cluster_R_indep_noins)-1,np.round(len(cluster_R_indep_noins)*0.9).astype('int'))]

            pseudo_labeled_dataset = []
            outliers = 0
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), pseudo_labels)):
                indep_score = cluster_R_indep[label.item()-src_nums]
                comp_score = R_comp[i]
                if ((indep_score<=indep_thres) and (comp_score.item()<=cluster_R_comp[label.item()-src_nums])):
                    pseudo_labeled_dataset.append((fname,label.item(),cid))
                else:
                    pseudo_labeled_dataset.append((fname,src_nums+len(cluster_R_indep)+outliers,cid))
                    pseudo_labels[i] = src_nums+len(cluster_R_indep)+outliers
                    outliers+=1
            memory.labels = torch.cat((torch.arange(source_classes), pseudo_labels)).cuda()
            #memory.label_cache=torch.cat((torch.arange(source_classes), pseudo_labels)).cuda()
            #memory.t_sub_label=torch.arange(source_classes+len(pseudo_labels)).cuda()

            #memory.t_sub_label=torch.cat((torch.arange(source_classes), pseudo_labels)).cuda()
            memory.t_sub_label=torch.cat((torch.arange(source_classes), torch.arange(len(pseudo_labels))+source_classes)).cuda()
        else:
            #update after every epoch
            for cnt_i in range(tgt_cnt):
                t_indexes=torch.arange(64*cnt_i,min(64*(cnt_i+1),len(dataset_target.train))).cuda()
                trainer.hierarchy_gcn.inference(t_indexes + trainer.source_classes, trainer.memory, 1)  # target domain-->inference

        pseudo_labels=memory.labels[src_nums:]
        # statistics of clusters and un-clustered instances
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: >2 {} clusters, =2 {} clusters,{} un-clustered instances, R_indep threshold is {}'
                    .format(epoch, (index2label>2).sum(), (index2label==2).sum(),(index2label==1).sum(), 1-indep_thres))
        #sub_cluster_label=memory.label_cache[source_classes:].tolist()
        if not args.src_feat:
            sub_cluster_label=memory.t_sub_label[source_classes:].tolist()
        else:
            sub_cluster_label=memory.sub_label[src_nums:].tolist()
        print('sub cluster num:',len(set(sub_cluster_label)))

        iters=args.iters
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
                    print_freq=args.print_freq, train_iters=len(train_loader_target),all_iters=all_iters,
                    lr_scheduler=lr_scheduler,base=1)
        all_iters+=iters
        if ((all_iters//(2.5*args.iters)-eval_cnt>0) and all_iters<15*args.iters):
            eval_cnt+=1
        if ((all_iters//(2.5*args.iters)-eval_cnt>0) and all_iters>=15*args.iters) or (all_iters>=60*args.iters):
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
    parser.add_argument('--k1', type=int, default=20,
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
    parser.add_argument('--lr-lp', type=float, default=0.0,#0.00035,
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
    parser.add_argument('--sub-clustre-thre', type=int, default=8)
    parser.add_argument('--src-feat', type=int, default=0)
    parser.add_argument('--tgt-only', type=int, default=0)

    parser.add_argument('--split-type', type=int, default=0)
    parser.add_argument('--thre', type=float, nargs='+',default=[0.06,0.30,0.25])
    parser.add_argument('--split-num', type=int,default=8)
    parser.add_argument('--topk-num', type=float,default=0.45) #avg linkage thre
    parser.add_argument('--neighbor-num', type=int, default=96)
    parser.add_argument('--method', type=int, default=6)
    parser.add_argument('--merge-wo-outlier', type=int, default=0)
    parser.add_argument('--split-alpha', type=float, default=0.5)
    parser.add_argument('--split-anchor-thre', type=float, default=0.6)

    parser.add_argument('--add-autoaug', type=int, default=0)

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
