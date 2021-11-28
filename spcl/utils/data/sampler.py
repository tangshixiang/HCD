from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if (pid<0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)
        #print("RandomMultipleGallerySampler")
        #import pdb;pdb.set_trace()

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        #print("--------sampler-------------")
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:
                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    #ind_indexes=select_indexes
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])
        return iter(ret)

class RandomMultipleGallerySampler_Target(Sampler):
    def __init__(self, data_source, num_instances=4,cnt=None):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if (pid<0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)
        self.cnt=cnt
        #self.cnt=np.zeros(len(self.index_pid.keys())).astype('int')
        #print("RandomMultipleGallerySampler")
        #import pdb;pdb.set_trace()

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        #print("--------sampler-------------")
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        loop=2
        while(loop):
            loop-=1
            for kid in indices:
                min_cnt=np.min(self.cnt[self.pid_index[self.pids[kid]]])
                remain_id=np.array(self.pid_index[self.pids[kid]])[np.where(self.cnt[self.pid_index[self.pids[kid]]]==min_cnt)[0]].tolist()
                i = random.choice(remain_id)

                _, i_pid, i_cam = self.data_source[i]

                ret.append(i)
                self.cnt[i]+=1

                pid_i = self.index_pid[i]
                cams = self.pid_cam[pid_i]
                index = self.pid_index[pid_i]
                select_cams = No_index(cams, i_cam)

                if select_cams:
                    cnt=self.cnt[np.array(index)[select_cams]]
                    min_cnt=np.min(cnt)
                    keep_select_cams=np.array(select_cams)[np.where(cnt==min_cnt)[0]]
                    if len(select_cams) >= self.num_instances:
                        if len(keep_select_cams)>=self.num_instances:
                            cam_indexes= np.random.choice(keep_select_cams, size=self.num_instances-1, replace=False)
                        else:
                            left_select_cams=list(set(select_cams) - set(keep_select_cams.tolist()))
                            left_indexes = np.random.choice(left_select_cams, size=self.num_instances-1-len(keep_select_cams), replace=False)
                            cam_indexes=np.concatenate((keep_select_cams,left_indexes))
                    else:
                        #cam_indexes = select_cams#203
                        cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                    for kk in cam_indexes:
                        ret.append(index[kk])
                        self.cnt[index[kk]]+=1

                else:
                    select_indexes = No_index(index, i)
                    if (not select_indexes): continue

                    cnt=self.cnt[np.array(index)[select_indexes]]
                    min_cnt=np.min(cnt)
                    keep_select_indexes=np.array(select_indexes)[np.where(cnt==min_cnt)[0]]
                    if len(select_indexes) >= self.num_instances:
                        if len(keep_select_indexes)>=self.num_instances:
                            ind_indexes= np.random.choice(keep_select_indexes, size=self.num_instances-1, replace=False)
                        else:
                            left_select_indexes=list(set(select_indexes) - set(keep_select_indexes.tolist()))
                            left_indexes = np.random.choice(left_select_indexes, size=self.num_instances-1-len(keep_select_indexes), replace=False)
                            ind_indexes=np.concatenate((keep_select_indexes,left_indexes))
                            #ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                    else:
                        #ind_indexes = select_indexes #203-->reduce duplicate
                        ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                    for kk in ind_indexes:
                        ret.append(index[kk])
                        self.cnt[index[kk]]+=1

        #make sure all index can change label
        # left_guys=np.where(self.cnt==0)[0]
        # np.random.shuffle(left_guys)
        # ret.extend(left_guys.tolist())
        #import pdb;pdb.set_trace()
        return iter(ret)

class RandomMultipleGallerySampler_Target_sub_cluster(Sampler):
    def __init__(self, data_source, num_instances=4,cnt=None,sub_label=None):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances
        self.sub_label=np.array(sub_label)

        for index, (_, pid, cam) in enumerate(data_source):
            if (pid<0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)
        self.cnt=cnt
        #self.cnt=np.zeros(len(self.index_pid.keys())).astype('int')
        #print("RandomMultipleGallerySampler")
        #import pdb;pdb.set_trace()

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        #print("--------sampler-------------")
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        loop=2
        check=0
        while(loop):
            loop-=1
            for kid in indices:
                min_cnt=np.min(self.cnt[self.pid_index[self.pids[kid]]])
                remain_id=np.array(self.pid_index[self.pids[kid]])[np.where(self.cnt[self.pid_index[self.pids[kid]]]==min_cnt)[0]].tolist()
                i = random.choice(remain_id)

                _, i_pid, i_cam = self.data_source[i]

                ret.append(i)
                self.cnt[i]+=1

                pid_i = self.index_pid[i]
                cams = self.pid_cam[pid_i]
                index = self.pid_index[pid_i]
                select_cams = No_index(cams, i_cam)

                if select_cams:
                    cnt=self.cnt[np.array(index)[select_cams]]
                    min_cnt=np.min(cnt)
                    keep_select_cams=np.array(select_cams)[np.where(cnt==min_cnt)[0]]
                    if len(select_cams) >= self.num_instances:
                        if len(keep_select_cams)>=self.num_instances:
                            cam_indexes= np.random.choice(keep_select_cams, size=self.num_instances-1, replace=False)
                        else:
                            left_select_cams=list(set(select_cams) - set(keep_select_cams.tolist()))
                            left_indexes = np.random.choice(left_select_cams, size=self.num_instances-1-len(keep_select_cams), replace=False)
                            cam_indexes=np.concatenate((keep_select_cams,left_indexes))
                    else:
                        #cam_indexes=select_cams#203
                        cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                    for kk in cam_indexes:
                        ret.append(index[kk])
                        self.cnt[index[kk]]+=1

                else:
                    select_indexes = No_index(index, i)
                    if (not select_indexes): continue

                    cnt=self.cnt[np.array(index)[select_indexes]]
                    min_cnt=np.min(cnt)
                    keep_select_indexes=np.array(select_indexes)[np.where(cnt==min_cnt)[0]]
                    if len(select_indexes) >= self.num_instances:
                        if len(keep_select_indexes)>=self.num_instances:
                            #sub cluster
                            sub_label=self.sub_label[np.array(index)[select_indexes]]
                            keep_select_sublabel=sub_label[np.where(cnt==min_cnt)[0]]
                            if len(np.unique(keep_select_sublabel))>=self.num_instances-1:
                                check+=1
                                keep_label=np.random.choice(np.unique(keep_select_sublabel),size=self.num_instances-1,replace=False)
                                ind_indexes=[]
                                for lab in keep_label:
                                    ind_indexes.append(int(np.random.choice(keep_select_indexes[np.where(keep_select_sublabel==lab)[0]],size=1)))
                                ind_indexes=np.array(ind_indexes)
                            else:
                                ind_indexes= np.random.choice(keep_select_indexes, size=self.num_instances-1, replace=False)
                        else:
                            left_select_indexes=list(set(select_indexes) - set(keep_select_indexes.tolist()))
                            left_indexes = np.random.choice(left_select_indexes, size=self.num_instances-1-len(keep_select_indexes), replace=False)
                            ind_indexes=np.concatenate((keep_select_indexes,left_indexes))
                            #ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                    else:
                        #ind_indexes = select_indexes  # 203-->reduce duplicate
                        ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                    for kk in ind_indexes:
                        ret.append(index[kk])
                        self.cnt[index[kk]]+=1

        #make sure all index can change label
        # left_guys=np.where(self.cnt==0)[0]
        # np.random.shuffle(left_guys)
        # ret.extend(left_guys.tolist())
        #import pdb;pdb.set_trace()
        print('--------check:{}--------'.format(check))
        return iter(ret)
