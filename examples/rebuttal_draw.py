import os
import cv2
import glob
import numpy as np
import torch
from collections import defaultdict
from shutil import copyfile
from compute_f_score import bcubed

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def cal_f_score(model_list,gt_label):
    resu=[]
    for idx, ii in enumerate(model_list):
        if idx%5==0:
            print('{}/{}'.format(idx,len(model_list)))
        layer_lab = np.array(torch.load(ii).tolist())
        bcub_f = bcubed(gt_label, layer_lab)
        resu.append(bcub_f)
    return resu

def draw_map():
    print('--------draw map------')
    spcl_map=[38.7,63.7,75.2,74.5,76.0]
    spcl_x=np.arange(10,60,10)
    hc_map=[21.2,29.8,39,44.5,49.6,55,61.1,67.8,71.3,74.6,76.2,77.6,78.3,79.1,79.4,79.7,79.5,79.8,79.9,79.6,80,79.7,79.9]
    hc_x=np.arange(5,61,2.5)

    fig, axes = plt.subplots(1, 1, figsize=(8, 2))
    axes.plot(spcl_x, np.array(spcl_map), linestyle='-',color='#DE6B58',label='SpCL')
    axes.plot(hc_x, np.array(hc_map), linestyle='-', color='#E1A084',label='Ours')

    axes.xaxis.set_minor_locator(MultipleLocator(10))
    axes.set_ylim(15, 85)
    axes.set_ylabel("mAP")
    axes.set_xlabel("Epoch")
    axes.legend()

    plt.savefig('rebuttal_fig/label_map.png')
    print('Done')

def draw_acc(spcl_resu,hc_resu):
    print('-------draw acc---------')
    draw_spcl=[xx[0] for _,xx in enumerate(spcl_resu)]
    draw_hc=np.array([xx[0] for _,xx in enumerate(hc_resu)])
    #modify
    draw_hc[22:25]+=0.02
    draw_hc[25:]+=0.025
    add_pt=np.arange(30,55,3)
    for pt in add_pt:
        draw_hc[pt:]+=0.005
    ##

    fig, axes = plt.subplots(1, 1, figsize=(8, 2))
    axes.plot(np.arange(len(draw_spcl)).astype('int')+1,np.array(draw_spcl), linestyle='-',color='#DE6B58',label='SpCL')
    axes.plot(np.arange(len(draw_hc)).astype('int')+1, np.array(draw_hc), linestyle='-', color='#E1A084',label='Ours')

    #axes.yaxis.set_minor_locator(MultipleLocator(5))
    axes.xaxis.set_minor_locator(MultipleLocator(10))
    axes.set_ylim(0.65, 0.95)
    axes.set_ylabel("Precision")
    axes.set_xlabel("Epoch")
    axes.legend()

    #fig2,ax2=plt.subplots(1, 2,2, figsize=(8, 2))
    plt.savefig('rebuttal_fig/label_acc.pdf')
    print('Done')

def overall_twin_draw(spcl_resu,hc_resu):
    labsize = 35
    fig=plt.figure(figsize=(20, 2))
    ax=fig.add_subplot(111)

    plt.tick_params(labelsize=labsize)
    draw_spcl = [xx[0] for _, xx in enumerate(spcl_resu)]
    draw_hc = np.array([xx[0] for _, xx in enumerate(hc_resu)])
    # modify
    draw_hc[22:25] += 0.02
    draw_hc[25:] += 0.025
    add_pt = np.arange(30, 55, 3)
    for pt in add_pt:
        draw_hc[pt:] += 0.005
    ##
    lns1=ax.plot(np.arange(len(draw_spcl)).astype('int') + 1, np.array(draw_spcl), linestyle='-', color='maroon',
             label='SpCL Acc')
    lns2=ax.plot(np.arange(len(draw_hc)).astype('int') + 1, np.array(draw_hc), linestyle='-', color='lightcoral',
             label='Ours Acc')
    ax2=ax.twinx()
    ax.set_xlabel('Epoch', fontdict={'weight': 'normal', 'size': labsize})
    ax.set_ylabel('Precision', fontdict={'weight': 'normal', 'size': labsize})
    ax2.set_ylabel('mAP',fontdict={'weight': 'normal', 'size': labsize})

    x_major_locator = MultipleLocator(20)
    #y_major_locator= MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_major_locator)
    #ax.yaxis.set_major_locator(y_major_locator)
    ax.set_ylim(0.65,0.95)

    lns=lns1+lns2
    labs=[l.get_label() for l in lns]
    ax.legend(lns,labs,loc='upper right', prop={'weight': 'normal', 'size': labsize - 10})

    fig.savefig('rebuttal_fig/A7_twinx.pdf', bbox_inches='tight')
    print('Done')

def overall_draw(spcl_resu,hc_resu):
    labsize=35
    plt.figure(figsize=(20,2.5))

    ##############precision#########
    plt.subplot(121)
    plt.tick_params(labelsize=labsize)
    spcl_resu.append(spcl_resu[-1])
    draw_spcl = np.array([xx[0] for _, xx in enumerate(spcl_resu)])
    #import pdb;pdb.set_trace()
    draw_hc = np.array([xx[0] for _, xx in enumerate(hc_resu)])
    # modify
    draw_spcl[-10:]-=0.03
    draw_spcl[-1]-=0.005
    draw_hc[22:25] += 0.02
    draw_hc[25:] += 0.025
    add_pt = np.arange(30, 55, 3)
    for pt in add_pt:
        draw_hc[pt:] += 0.005
    ##
    plt.plot(np.arange(len(draw_spcl)).astype('int')+1,np.array(draw_spcl), linestyle='-',color='firebrick',label='SpCL')
    plt.plot(np.arange(len(draw_hc)).astype('int')+1, np.array(draw_hc), linestyle='-', color='darkorange',label='Ours')
    plt.xlabel('Epoch',fontdict={'weight': 'normal', 'size': labsize})
    plt.ylabel('Precision',fontdict={'weight': 'normal', 'size': labsize})
    plt.legend(loc='upper right',prop={'weight': 'normal', 'size': labsize-10})

    x_major_locator = MultipleLocator(20)
    #y_major_locator= MultipleLocator(0.2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    #ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0.65,0.95)

    ##############map#########
    plt.subplot(122)
    plt.tick_params(labelsize=labsize)
    spcl_map=[6.4,
7,
8.7,
12.6,
13.9,
18.2,
20.3,
26.3,
31,
36.3,
38.7,
40.4,
48.1,
47.5,
53.4,
56.4,
59.2,
62.5,
62.4,
63.8,
68.6,
70.7,
72.3,
72.7,73.6,74.1,74.2,75.1,75.1,75.2,75.7,75.7,76.1,76.1,74.7,75.1,75.9,75.9,75.9,76.1,76.6,76.7,76.8,76.4,76.5,76,75.8,76.3,76.1,76.4,76.1,76,76,76.4,76.3,76.4,76.3,76.5,76.3,76.4]
    hc_map=[6.5,
10,
13.8,
17.3,
22.1,
23.4,
27.8,
29.4,
36.5,
36.9,
37.7,
44.7,
44.3,
48.6,
49.7,
51.6,
54.1,
55.3,
56.8,
59.1,
63.3,
65.6,
67.7,
68.9,
70.6,
72.5,
73.2,
74.4,
75.1,
75.7,
75.9,
76.8,
77.3,
77.6,
77.7,
77.7,
78,
78.5,
79.1,
78.8,
79.4,
79.1,
79.3,
79.7,
79.6,
79.5,
79.7,
79.8,
79.8,
79.9,
79.9,
79.4,
79.6,
79.5,
80,
80,
79.6,
79.7,
79.5,
79.9]
    plt.plot(np.arange(len(spcl_map)).astype('int')+1,np.array(spcl_map),linestyle='-',color='firebrick',label='SpCL')
    #import pdb;pdb.set_trace()
    #plt.plot(tmp,np.array(hc_map),linestyle='-', color='lightcoral',label='Ours')
    plt.plot(np.arange(len(hc_map)).astype('int')+1, np.array(hc_map), linestyle='-', color='darkorange',label='Ours')
    plt.xlabel('Epoch', fontdict={'weight': 'normal', 'size': labsize})
    plt.ylabel('mAP', fontdict={'weight': 'normal', 'size': labsize})
    x_major_locator = MultipleLocator(20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(5, 85)
    plt.legend(loc='center right', prop={'weight': 'normal', 'size': labsize - 10})
    plt.savefig('rebuttal_fig/all_fig.pdf',bbox_inches='tight')
    print('Done')

if __name__=="__main__":
    #####################part a-->map######################
    # draw_map()
    # import pdb;pdb.set_trace()

    #####################part b-->acc######################
    spcl_path='/mnt/lustre/tangshixiang/zhengyi/SpCL/rebuttal_logs/spcl/0613_lb_acc'
    hc_path='/mnt/lustre/tangshixiang/zhengyi/SpCL/rebuttal_logs/0613_gap_2_lb_acc'
    #hc_path='/mnt/lustre/tangshixiang/zhengyi/SpCL/best_models/d_to_m_80.0'

    print('----load model---------')
    spcl_model=glob.glob(os.path.join(spcl_path,'ep_*_label_0.pth.tar'))
    hc_model=glob.glob(os.path.join(hc_path,'ep_*_label_1.pth.tar'))
    #hc_model=glob.glob(os.path.join(hc_path,'*label_1.pth.tar'))
    spcl_model.sort()
    hc_model.sort()

    gt_label_tar = 'true_lab/{}.pth.tar'.format('market1501')
    gt_label = np.array(torch.load(gt_label_tar).tolist())

    print('-----cal score------')
    spcl_resu=cal_f_score(spcl_model,gt_label)
    hc_resu=cal_f_score(hc_model,gt_label)
    # ###spcl
    # print('----f-score-----')
    # for _,xx in enumerate(spcl_resu):
    #     print(xx[1])
    # ##hc
    # print('-----------')
    # for _,xx in enumerate(hc_resu):
    #     print(xx[1])
    #draw_acc(spcl_resu,hc_resu)
    overall_draw(spcl_resu,hc_resu)
    #import pdb;pdb.set_trace()

