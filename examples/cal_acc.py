import os
import cv2
import glob
import numpy as np
import torch
from collections import defaultdict
from shutil import copyfile


def acc(gt_label,pred_label):#gt_label:list;pred_label:list
    gt_map=defaultdict(list)
    for idx, pid in enumerate(gt_label):
        gt_map[pid].append(idx)

    pred_map=defaultdict(list)
    for idx,pid in enumerate(pred_label):
        pred_map[pid].append(idx)

    recall=0
    precision=0
    F_score=0
    F_beta=1
    for idx,pid in enumerate(pred_label):
        recall_i=1.0*len(set(pred_map[pid]) & set(gt_map[gt_label[idx]]))/len(gt_map[gt_label[idx]])
        precision_i=1.0*len(set(pred_map[pid]) & set(gt_map[gt_label[idx]]))/len(pred_map[pid])
        f_i=(1+F_beta*F_beta)*recall_i*precision_i/(F_beta*F_beta*(recall_i+precision_i))

        recall+=recall_i
        precision+=precision_i
        F_score+=f_i
    avg_recall=recall/len(pred_label)
    avg_precision=precision/len(pred_label)
    avg_F=F_score/(len(pred_label))

    print('avg recall: {}\navg precision: {}\navg_F:{}'.format(avg_recall,avg_precision,avg_F))
    return avg_recall,avg_precision

def draw_iter_pic():
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    map_d=[]
    map_m=[]
    assert len(map_d)==len(map_m)

    plt.figure()
    plt.plot(np.arange(1, len(map_d) + 1).astype('int'), np.array(map_d), label=r'M $\rightarrow$ D', marker='o')
    plt.plot(np.arange(1, len(map_m) + 1).astype('int'), np.array(map_m), label=r'D $\rightarrow$ M', marker='o')
    plt.xlabel('number of levels')
    plt.legend()
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0.8, len(map_d) + 0.2)
    plt.savefig('ablation_fig/map_{}.pdf'.format(dataset_name))

def draw_iter_pic_single():
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    map_d=[68.1,68.4,68.4,68.8,70.1]
    #map_d=[]
    map_m=[74.3,79.3,79.3,78.0,79.9]

    if len(map_d)>0:
        plt.figure()
        plt.plot(np.arange(1, len(map_d) + 1).astype('int'), np.array(map_d), marker='o')
        plt.xlabel('number of levels')
        plt.title(r'M $\rightarrow$ D')
        x_major_locator = MultipleLocator(1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(0.8, len(map_d) + 0.2)
        plt.ylim(67,70)
        plt.savefig('ablation_fig/6.1map_M2D.pdf')

    if len(map_m)>0:
        plt.figure()
        plt.plot(np.arange(1, len(map_m) + 1).astype('int'), np.array(map_m), label=r'D $\rightarrow$ M', marker='o')
        plt.xlabel('number of levels')
        plt.legend()
        x_major_locator = MultipleLocator(1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(0.8, len(map_m) + 0.2)
        plt.savefig('ablation_fig/6.1map_D2M.pdf')


def draw_iter_pic_single_bar():
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    map_d=[69.1,69.4,69.4,69.8,70.1]
    #map_d=[]
    #map_m=[74.3,79.3,79.3,78.0,79.9]
    map_m=[]
    width=0.6

    if len(map_d)>0:
        plt.figure()
        plt.bar(np.arange(1, len(map_d) + 1),np.array(map_d),width)
        #plt.plot(np.arange(1, len(map_d) + 1).astype('int'), np.array(map_d), marker='o')
        plt.xlabel('number of levels')
        plt.title(r'M $\rightarrow$ D')
        x_major_locator = MultipleLocator(1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        #plt.xlim(0.8, len(map_d) + 0.2)
        plt.ylim(68,70.5)
        plt.savefig('ablation_fig/bar_map_M2D.pdf')

    if len(map_m)>0:
        plt.figure()
        plt.plot(np.arange(1, len(map_m) + 1).astype('int'), np.array(map_m), label=r'D $\rightarrow$ M', marker='o')
        plt.xlabel('number of levels')
        plt.legend()
        x_major_locator = MultipleLocator(1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        #plt.xlim(0.8, len(map_m) + 0.2)
        plt.savefig('ablation_fig/6.1map_D2M.pdf')


def draw_iter_pic_single_bar_subplot():
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    map_d=[68.9,69.8,70.1,70.0,70.0]
    map_m=[74.6,79.3,80.0,80.2,80.1]

    total_width, n = 0.6, 2
    width = total_width / n
    x = np.arange(1, len(map_d) + 1)
    x = x - (total_width - width) / 2

    assert len(map_d)==len(map_m)

    plt.figure(figsize=(25, 10))
    ####market#######
    plt.subplot(121)
    plt.tick_params(labelsize=45)
    plt.bar(x + width, np.array(map_m), color='lightblue')
    plt.title(r'D $\rightarrow$ M',fontdict={'weight':'normal','size': 50})
    plt.xlabel('Total number of levels', fontdict={'weight': 'normal', 'size': 45})
    plt.ylabel('mAP',fontdict={'weight': 'normal', 'size': 45})
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(73,81)

    #########dukemtmc#######
    plt.subplot(122)
    plt.tick_params(labelsize=45)
    plt.bar(x,np.array(map_d),label=r'M $\rightarrow$ D',color='lightblue')
    plt.title(r'M $\rightarrow$ D', fontdict={'weight': 'normal', 'size': 50})
    plt.ylabel('mAP', fontdict={'weight': 'normal', 'size': 45})
    plt.xlabel('Total number of levels', fontdict={'weight': 'normal', 'size': 45})
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(67, 71)

    plt.savefig('ablation_fig/bar_map_all.pdf',bbox_inches='tight')
    print('save fig')

def draw_r_p_pic(rec,pre):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator

    plt.figure()
    plt.plot(np.arange(1,len(rec)+1).astype('int'),np.array(rec),label='recall',marker='o')
    plt.plot(np.arange(1,len(pre)+1).astype('int'),np.array(pre),label='precision',marker='o')
    plt.xlabel('Hierachy')
    if dataset_name.startswith('d'):
        plt.title(r'M $\rightarrow$ D')
    else:
        plt.title(r'D $\rightarrow$ M')
    plt.legend()
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(0, 1.0)
    plt.xlim(0.8, len(rec)+0.2)
    plt.savefig('ablation_fig/{}.pdf'.format(dataset_name))
    print('save fig')

def draw_r_p_pic_bar(rec,pre):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator

    plt.figure()
    total_width, n = 0.6, 2
    width = total_width / n
    x=np.arange(1,len(rec)+1)
    x = x - (total_width - width) / 2
    #import pdb;pdb.set_trace()

    plt.bar(x,np.array(rec),width,label='recall')
    plt.bar(x+width,np.array(pre),width,label='precision')
    plt.xlabel('Hierachy,')
    if dataset_name.startswith('d'):
        plt.title(r'M $\rightarrow$ D',fontdict={'weight':'normal','size': 20})
    else:
        plt.title(r'D $\rightarrow$ M',fontdict={'weight':'normal','size': 20})

    plt.legend(loc='lower right')
    #x_major_locator = MultipleLocator(1)
    #ax = plt.gca()
    #ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(0, 1.0)
    #plt.xlim(0.8, len(rec)+0.2)
    plt.savefig('ablation_fig/bar_{}.pdf'.format(dataset_name))
    print('save fig')

def draw_r_p_pic_bar_subplot():
    dataset_name_1 = 'market1501'
    label_dir_1 = 'ablation_logs/21_306_final_iter_5_uda_trainer_update_method_4_new_method17_con_30_thre0.45_add_split_self_split_method10_alpha_0.99_1.0_nei_64_0.25_0.25_0.25_gap_1_split_8_d_to_m'
    dataset_name_2 = 'dukemtmc'
    label_dir_2 = 'ablation_logs/21_303_final_iter_5_uda_trainer_update_method_4_new_method15_con_64_thre0.45_add_split_self_split_method10_alpha_0.5_wo_outlier_0_1.0_nei_64_0.25_0.25_0.25_split_8_m_to_duke'

    ######market########
    all_label = glob.glob(os.path.join(label_dir_1, 'label_*.pth.tar'))
    all_label.sort()
    gt_label_tar='true_lab/{}.pth.tar'.format(dataset_name_1)
    gt_label_npy='true_lab/{}.npy'.format(dataset_name_1)
    gt_label=torch.load(gt_label_tar).tolist()
    labels = []
    pre_1, rec_1 = [], []
    for idx,ii in enumerate(all_label):
        layer_lab = torch.load(ii).tolist()
        labels.append(layer_lab)

    for idx,ii in enumerate(all_label):
        print('idx:',idx)
        #import pdb;pdb.set_trace()
        assert ii.split('/')[-1].startswith('label_{}'.format(idx))
        layer_lab=torch.load(ii).tolist()
        rec_i, pre_i =acc(gt_label,layer_lab)
        pre_1.append(pre_i)
        rec_1.append(rec_i)
    fscore_1=[0.57,0.63,0.68,0.73,0.76]

    ########dukemtmc##########
    all_label = glob.glob(os.path.join(label_dir_2, 'label_*.pth.tar'))
    all_label.sort()
    gt_label_tar = 'true_lab/{}.pth.tar'.format(dataset_name_2)
    gt_label = torch.load(gt_label_tar).tolist()
    labels = []
    pre_2, rec_2 = [], []
    for idx, ii in enumerate(all_label):
        layer_lab = torch.load(ii).tolist()
        labels.append(layer_lab)

    for idx, ii in enumerate(all_label):
        print('idx:', idx)
        # import pdb;pdb.set_trace()
        assert ii.split('/')[-1].startswith('label_{}'.format(idx))
        layer_lab = torch.load(ii).tolist()
        rec_i, pre_i = acc(gt_label, layer_lab)
        pre_2.append(pre_i)
        rec_2.append(rec_i)
    fscore_2=[0.58,0.62,0.66,0.71,0.75]

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator

    plt.figure(figsize=(25,10))
    total_width, n = 0.8, 3
    width = total_width / n
    x=np.arange(1,len(rec_1)+1)
    x = x - (total_width - width) / 2

    ####market#######
    plt.subplot(121)
    plt.bar(x,np.array(rec_1),width,label='recall',color='lightblue')
    plt.bar(x+width,np.array(pre_1),width,label='precision',color='paleturquoise')
    plt.bar(x+2*width,np.array(fscore_1),width,label='F-score',color='mistyrose')
    plt.xlabel('Level index',fontdict={'weight':'normal','size': 45})
    plt.tick_params(labelsize=45)
    if dataset_name_1.startswith('d'):
        plt.title(r'M $\rightarrow$ D',fontdict={'family' : 'Times New Roman','weight':'normal','size': 50})
    else:
        plt.title(r'D $\rightarrow$ M',fontdict={'family' : 'Times New Roman','weight':'normal','size': 50})
    plt.legend(loc='lower right', prop={'family' : 'Times New Roman','weight': 'normal', 'size': 35})
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(0, 1.0)
    #####dukemtmc#######
    plt.subplot(122)
    plt.bar(x, np.array(rec_2), width, label='recall',color='lightblue')
    plt.bar(x + width, np.array(pre_2), width, label='precision',color='paleturquoise')
    plt.bar(x + 2 * width, np.array(fscore_2), width, label='F-score', color='mistyrose')
    plt.xlabel('Level index', fontdict={'family' : 'Times New Roman','weight': 'normal', 'size': 45})
    plt.tick_params(labelsize=45)
    if dataset_name_2.startswith('d'):
        plt.title(r'M $\rightarrow$ D', fontdict={'family' : 'Times New Roman','weight': 'normal', 'size': 50})
    else:
        plt.title(r'D $\rightarrow$ M', fontdict={'family' : 'Times New Roman','weight': 'normal', 'size': 50})
    plt.legend(loc='lower right', prop={'family' : 'Times New Roman','weight': 'normal', 'size': 35})


    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(0, 1.0)
    #plt.xlim(0.8, len(rec)+0.2)
    plt.savefig('ablation_fig/bar_all_rp.pdf', bbox_inches='tight')
    print('save fig')

def vis(pred_label,index=0):#select one to visualize
    pred_map=defaultdict(list)
    for idx, pid in enumerate(pred_label):
        pred_map[pid].append(idx)
    #print('len:',len(pred_map[pred_label[index]]))
    return pred_map[pred_label[index]]


if __name__=='__main__':
    #draw_iter_pic_single_bar_subplot()
    #draw_r_p_pic_bar_subplot()
    #draw_r_p_pic_bar_subplot()
    label_dir_1='ablation_logs/21_306_final_iter_5_uda_trainer_update_method_4_new_method17_con_30_thre0.45_add_split_self_split_method10_alpha_0.99_1.0_nei_64_0.25_0.25_0.25_gap_1_split_8_d_to_m'
    label_dir_2 = 'ablation_logs/21_303_final_iter_5_uda_trainer_update_method_4_new_method15_con_64_thre0.45_add_split_self_split_method10_alpha_0.5_wo_outlier_0_1.0_nei_64_0.25_0.25_0.25_split_8_m_to_duke'

    label_dir=label_dir_2
    dataset_name = 'market1501' if label_dir==label_dir_1 else 'dukemtmc'
    all_label=glob.glob(os.path.join(label_dir,'label_*.pth.tar'))
    all_label.sort()
    #
    #gt_label_tar='true_lab/market1501.pth.tar'
    gt_label_tar='true_lab/{}.pth.tar'.format(dataset_name)
    gt_label_npy='true_lab/{}.npy'.format(dataset_name)
    gt_label=torch.load(gt_label_tar).tolist()
    gt_label_img=np.load(gt_label_npy)
    # #import pdb;pdb.set_trace()

    labels=[]
    pre,rec=[],[]

    #vis
    os.system('rm -r -f tmp')
    os.system('mkdir tmp;cd tmp;mkdir {};cd ..'.format(dataset_name))
    for idx,ii in enumerate(all_label):
        layer_lab = torch.load(ii).tolist()
        labels.append(layer_lab)

    #select index
    pic_index=[]
    for pic_i in range(0,2000):
        vis_idx=[]
        cnts=0
        for idx, ii in enumerate(all_label):
            vis_idx.append(vis(labels[idx],pic_i))
            if idx>0:
                keep_idx=list(set(vis_idx[-1])-set(vis_idx[-2]))
            else:
                keep_idx=vis_idx[-1]

            if len(keep_idx)==0 or len(keep_idx)>10:
                continue
            else:
                cnts+=1
        if cnts>=3:
            print(pic_i)
            pic_index.append(pic_i)
    assert len(pic_index)>0
    #pic_i=392
    #print(gt_label_img[pic_i])
    print('---------')
    #import pdb;pdb.set_trace()
    # for pic_i in range(0,200):
    #     pic_index.append(pic_i)
    for cnts,pic_i in enumerate(pic_index):
        vis_idx=[]
        print('{}/{}'.format(cnts,len(pic_index)))
        for idx,ii in enumerate(all_label):
            #print('idx:',idx)
            vis_idx.append(vis(labels[idx], pic_i))
            os.system('mkdir -p tmp/{}/{}/level_{}'.format(dataset_name,pic_i,idx+1))
            if idx > 0:
                keep_idx = list(set(vis_idx[-1]) - set(vis_idx[-2]))
            else:
                keep_idx = vis_idx[-1]
            for kep in keep_idx:
                #print(gt_label_img[kep])
                os.system('cp {} tmp/{}/{}/level_{}/'.format(gt_label_img[kep],dataset_name,pic_i,idx+1))
        #
    # # #compute rec & pre
    # # for idx,ii in enumerate(all_label):
    # #     print('idx:',idx)
    # #     #import pdb;pdb.set_trace()
    # #     assert ii.split('/')[-1].startswith('label_{}'.format(idx))
    # #     layer_lab=torch.load(ii).tolist()
    # #     rec_i, pre_i =acc(gt_label,layer_lab)
    # #     pre.append(pre_i)
    # #     rec.append(rec_i)
    # #
    # # #draw_r_p_pic(rec,pre)
    # # draw_r_p_pic_bar(rec,pre)
    #
    # draw_iter_pic_single_bar()
    print('Done')
