import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

def loss_r(loss_txt):
    fo=open(loss_txt,'r')
    loss_s=[]
    loss_t=[]
    xiter=[]

    all_iter=0
    cnt=0
    ep=0
    for line in fo.readlines():
        line=line.strip()
        if line.startswith('Epoch'):
            tmp=line.strip().split('\t')
            ep_tmp=tmp[0].split()[-1].split('][')
            ep_0,ep_1,ep_2=int(ep_tmp[0][1:]),int(ep_tmp[1].split('/')[0]),int(ep_tmp[1].split('/')[-1][:-1])
            if ep!= ep_0:
                all_iter+=ep_2
                ep=ep_0
            l_s,l_t=tmp[-2],tmp[-1]
            if not l_s.startswith('Loss_s'):
                import pdb;pdb.set_trace()
            assert l_t.startswith('Loss_t')
            l_s=l_s.split()[1]
            l_t=l_t.split()[1]
            xiter.append(all_iter-ep_2+ep_1)
            loss_s.append(float(l_s))
            loss_t.append(float(l_t))
            cnt+=1
            if cnt%500==0:
                print('cnt:',cnt)
    return loss_t,xiter


if __name__=="__main__":
    loss_txt_1 = './ablation_logs/21_306_final_iter_5_uda_trainer_update_method_4_new_method17_con_30_thre0.45_add_split_self_split_method10_alpha_0.99_1.0_nei_64_0.25_0.25_0.25_gap_1_split_8_d_to_m/log.txt'
    loss_txt_2='./logs/base_d2m/log.txt'

    loss_t_1,xiter_1=loss_r(loss_txt_1)
    loss_t_2,xiter_2=loss_r(loss_txt_2)

    print('draw fig')
    plt.figure()
    plt.plot(np.array(xiter_1)[0:len(xiter_1):10],np.array(loss_t_1)[0:len(loss_t_1):10],label='loss_target_hier')
    plt.plot(np.array(xiter_2)[0:len(xiter_2):10], np.array(loss_t_2)[0:len(loss_t_2):10], label='loss_target_ori')
    #plt.plot(np.array(xiter),np.array(loss_s),label='loss_source')
    plt.title('d2m')
    plt.xlabel('iteration')
    plt.ylim(0,4)
    plt.legend()
    plt.savefig('loss_d2m.png')