"""to analyse the cmc results from different models, by chencp"""
import os
import numpy as np
import shutil

def analyse():
    query_file = '/home/tusimple/ccp/dataset/binary-annotation-market1501/lst_dir/query_fuse.lst'
    img_root = '/home/tusimple/ccp/dataset/Market-1501-v15.09.15/query'
    cmc_file_baseline = 'rgb_baseline-v2-cmc.npy'
    cmc_file_soft_mask = 'rgbm_soft_mask_0823-v4-cmc.npy'
    cmc_baseline_top1 = np.load(cmc_file_baseline)[:, 0]
    cmc_soft_mask_top1 = np.load(cmc_file_soft_mask)[:, 0]

    with open(query_file, 'r') as f:
        img_list = f.readlines()

    # both false
    ind = np.where(2*cmc_baseline_top1-cmc_soft_mask_top1==0)[0]
    save_dir = 'img_both_false'
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    for i in ind:
        img_name = os.path.join(img_root, img_list[i][:-1]+'.jpg')
        assert os.path.exists(img_name), 'image not exists, not the path'
        shutil.copy(img_name, save_dir)

    # true vs false
    ind = np.where(cmc_baseline_top1-cmc_soft_mask_top1==1)[0]
    save_dir = 'img_true_false'
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    for i in ind:
        img_name = os.path.join(img_root, img_list[i][:-1]+'.jpg')
        assert os.path.exists(img_name), 'image not exists, not the path'
        shutil.copy(img_name, save_dir)

    # false vs true
    ind = np.where(cmc_baseline_top1-cmc_soft_mask_top1==-1)[0]
    save_dir = 'img_false_true'
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    for i in ind:
        img_name = os.path.join(img_root, img_list[i][:-1]+'.jpg')
        assert os.path.exists(img_name), 'image not exists, not the path'
        shutil.copy(img_name, save_dir)

    # both true
    ind = np.where(cmc_baseline_top1*cmc_soft_mask_top1==1)[0]
    save_dir = 'img_both_true'
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    for i in ind:
        img_name = os.path.join(img_root, img_list[i][:-1]+'.jpg')
        assert os.path.exists(img_name), 'image not exists, not the path'
        shutil.copy(img_name, save_dir)


if __name__ == '__main__':
    analyse()

        
