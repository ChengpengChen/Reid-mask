"""list related process, by chencp"""
import shutil
import glob
import os

from utils.misc import load_lst

mask_root = '/home/chencp/dataset/annotation-market1501/'
rgb_root = '/home/chencp/dataset/Market-1501-v15.09.15/'

def gen_lst_sample(new_lst, set='query'):
    """sample line from ori_lst according to new_lst"""

    # dir_list_ori = ['bounding_box_test', 'bounding_box_train', 'query']
    # img_list_ori = []
    # for list_ori in dir_list_ori:
    #     img_list_tmp = glob.glob(os.path.join(rgb_root, list_ori, "*.jpg"))
    #     img_list_ori.extend(img_list_tmp)
    if set == 'query':
        ori_lst = '/home/chencp/data/market-list/query2.lst'
    elif set == 'test':
        ori_lst = '/home/chencp/data/market-list/test2.lst'
    elif set == 'train':
        ori_lst = '/home/chencp/data/market-list/train2.lst'
    else:
        raise AssertionError('query, test or train support')
    with open(ori_lst, 'r') as f:
        img_list_ori = f.readlines()

    new_list_file = new_lst.replace('.lst', '_v2.lst')
    mask_list_file = new_lst.replace('.lst', '_v2-mask.lst')
    fout_new = open(new_list_file, 'w')
    fout_mask = open(mask_list_file, 'w')
    fin = open(new_lst, 'r')

    for line in fin:
        img_full_path_ori = [p for p in img_list_ori if line[:-1] in p]
        assert len(img_full_path_ori) == 1, 'full path match failed'
        fout_new.writelines(img_full_path_ori[0])

        if set == 'query':
            mask_line = img_full_path_ori[0].replace(rgb_root+'query', mask_root+'query_seg')
        elif set == 'test':
            mask_line = img_full_path_ori[0].replace(rgb_root+'bounding_box_test', mask_root+'bounding_box_test_seg')
        elif set == 'train':
            mask_line = img_full_path_ori[0].replace(rgb_root+'bounding_box_train', mask_root+'bounding_box_train_seg')
        else:
            raise AssertionError('query, test or train support')
        mask_line = mask_line.replace('.jpg', '.png')
        fout_mask.writelines(mask_line)
    fout_new.close()
    fout_mask.close()
    fin.close()


def sample_cls():
    """sample query images according to cls of test images with occlusion"""
    import numpy as np

    query_ori_lst = '/home/chencp/data/market-list/query2.lst'
    query_lst_new = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_cls_test.lst'
    query_lst_new_mask = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_cls_test-mask.lst'
    test_fuse_lst = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/test_fuse_v2.lst'

    query_lst = load_lst(query_ori_lst)
    test_lst = load_lst(test_fuse_lst)
    query_cls = np.array([ins.class_id for ins in query_lst])
    test_cls = np.unique([ins.class_id for ins in test_lst])
    print('num of cls in test set with occlusion: {}'.format(len(test_cls)))

    fout = open(query_lst_new, 'w')
    fout_mask = open(query_lst_new_mask, 'w')
    cnt = 0
    for cls in test_cls:
        ind = np.where(query_cls == cls)[0]
        # if np.alen(ind) > 2:
        #     ind = np.random.choice(ind, replace=False, size=2)
        cnt += np.alen(ind)
        for i in ind:
            ins = query_lst[i]
            line = '{}\t{}\t{}\n'.format(ins.index, ins.class_id, ins.img_path)
            fout.writelines(line)
            # mask info
            mask_line = line.replace(rgb_root + 'query', mask_root + 'query_seg')
            mask_line = mask_line.replace('.jpg', '.png')
            fout_mask.writelines(mask_line)
    fout.close()
    fout_mask.close()
    print('sampled {} images, and save in lst file:\n\t{}'.format(cnt, query_lst_new))


def concate_query():
    query_lst_ori_1 = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_fuse_v2.lst'
    query_lst_ori_2 = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_cls_test.lst'
    query_lst_new = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_fuse_v3.lst'
    query_lst_new_mask = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_fuse_v3-mask.lst'
    query_lst = []
    with open(query_lst_ori_1, 'r') as f:
        query_lst.extend(f.readlines())
    with open(query_lst_ori_2, 'r') as f:
        query_lst.extend(f.readlines())
    # remove duplicate
    print('concate lst length: {}'.format(len(query_lst)))
    query_lst = list(set(query_lst))
    print('\tafter removing duplicate: {}'.format(len(query_lst)))

    fout = open(query_lst_new, 'w')
    fout_mask = open(query_lst_new_mask, 'w')
    for line in query_lst:
        fout.writelines(line)
        # mask info
        mask_line = line.replace(rgb_root + 'query', mask_root + 'query_seg')
        mask_line = mask_line.replace('.jpg', '.png')
        fout_mask.writelines(mask_line)
    fout.close()
    fout_mask.close()
    print('concatenated the two query list and save to \n\t{}'.format(query_lst_new))


def sample_clean_img():
    """to sample clean img according images with occlusion in query and test list"""
    query_lst_ori = '/home/chencp/data/market-list/query2.lst'
    test_lst_ori = '/home/chencp/data/market-list/test2.lst'
    query_lst_fuse = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_fuse_v2.lst'
    test_lst_fuse = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/test_fuse_v2.lst'
    query_lst_new = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_fuse_verify.lst'
    test_lst_new = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/test_fuse_verify.lst'
    query_lst_mask_new = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_fuse_verify-mask.lst'
    test_lst_mask_new = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/test_fuse_verify-mask.lst'

    query_lst_ori = load_lst(query_lst_ori)
    test_lst_ori = load_lst(test_lst_ori)
    query_lst_fuse = load_lst(query_lst_fuse)
    test_lst_fuse = load_lst(test_lst_fuse)

    # sample test set according to query
    fout = open(test_lst_new, 'w')
    fout_mask = open(test_lst_mask_new, 'w')
    cnt = 0
    for ins in query_lst_fuse:
        ins_sample = [i for i in test_lst_ori if i.class_id == ins.class_id and i.cam_id != ins.cam_id]
        cnt += len(ins_sample)
        for i_s in ins_sample:
            line = '{}\t{}\t{}\n'.format(i_s.index, i_s.class_id, i_s.img_path)
            fout.writelines(line)
            # mask info
            mask_line = line.replace(rgb_root + 'bounding_box_test', mask_root + 'bounding_box_test_seg')
            mask_line = mask_line.replace('.jpg', '.png')
            fout_mask.writelines(mask_line)
    print('sample {} images from test set according to occlusion query set'.format(cnt))
    print('\tand save the list file to {}'.format(test_lst_new))
    fout.close()
    fout_mask.close()

    # sample query set according to test
    fout = open(query_lst_new, 'w')
    fout_mask = open(query_lst_mask_new, 'w')
    cnt = 0
    for ins in test_lst_fuse:
        ins_sample = [i for i in query_lst_ori if i.class_id == ins.class_id and i.cam_id != ins.cam_id]
        cnt += len(ins_sample)
        for i_s in ins_sample:
            line = '{}\t{}\t{}\n'.format(i_s.index, i_s.class_id, i_s.img_path)
            fout.writelines(line)
            # mask info
            mask_line = line.replace(rgb_root + 'query', mask_root + 'query_seg')
            mask_line = mask_line.replace('.jpg', '.png')
            fout_mask.writelines(mask_line)
    print('sample {} images from query set according to occlusion test set'.format(cnt))
    print('\tand save the list file to {}'.format(query_lst_new))
    fout.close()
    fout_mask.close()


if __name__ == '__main__':
    # new_lst = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/query_fuse.lst'
    # gen_lst_sample(new_lst, set='query')

    # new_lst = '/home/chencp/dataset/binary-annotation-market1501/lst_dir/test_fuse.lst'
    # gen_lst_sample(new_lst, set='test')

    # sample_cls()

    # concate_query()

    sample_clean_img()
