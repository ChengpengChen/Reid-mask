"""mask related process, by chencp"""
import shutil
import glob
import os

from utils.misc import load_lst

mask_root = '/home/chencp/dataset/annotation-market1501/'
rgb_root = '/home/chencp/dataset/Market-1501-v15.09.15/'

def copy_old_mask(save_dir):
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    dir_select = 'select_mask'
    img_list = glob.glob(os.path.join(dir_select, "*.png"))
    dir_root_ori = '/home/chencp/dataset/annotation-market1501'
    dir_list_ori = ['bounding_box_test_seg', 'bounding_box_train_seg', 'query_seg']
    img_list_ori = []
    for list_ori in dir_list_ori:
        img_list_tmp = glob.glob(os.path.join(dir_root_ori, list_ori, "*.png"))
        img_list_ori.extend(img_list_tmp)

    for img in img_list:
        ind = img.find('/')
        img_name = img[ind+1:]
        img_full_path_ori = [p for p in img_list_ori if img_name in p]
        assert len(img_full_path_ori) == 1, 'full path match failed'
        shutil.copy(img_full_path_ori[0], save_dir)


def replace_old_mask():
    dir_select = 'select_mask'
    img_list = glob.glob(os.path.join(dir_select, "*.png"))
    dir_root_ori = '/home/chencp/dataset/annotation-market1501'
    dir_list_ori = ['bounding_box_test_seg', 'bounding_box_train_seg', 'query_seg']
    img_list_ori = []
    for list_ori in dir_list_ori:
        img_list_tmp = glob.glob(os.path.join(dir_root_ori, list_ori, "*.png"))
        img_list_ori.extend(img_list_tmp)

    for img in img_list:
        ind = img.find('/')
        img_name = img[ind+1:]
        img_full_path_ori = [p for p in img_list_ori if img_name in p]
        assert len(img_full_path_ori) == 1, 'full path match failed'
        shutil.copy(img, img_full_path_ori[0])


if __name__ == '__main__':
    pass

