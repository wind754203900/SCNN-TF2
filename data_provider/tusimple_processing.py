'''
    Tusimple Datasets Tools
'''

import glob
import json
import os
import os.path as ops
import shutil

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def process_json_file(json_file_path, src_dir,ori_dst_dir, binary_dst_dir, instance_dst_dir):
    """

    :param json_file_path:
    :param src_dir: origin clip file path
    :param ori_dst_dir:
    :param binary_dst_dir:
    :param instance_dst_dir:
    :return:
    """
    assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)

    image_nums = len(os.listdir(ori_dst_dir))

    with open(json_file_path, 'r') as file:
        for line_index, line in enumerate(file):
            info_dict = json.loads(line)

            image_dir = ops.split(info_dict['raw_file'])[0]
            image_dir_split = image_dir.split('/')[1:]
            image_dir_split.append(ops.split(info_dict['raw_file'])[1])
            image_name = '_'.join(image_dir_split)
            image_path = ops.join(src_dir, info_dict['raw_file'])
            assert ops.exists(image_path), '{:s} not exist'.format(image_path)

            h_samples = info_dict['h_samples']
            lanes = info_dict['lanes']

            src_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            dst_binary_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)
            dst_instance_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)

            #   generate label images from json files
            lane_exist_list = []
            for lane_index, lane in enumerate(lanes):
                assert len(h_samples) == len(lane)
                lane_x = []
                lane_y = []
                lane_exist = '1' if len(set(lane))!=1 else '0'
                lane_exist_list.append(lane_exist)
                for index in range(len(lane)):
                    if lane[index] == -2:
                        continue
                    else:
                        ptx = lane[index]
                        pty = h_samples[index]
                        lane_x.append(ptx)
                        lane_y.append(pty)
                if not lane_x:
                    continue
                lane_pts = np.vstack((lane_x, lane_y)).transpose()
                lane_pts = np.array([lane_pts], np.int64)

                # 二值分割用01而不是0、255表示，方便预处理，实例分割用1-N表示N个实例，0表示背景
                cv2.polylines(dst_binary_image, lane_pts, isClosed=False,
                              color=1, thickness=5)
                cv2.polylines(dst_instance_image, lane_pts, isClosed=False,
                              color=(lane_index+1), thickness=5)

            str_lane_exist = ''.join(lane_exist_list)
            image_name_new = '{:s}_{:s}.png'.format('{:d}'.format(line_index + image_nums).zfill(4),
                                                    str_lane_exist.ljust(5,'0'))

            dst_binary_image_path = ops.join(binary_dst_dir, image_name_new)
            dst_instance_image_path = ops.join(instance_dst_dir, image_name_new)
            dst_rgb_image_path = ops.join(ori_dst_dir, image_name_new)

            cv2.imwrite(dst_binary_image_path, dst_binary_image)
            cv2.imwrite(dst_instance_image_path, dst_instance_image)
            cv2.imwrite(dst_rgb_image_path, src_image)

            # print(str)
            # print('Process {:s} success'.format(image_name))

def split_train_and_valid_sample(all_train_path,split_rate=0.15):
    '''
    :param all_train_path:
    :return:
    '''
    root_dir = ops.split(all_train_path)[0]
    train_bin_path=ops.join(root_dir,'train_binary.txt')
    train_ins_path = ops.join(root_dir, 'train_instance.txt')
    valid_bin_path=ops.join(root_dir,'validation_binary.txt')
    valid_ins_path = ops.join(root_dir, 'validation_instance.txt')

    sample_list = []
    with open(all_train_path, 'r') as file:
        for line in file:
            sample_list.append(line)

    # split tools and validation samples
    train_list,valid_list = train_test_split(sample_list, test_size=split_rate,shuffle=True)

    # write training file
    with open(train_bin_path, 'w') as bfile,\
        open(train_ins_path, 'w') as ifile:
            for train_sample in train_list:
                info = train_sample.split(' ')
                bin_info = info.copy()
                bin_info.pop(2)
                bin_info = ' '.join(bin_info)
                bfile.write(bin_info)

                ins_info = info.copy()
                ins_info.pop(1)
                ins_info = ' '.join(ins_info)
                ifile.write(ins_info)

    with open(valid_bin_path, 'w') as bfile,\
        open(valid_ins_path, 'w') as ifile:
            for valid_samples in valid_list:
                info = valid_samples.split(' ')
                bin_info = info.copy()
                bin_info.pop(2)
                bin_info = ' '.join(bin_info)
                bfile.write(bin_info)

                ins_info = info.copy()
                ins_info.pop(1)
                ins_info = ' '.join(ins_info)
                ifile.write(ins_info)

def gen_train_sample(dst_dir, b_gt_image_dir, i_gt_image_dir, image_dir, check_file=False):
    """
    generate sample index file
    :param dst_dir:
    :param b_gt_image_dir:
    :param i_gt_image_dir:
    :param image_dir:
    :return:
    """
    # 生成txt文件，存储ori——img binary——image instance——imge lane——exist信息
    with open('{:s}/training/all_train.txt'.format(dst_dir), 'w') as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith('.png'):
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            instance_gt_image_path = ops.join(i_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            assert ops.exists(image_path), '{:s} not exist'.format(image_path)
            assert ops.exists(instance_gt_image_path), '{:s} not exist'.format(instance_gt_image_path)

            str_lane_exist = image_name.split('_')[1]
            str_lane_exist = str_lane_exist.split('.')[0]
            new_str_lane_exist = [i for i in str_lane_exist]
            new_str_lane_exist = ' '.join(new_str_lane_exist)

            if check_file:
                b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
                i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                if b_gt_image is None or image is None or i_gt_image is None:
                    print('图像对: {:s}损坏'.format(image_name))
                    continue
                else:
                    info = '{:s} {:s} {:s} {:s}'.format(image_path, binary_gt_image_path, instance_gt_image_path,new_str_lane_exist)
                    file.write(info + '\n')
            else:
                info = '{:s} {:s} {:s} {:s}'.format(image_path, binary_gt_image_path, instance_gt_image_path, new_str_lane_exist)
                file.write(info + '\n')
    return

def process_tusimple_dataset(src_dir,dst_dir,split_rate=0.15):
    """
    :param src_dir:
    :param dst_dir:
    :return:
    """
    traing_folder_path = ops.join(dst_dir, 'training')

    os.makedirs(traing_folder_path, exist_ok=True)

    for json_label_path in glob.glob('{:s}/label*.json'.format(src_dir)):
        json_label_name = ops.split(json_label_path)[1]

        shutil.copyfile(json_label_path, ops.join(traing_folder_path, json_label_name))

    gt_image_dir = ops.join(traing_folder_path, 'gt_image')
    gt_binary_dir = ops.join(traing_folder_path, 'gt_binary_image')
    gt_instance_dir = ops.join(traing_folder_path, 'gt_instance_image')

    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(gt_binary_dir, exist_ok=True)
    os.makedirs(gt_instance_dir, exist_ok=True)

    '''Generate binary and instance image'''
    for json_label_path in glob.glob('{:s}/*.json'.format(traing_folder_path)):
        process_json_file(json_label_path, src_dir, gt_image_dir, gt_binary_dir, gt_instance_dir)

    '''Generate tools samples lines in txt'''
    '''Default: Disable check_file will improve performance but cann't guarantee the completeness of images'''
    gen_train_sample(dst_dir, gt_binary_dir, gt_instance_dir, gt_image_dir,check_file=False)

    # split tools and validation samples
    path = ops.join(dst_dir, 'training', 'all_train.txt')
    split_train_and_valid_sample(path, split_rate=split_rate)

    return

def process_test_dataset(test_dir):
    '''
        generate test file list to txt,
        store in {test_dir}/test.txt
    '''
    # modify the correct json file name if your json file name is not 'test_tasks_0627.json'
    json_path = ops.join(test_dir,'test_tasks_0627.json')
    save_anno_path = ops.join(test_dir,'test.txt')
    with open(json_path, 'r') as file,open(save_anno_path,'w') as wf:
        for line in file.readlines():
            json_dict = json.loads(line)
            wf.write(json_dict['raw_file']+'\n')

if __name__ == '__main__':
    '''
        Example:
        数据集目录，内含clips文件和JSON标注文件
    '''
    # src_dir: Tusimple training data floder
    src_dir = '/media/wind/MyFile/tusimple_dataset/train_set'

    # dst_dir: The floder where generated images store
    #          Generated images will be store in '{dst_dir}/training/'
    dst_dir = '/media/wind/MyFile/tusimple_dataset/train_set'

    # process images and split the train and validation data with rate
    process_tusimple_dataset(src_dir,dst_dir,split_rate=0.15)

    #  generate test annotation file
    test_dir = '/media/wind/MyFile/tusimple_dataset/test_set'
    process_test_dataset(test_dir)
