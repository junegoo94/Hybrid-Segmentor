# This is the code to check if
# 1. .jpg to .png file
# 2. the size of images and their ground truth are the same

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

init_path = os.getcwd()
os.chdir('raw_dataset')
raw_dataset_path = os.getcwd()
dataset_list = os.listdir(os.getcwd())

for i in range(len(dataset_list)):
    print(f'Processing {dataset_list[i]}')
    IMG_folder_path = os.path.join(raw_dataset_path, dataset_list[i], dataset_list[i] + '_IMG')
    GT_folder_path = os.path.join(raw_dataset_path, dataset_list[i], dataset_list[i] + '_GT')
    IMG_file_list = os.listdir(IMG_folder_path)
    GT_file_list = os.listdir(GT_folder_path)
    # check if the number of data for images and GT are the same

    if 'desktop.ini' in IMG_file_list:
        IMG_file_list.remove('desktop.ini')
    if 'desktop.ini' in GT_file_list:
        GT_file_list.remove('desktop.ini')
    if len(IMG_file_list) == len(GT_file_list):
        print(f'The number of data for both IMG and GT are {len(IMG_file_list)}')

    for ind in range(len(IMG_file_list)):
        # check the name of the pairs
        img_file = IMG_file_list[ind]
        gt_file = GT_file_list[ind]
        img_name, img_format = os.path.splitext(img_file)
        gt_name, gt_format = os.path.splitext(gt_file)
        if (gt_name in img_name) or (img_name in gt_name):
            pass
        else:
            print('NOt a Pair! ---> ', img_file, gt_file)

        os.chdir(IMG_folder_path)
        img = cv2.imread(img_file, 1)
        os.chdir(GT_folder_path)
        gt = cv2.imread(gt_file, 0)

        # check the size of the pairs
        if img.shape[0:2] == gt.shape:
            pass
        else:
            print('Different Size! ---> ', img.shape, gt.shape)
            # os.chdir(IMG_folder_path)
            # resized_img = cv2.resize(img, (gt.shape), interpolation=cv2.INTER_NEAREST)      
            # cv2.imwrite(img_file, resized_img)      

        # check data format is all .png if not, modify to .png
        if img_format != '.png':
            os.chdir(IMG_folder_path)
            img = cv2.imread(img_file, 1)
            file_name = img_name + '.png'
            cv2.imwrite(file_name, img)
        if gt_format != '.png':
            os.chdir(GT_folder_path)
            gt = cv2.imread(gt_file, 0)
            file_name = gt_name + '.png'
            print(file_name)
            cv2.imwrite(file_name, gt)
