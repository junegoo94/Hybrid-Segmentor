# Rename raw data Files

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
        file_name = dataset_list[i] + '_' + str(ind+1) + '.png'


        if (gt_name in img_name) or (img_name in gt_name):
            os.chdir(IMG_folder_path)
            os.rename(img_file, file_name)
            os.chdir(GT_folder_path)
            os.rename(gt_file, file_name)
        else:
            print('NOt a Pair! ---> ', img_file, gt_file)

