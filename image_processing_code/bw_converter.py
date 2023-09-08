# Invert black and white pixels of AEL GT
# check if GTs of all datasets have only black or white pixels.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

init_path = os.getcwd()
os.chdir('raw_dataset')
raw_dataset_path = os.getcwd()
dataset_list = os.listdir(os.getcwd())

# check if all the pixels are white or black
def check_b_w(mask_filename_list):
    non_bw_list = []
    for i in range(len(mask_filename_list)):
        if i % 10 == 0:
            print(i, end="\r")
        if mask_filename_list[i] == 'desktop.ini':
            pass
        else:
            mask = cv2.imread(mask_filename_list[i], 0)
            num_w_pixel = np.sum(mask == 255)
            num_b_pixel = np.sum(mask == 0)
            if (num_b_pixel+num_w_pixel) == mask.shape[0] * mask.shape[1]:
                pass
            else:
                print('non-black or white pixel exists')
                print(mask_filename_list[i])
                non_bw_list.append(mask_filename_list[i])
                print('The number of non-black or white pixels: ',mask.shape[0] * mask.shape[1]-(num_b_pixel+num_w_pixel))
                print('------------------------------------------------------------')
                print()
    if len(non_bw_list) == 0:
        print('All masks pixels are black or white')

    return non_bw_list

to_be_inverted_dataset = ['AIGLE_RN', 'ESAR', 'LCMS']

# Image Complement for AEL datasets
for dataset in to_be_inverted_dataset:
    os.chdir(os.path.join(raw_dataset_path, dataset, dataset+'_GT'))
    GT_lst = os.listdir(os.getcwd())
    for gt in GT_lst:
        mask = cv2.imread(gt, 0)
        inverted_mask = np.invert(mask)
        cv2.imwrite(gt, inverted_mask)

os.chdir(raw_dataset_path)

# check if all the pixels are white or black
for dataset in dataset_list:
    print('Processing Dataset', dataset)
    os.chdir(os.path.join(raw_dataset_path, dataset, dataset+'_GT'))
    GT_lst = os.listdir(os.getcwd())
    non_bw_GT_list = check_b_w(GT_lst)
    for non_bw_gt in non_bw_GT_list:
        gt = cv2.imread(non_bw_gt, 0)
        h, w = gt.shape
        for i in range(h):
            for j in range(w):
                if (gt[i, j] < 255) and (gt[i, j] > 0):
                    if gt[i, j] < 255/2:
                        gt[i, j] = 0
                    else:
                        gt[i, j] = 255

        cv2.imwrite(non_bw_gt, gt)
