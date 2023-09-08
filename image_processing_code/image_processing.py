# Image processing to solve GT distortion

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

init_path = os.getcwd()
os.chdir('raw_dataset')
raw_dataset_path = os.getcwd()
dataset_list = os.listdir(os.getcwd())

def closing(gt, kernel_size, iteration=1):
    kernel = np.ones((kernel_size, kernel_size),np.uint8)
    closing_results = cv2.morphologyEx(gt, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing_results

def opening(gt, kernel_size, iteration=1):
    kernel = np.ones((kernel_size, kernel_size),np.uint8)
    opening_results = cv2.morphologyEx(gt, cv2.MORPH_OPEN, kernel, iterations=1)
    return opening_results

def erode(gt, kernel_size, iteration=1):
    kernel = np.ones((kernel_size, kernel_size),np.uint8)
    erosion_results = cv2.erode(gt, kernel, iterations = 1)
    return erosion_results

def dilate(gt, kernel_size, iteration=1):
    kernel = np.ones((kernel_size, kernel_size),np.uint8)
    dilation_results = cv2.dilate(gt, kernel, iterations = 1)
    return dilation_results

def centre_crop(img):
    centre_cropped = img[76:436, 76:436]
    return centre_cropped

def upscaling(img):
    upscaled = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
    return upscaled


for dataset in dataset_list:
    os.chdir(os.path.join(raw_dataset_path, dataset, dataset + '_GT'))
    GT_path = os.getcwd()
    GT_lst = os.listdir(GT_path)
    if 'desktop.ini' in GT_lst:
        GT_lst.remove('desktop.ini')
    os.chdir(os.path.join(raw_dataset_path, dataset, dataset + '_IMG'))
    IMG_path = os.getcwd()
    IMG_lst = os.listdir(IMG_path)
    if 'desktop.ini' in IMG_lst:
        IMG_lst.remove('desktop.ini')

    for i in range(len(GT_lst)):
        if GT_lst[i] != IMG_lst[i]:
            print('Not a pair!')
        else:
            os.chdir(os.path.join(raw_dataset_path, dataset, dataset + '_IMG'))
            img = cv2.imread(IMG_lst[i], 1)
            os.chdir(os.path.join(raw_dataset_path, dataset, dataset + '_GT'))
            gt = cv2.imread(GT_lst[i], 0)
            if (dataset == 'AIGLE_RN') or (dataset == 'CrackLS315') or (dataset == 'CRKWH100') or (dataset == 'DeepCrack') or (dataset == 'ct260') or (dataset == 'LCMS') or (dataset == 'Stone331'):
                print('Processing Closing')
                print()
                refined = closing(gt, 3)
                cv2.imwrite(GT_lst[i], refined)
            if (dataset == 'GAPs384'):
                print('Processing Erosion')
                print()
                refined = erode(gt, 2)
                cv2.imwrite(GT_lst[i], refined)
            if (dataset == 'Stone331'):
                print('Processing Centre Cropping')
                print()
                refined_GT = centre_crop(gt)
                cv2.imwrite(GT_lst[i], refined_GT)
                os.chdir(os.path.join(raw_dataset_path, dataset, dataset + '_IMG'))
                refined_IMG = centre_crop(img)
                cv2.imwrite(IMG_lst[i], refined_IMG)
            if (dataset == 'masonry'):
                print('Processing Upscaling')
                print()
                refined_GT = upscaling(gt)
                cv2.imwrite(GT_lst[i], refined_GT)
                os.chdir(os.path.join(raw_dataset_path, dataset, dataset + '_IMG'))
                refined_IMG = upscaling(img)
                cv2.imwrite(IMG_lst[i], refined_IMG)

            # Create a figure with two subplots
            # fig, axes = plt.subplots(1, 2, figsize=(12, 12))
            # axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # axes[0].title.set_text('Image')
            # axes[1].imshow(gt[56:456, 56:456], cmap='gray')
            # axes[1].title.set_text('Ground Truth')
            # axes[1].imshow(refined[56:456, 56:456], cmap='gray')
            # axes[1].title.set_text('Refined Ground Truth')
            # plt.tight_layout()
            # plt.show()