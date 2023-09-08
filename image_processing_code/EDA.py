# Exploratory Data Analysis of the raw data & preprocessed data
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

init_path = os.getcwd()
os.chdir('final_dataset_v2')
raw_dataset_path = os.getcwd()
dataset_list = os.listdir(os.getcwd())

def crack_pixel_count(dataset):
    os.chdir(os.path.join(raw_dataset_path, dataset, 'GT'))
    dir = os.getcwd()
    gt_lst = os.listdir(dir)
    crack_pixel = 0
    total_pixel = 0
    num_gt = len(gt_lst)
    for file in gt_lst:
        if file == 'desktop.ini':
            num_gt -= 1
            pass
        else:
            mask = cv2.imread(file, 0)
            h, w = mask.shape
            num_w_pixel = np.sum(mask == 255)
            num_b_pixel = np.sum(mask == 0)
            crack_pixel += num_w_pixel
            total_pixel += h*w
    print('The number of crack pixel: ', crack_pixel)
    print('The number of masks: ', num_gt)
    crack_ratio = crack_pixel / total_pixel
    print('The crack ratio: ', crack_ratio*100)

    return crack_pixel, crack_ratio, num_gt, total_pixel

# raw dataset EDA
total_crack_pix = 0
total_pix = 0
total_gts = 0
for i in range(len(dataset_list)):
    print('-'*100)
    print('Dataset: ', dataset_list[i])
    pix_count, _, gt_count, total_pixel = crack_pixel_count(dataset_list[i])
    print('Images Count', gt_count)
    total_crack_pix += pix_count
    total_gts += gt_count
    total_pix += total_pixel
    print()

print()
print('-'*100)
print('Total Crack Pixels:', total_crack_pix)
print('Total number of masks: ', total_gts)
print('Total crack ratio of the masks: ', (total_crack_pix / (total_pix))*100)