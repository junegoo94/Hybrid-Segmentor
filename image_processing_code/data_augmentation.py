# Exploratory Data Analysis of the raw data & preprocessed data
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

init_path = os.getcwd()
os.chdir('data_augmentation_result')
raw_dataset_path = os.getcwd()
dataset_list = os.listdir(os.getcwd())

def crack_pixel_count(dataset):
    os.chdir(os.path.join(raw_dataset_path, dataset, 'GT'))
    dir = os.getcwd()
    gt_lst = os.listdir(dir)
    crack_pixel = 0
    total_pixel = 0
    num_gt = len(gt_lst)
    over_5000_crack_file_list = []
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
            if num_w_pixel > 5000:
                over_5000_crack_file_list.append(file)

    print('The number of crack pixel: ', crack_pixel)
    print('The number of masks: ', num_gt)
    crack_ratio = crack_pixel / total_pixel
    print('The crack ratio: ', crack_ratio*100)

    return crack_pixel, crack_ratio, num_gt, total_pixel, over_5000_crack_file_list

# Final dataset EDA
total_crack_pix = 0
total_pix = 0
total_gts = 0
total_over_5000_list = []
for i in range(len(dataset_list)):
    print('-'*100)
    print('Dataset: ', dataset_list[i])
    pix_count, _, gt_count, total_pixel, big_crack_files = crack_pixel_count(dataset_list[i])
    total_over_5000_list.append(big_crack_files)
    print('over 5000', len(big_crack_files))
    total_crack_pix += pix_count
    total_gts += gt_count
    total_pix += total_pixel
    print()

print()
print('-'*100)
print('Total Crack Pixels:', total_crack_pix)
print('Total number of masks: ', total_gts)
print('Total crack ratio of the masks: ', (total_crack_pix / (total_pix))*100)

for i in range(len(total_over_5000_list)):
    if len(total_over_5000_list[i]) == 0:
        pass
    else:
        if i == 10:
            os.chdir(r'C:\Users\juneg\UCL\final_dataset_v2\cropped_masonry\IMG')
            file_list = os.listdir(os.getcwd())
            for j in range(len(file_list)):
                os.chdir(r'C:\Users\juneg\UCL\final_dataset_v2\cropped_masonry\IMG')
                img = cv2.imread(file_list[j], 1)
                os.chdir(r'C:\Users\juneg\UCL\final_dataset_v2\cropped_masonry\GT')
                gt = cv2.imread(file_list[j], 0)
                # Generate random Gaussian noise
                mean = 0
                stddev = 180
                noise = np.zeros(img.shape, np.uint8)
                cv2.randn(noise, mean, stddev)

                # Add noise to image
                noisy_img = cv2.add(img, noise)
                rotation_option = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
                random_choice = random.choice(rotation_option)
                rotate_img = cv2.rotate(noisy_img, random_choice)
                rotate_gt = cv2.rotate(gt, random_choice)

                os.chdir(r'C:\Users\juneg\UCL\data_augmentation_result\IMG')
                cv2.imwrite('augmented_' + file_list[j], rotate_img)
                os.chdir(r'C:\Users\juneg\UCL\data_augmentation_result\GT')
                cv2.imwrite('augmented_' + file_list[j], rotate_gt)


        else:
            for file in total_over_5000_list[i]:
                os.chdir(r'C:\Users\juneg\UCL\combined_dataset_v2\IMG')
                img = cv2.imread(file, 1)
                os.chdir(r'C:\Users\juneg\UCL\combined_dataset_v2\GT')
                gt = cv2.imread(file, 0)
                # Generate random Gaussian noise
                mean = 0
                stddev = 180
                noise = np.zeros(img.shape, np.uint8)
                cv2.randn(noise, mean, stddev)

                # Add noise to image
                noisy_img = cv2.add(img, noise)
                rotation_option = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
                random_choice = random.choice(rotation_option)
                rotate_img = cv2.rotate(noisy_img, random_choice)
                rotate_gt = cv2.rotate(gt, random_choice)


                # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # plt.show()
                # # plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
                # # plt.show()
                # plt.imshow(cv2.cvtColor(rotate_img, cv2.COLOR_BGR2RGB))
                # plt.show()
                # plt.imshow(cv2.cvtColor(rotate_gt, cv2.COLOR_BGR2RGB))
                # plt.show()

                os.chdir(r'C:\Users\juneg\UCL\data_augmentation_result\IMG')
                cv2.imwrite('augmented_' + file, rotate_img)
                os.chdir(r'C:\Users\juneg\UCL\data_augmentation_result\GT')
                cv2.imwrite('augmented_' + file, rotate_gt)

