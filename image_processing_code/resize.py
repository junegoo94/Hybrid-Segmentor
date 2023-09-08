# Image processing to solve GT distortion

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

init_path = os.getcwd()
os.chdir('raw_dataset')
raw_dataset_path = os.getcwd()
og_dataset_list = os.listdir(os.getcwd())

def mask_cropping(filename, dataset_name, crop_size):
    x, y = 0, 0
    coordinates = []
    run = True
    og_mask = cv2.imread(filename, 0)
    n_selected_images = 0
    name_list = []
    while run:
        cropped_mask = og_mask[y:y+crop_size, x:x+crop_size]
        num_w_pixel = np.sum(cropped_mask == 255)
        if num_w_pixel > 500:
            if (cropped_mask.shape[0] < crop_size) or (cropped_mask.shape[1] < crop_size):
                pass
            else:
                n_selected_images += 1
                to_save_path = os.path.join(raw_dataset_path, 'cropped_'+ dataset_name, 'GT')
                f, format = os.path.splitext(filename)
                new_name = f + '_' + str(n_selected_images) + '.png'
                cv2.imwrite(to_save_path + '/'+ new_name, cropped_mask)
                coordinates.append([y, x])
                name_list.append(new_name)

            x += crop_size
            if x > og_mask.shape[1]:
                x = 0
                y += crop_size
                if y > og_mask.shape[0]:
                    break
        else:
            x += crop_size
            if x > og_mask.shape[1]:
                x = 0
                y += crop_size
                if y > og_mask.shape[0]:
                    break
    # print('# of Selected Cropped Images: ', n_selected_images)
    return coordinates, n_selected_images

def crop_image(cd_dict, dataset_name, crop_size):
    for inx, key in enumerate(cd_dict.keys()):
        to_crop = cv2.imread(key, 1)
        
        for sub_inx, [y, x] in enumerate(cd_dict[key]):
            # print(key)
            # plt.figure()
            cropped_image = to_crop[y:y+crop_size, x:x+crop_size, :]
            # plt.imshow(cropped_image)
            to_save_path = os.path.join(raw_dataset_path, 'cropped_'+ dataset_name, 'IMG')
            f, format = os.path.splitext(key)
            cv2.imwrite(to_save_path+ '/'+ f + '_' + str(sub_inx+1) + format, cropped_image)

os.chdir(raw_dataset_path)
for i in range(len(og_dataset_list)):
    dataset = og_dataset_list[i]
    os.mkdir('cropped_'+dataset)

for i in range(len(og_dataset_list)):
    os.chdir(raw_dataset_path)
    dataset = og_dataset_list[i]
    os.chdir(os.path.join(raw_dataset_path, 'cropped_'+dataset))
    os.mkdir('IMG')
    os.mkdir('GT')

os.chdir(raw_dataset_path)

for dataset_ind in range(len(og_dataset_list)):

    print('Running: ', og_dataset_list[dataset_ind])

    os.chdir(os.path.join(raw_dataset_path, og_dataset_list[dataset_ind], og_dataset_list[dataset_ind] + '_GT'))
    GT_path = os.getcwd()
    os.chdir(os.path.join(raw_dataset_path, og_dataset_list[dataset_ind], og_dataset_list[dataset_ind] + '_IMG'))

    IMG_path = os.getcwd()

    masks_files = os.listdir(GT_path)
    images_files = os.listdir(IMG_path)

    if 'desktop.ini' in images_files:
        images_files.remove('desktop.ini')
    if 'desktop.ini' in masks_files:
        masks_files.remove('desktop.ini')



    for i in range(len(images_files)):
        os.chdir(GT_path)
        GT = cv2.imread(masks_files[i], 0)
        os.chdir(IMG_path)
        img = cv2.imread(images_files[i], 1)
        if GT.shape != img.shape[:2]:
            print(GT.shape ,img.shape)
            print(images_files[i])


    os.chdir(GT_path)
    index_lst = []
    coordinates_dic = {}
    n_masks_dataset = 0
    for i in range(len(masks_files)):
        coordn, n_masks = mask_cropping(masks_files[i], og_dataset_list[dataset_ind], crop_size=256)
        if len(coordn) != 0:
            coordinates_dic[masks_files[i]] = coordn
        n_masks_dataset += n_masks
    print('The number of masks for the dataset ', og_dataset_list[dataset_ind],': ', n_masks_dataset)
    os.chdir(IMG_path)
    crop_image(coordinates_dic, og_dataset_list[dataset_ind], crop_size=256)