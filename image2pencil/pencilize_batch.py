# sketchize world image using image process method (world image -> pencil image)
# It provide server version and PC version by the first arg
import cv2 as cv
import matplotlib.pylab as plt
import os
import numpy as np
import argparse


# Crop the foreground object in order to fill the whole image
def max_object(img):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    col_sum = np.sum(img, axis=0)  # column
    row_sum = np.sum(img, axis=1)  # row
    cols = img.shape[1]
    rows = img.shape[0]
    boundary = np.zeros([2, 2], dtype=np.uint16)  # first axis represent row[up, down]

    # left-right crop
    for col_left in range(cols):
        if col_sum[col_left] != 0:
            boundary[1, 0] = col_left
            break
    for col_right in range(cols):
        if col_sum[cols - col_right - 1] != 0:
            boundary[1, 1] = cols - col_right - 1
            break

    # up-down crop
    for row_up in range(rows):
        if row_sum[row_up] != 0:
            boundary[0, 0] = row_up
            break
    for row_down in range(rows):
        if row_sum[rows - row_down - 1] != 0:
            boundary[0, 1] = rows - row_down - 1
            break

    # if the window is too small, abort this image
    flag = False
    small_threshold = 100
    edge = np.zeros([2, ], np.uint16)
    edge[0] = boundary[0, 1] - boundary[0, 0]
    edge[1] = boundary[1, 1] - boundary[1, 0]
    if edge[0] < small_threshold or edge[1] < small_threshold:
        flag = True  # too small flag

    print('  +output image size:[', boundary[0, 1] - boundary[0, 0], ', ', boundary[1, 1] - boundary[1, 0], ']')
    return flag, boundary


def enhance_contrast(img):
    img_norm = cv.normalize(img, dst=None, alpha=150, beta=10, norm_type=cv.NORM_MINMAX)
    cv.imshow('d', img_norm)
    cv.waitKey(0)
    return img_norm


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('--voc_path', type=str, default='F:/IMAGE_DATABASE/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/',
                       help='voc path in your PC, normally end with "../VOC20xx/"')
    parse.add_argument('--image_path', type=str, default='../data/dataset/image/', help='The path to store cropped world images')
    parse.add_argument('--pencil_path', type=str, default='../data/dataset/pencil/', help='The path to store pencil images')
    parse.add_argument('--num', type=int, default=10, help='The number of data you wanna generate')
    parse.add_argument('--kernel', type=int, default=15, help='Gaussian blur kernel size')
    parse.add_argument('--show', type=bool, default=False, help='whether to dynamically show the result')

    args = parse.parse_args()

    # validation check
    assert os.path.exists(args.voc_path), 'ERROR: VOC path is not found, please check!!'
    assert args.num != 0, 'ERROR: Please input the image number you want to generate'
    assert args.kernel >= 3, 'Kernel is too small'
    if not os.path.exists(args.image_path):  os.makedirs(args.image_path)
    if not os.path.exists(args.pencil_path):  os.makedirs(args.pencil_path)

    save_count = 0
    while True:
        # get the path of masks and images
        seg_path = args.voc_path + 'SegmentationClass/'
        img_path = args.voc_path + 'JPEGImages/'
        img_idx = save_count

        print('Processing img:', img_idx)
        path_list = os.listdir(seg_path)
        mask_name = seg_path + path_list[img_idx]
        img_name = img_path + path_list[img_idx].replace('.png', '.jpg')

        # read mask and color images in and do threshold to mask image
        color_img = cv.imread(img_name)
        mask_img = cv.imread(mask_name, 0)
        mask_img[mask_img == 220] = 0  # eliminate white contour line
        _, bin_mask_img = cv.threshold(mask_img, 1, 255,
                                       cv.THRESH_BINARY)  # make multi gray level image to binary image
        _, bin_background_img = cv.threshold(bin_mask_img, 1, 255, cv.THRESH_BINARY_INV)
        bin_mask_img = cv.cvtColor(bin_mask_img, cv.COLOR_GRAY2RGB)  # convert back to 3-channel
        bin_background_img = cv.cvtColor(bin_background_img, cv.COLOR_GRAY2RGB)

        # crop image to maximize object
        too_small, crop_boundary = max_object(bin_mask_img)
        if too_small:
            print('  +info: picture {} is too small, skip'.format(img_idx))
            save_count += 1
            continue
        bin_mask_img = bin_mask_img[crop_boundary[0, 0]:crop_boundary[0, 1], crop_boundary[1, 0]:crop_boundary[1, 1]]
        color_img = color_img[crop_boundary[0, 0]:crop_boundary[0, 1], crop_boundary[1, 0]:crop_boundary[1, 1]]
        bin_background_img = bin_background_img[crop_boundary[0, 0]:crop_boundary[0, 1],
                             crop_boundary[1, 0]:crop_boundary[1, 1]]

        # extract object and paint background with white
        crop_img = cv.bitwise_and(color_img, bin_mask_img)
        crop_img = crop_img + bin_background_img

        # if the width ratio is too big, adjust image size to make it close to a square
        edge = crop_img.shape[:2]
        ratio = edge[0] / edge[1]
        ratio_tol = 1  # ratio tolerance
        if ratio > ratio_tol or ratio < 1 / ratio_tol:
            print('  +info: cropped image has too big width-ratio, re-crop')
            small_edge_idx = 1 if edge[0] > edge[1] else 0
            larger_edge_len = edge[1 - small_edge_idx]
            new_sheet = np.zeros([larger_edge_len, larger_edge_len, 3], np.uint8) + 255

            start_position = round((larger_edge_len - edge[small_edge_idx]) / 2)
            if small_edge_idx == 0:
                new_sheet[start_position:start_position + edge[small_edge_idx], :, :] = crop_img
            else:
                new_sheet[:, start_position:start_position + edge[small_edge_idx], :] = crop_img
            crop_img = new_sheet

        # Save cropped world image
        cv.imwrite(args.image_path + '{:04d}.png'.format(save_count), crop_img)
        if args.show:
            cv.imshow('d', crop_img)
            cv.waitKey(10)

        # pencil transformation
        img = crop_img
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray_inv = cv.bitwise_not(gray)

        kernel_sz = args.kernel
        gray_smooth = cv.GaussianBlur(gray_inv, (kernel_sz, kernel_sz), 0, 0)
        pencil = cv.divide(gray, 255 - gray_smooth, scale=256)

        # Save pencil image
        cv.imwrite(args.pencil_path+'{:04d}.png'.format(save_count), pencil)
        if args.show:
            cv.imshow('d', pencil)
            cv.waitKey(0)

        save_count += 1
        if save_count == args.num:
            print('Pencil generation work done!')
            break
