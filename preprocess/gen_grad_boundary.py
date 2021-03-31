import glob
import os
import random
import sys
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from torch.utils.data import Dataset


grad_progress = 0
boundary_progress = 0




def generate_grad(path, image_name, mode, total_num, dst=None):
    # create the output filename
    if dst == None:
        dst = f"../grad/{mode}/{image_name}"

    # do the conversion
    grad_image = cv2.Canny(cv2.imread(f"{path}/{image_name}"), 10, 100)
    cv2.imwrite(dst, grad_image)
    global grad_progress
    grad_progress += 1
    print("\rProgress: {:>3} %".format(grad_progress * 100 / total_num), end=' ')
    sys.stdout.flush()


def generate_boundary(path, image_name, mode, num_classes, ignore_label, total_num):
    # create the output filename
    dst = f"../boundary/{mode}/{image_name}"
    # do the conversion
    semantic_image = cv2.imread(f"{path}/{image_name}", cv2.IMREAD_GRAYSCALE)
    onehot_image = np.array([semantic_image == i for i in range(num_classes)]).astype(np.uint8)
    # change the ignored label to 0
    onehot_image[onehot_image == ignore_label] = 0
    boundary_image = np.zeros(onehot_image.shape[1:])
    # for boundary conditions
    onehot_image = np.pad(onehot_image, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    for i in range(num_classes):
        dist = distance_transform_edt(onehot_image[i, :]) + distance_transform_edt(1.0 - onehot_image[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > 2] = 0
        boundary_image += dist
    boundary_image = (boundary_image > 0).astype(np.uint8)
    cv2.imwrite(dst, boundary_image)
    global boundary_progress
    boundary_progress += 1
    print("\rProgress: {:>3} %".format(boundary_progress * 100 / total_num), end=' ')
    sys.stdout.flush()

if __name__ == "__main__":
    img_path = "../data/vaihingen/train/cropped/images/train"
    mask_path = "../data/vaihingen/train/cropped/masks/train"

    imgs = os.listdir(img_path)
    masks = os.listdir(mask_path)

    # print(masks)
    for img in imgs:
        generate_grad(img_path, img, "train", len(imgs))

    for mask in masks:
        generate_boundary(mask_path, mask, "train", 6, 0, len(masks))


    img_path = "../data/vaihingen/train/cropped/images/val"
    mask_path = "../data/vaihingen/train/cropped/masks/val"

    imgs = os.listdir(img_path)
    masks = os.listdir(mask_path)

    # print(masks)
    for img in imgs:
        generate_grad(img_path, img, "val", len(imgs))

    for mask in masks:
        generate_boundary(mask_path, mask, "val", 6, 0, len(masks))