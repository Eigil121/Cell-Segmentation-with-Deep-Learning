import os
import matplotlib.pyplot as plt
import numpy as np

main_dir = "C:\Skrivebord\Mask-R-CNN-modified-for-cell-segmentation"
os.chdir(main_dir)
from utils import IoU


def count_cells(dataset):
    mask_dir = "Data/" + dataset + "/masks"
    os.chdir(mask_dir)
    distribution = []

    for masks in os.listdir():
        if len(os.listdir(masks)):
            distribution.append(len(os.listdir(masks)))

    os.chdir(main_dir)
    return distribution

def cell_bbx_dimensions(dataset):
    mask_dir = "Data/" + dataset + "/masks"
    os.chdir(mask_dir)
    dim_heatmap = np.zeros((500,500))
    count = 0

    extreme_masks = []

    for image in os.listdir():
        os.chdir(image)

        for mask in os.listdir():
            (h, w) = np.load(mask).shape
            dim_heatmap[h, w] += 1

            if h > 350 or w > 350:
                extreme_masks.append(image)

        count += 1
        os.chdir("../")
        print(count, " out of ", len(os.listdir()), " images handled")
    os.chdir(main_dir)
    return dim_heatmap, list(set(extreme_masks))

def cumsum2d(heatmap, in_percent=True):
    cum_sum_map = np.cumsum(np.cumsum(heatmap, axis=1), axis=0)

    if in_percent:
        return cum_sum_map/np.sum(heatmap)
    else:
        return cum_sum_map

def propose_anchors(anchor_boxes, heatmap, threshold=0.7):
    anchor_reach = np.zeros_like(heatmap)

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            for anchor in anchor_boxes:
                if IoU((0,2*anchor[0], 0, 2*anchor[1]), (0, i, 0, j)) >= threshold:
                    anchor_reach[i,j] = 1

    print(f"{np.sum(anchor_reach*heatmap)*100/np.sum(heatmap)}% of the masks are reachable.")

    return anchor_reach

def propose_anchors2(heatmap, threshold=0.7):

    ret = np.zeros_like(heatmap)

    for ii in range(1,heatmap.shape[0]):
        for jj in range(1,heatmap.shape[1]):
            anchor_reach = np.zeros_like(heatmap)

            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[1]):

                    if IoU((0, ii, 0, jj), (0, i, 0, j)) >= threshold:
                        anchor_reach[i, j] = 1

            ret[ii,jj] = np.sum(anchor_reach*heatmap)/np.sum(heatmap)


        print(f"{ii*heatmap.shape[1]} of {heatmap.shape[0]*heatmap.shape[1]}")

    return ret

def binary_image_to_plotxy(binary_image):
    x, y = [], []
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i,j]:
                x.append(i)
                y.append(j)

    return x, y




