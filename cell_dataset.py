from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch

class Dataset(Dataset):
    """This is a dataset object that can load training examples from the BBBC021
    dataset using .npy for mask cutouts and jpg for images
    This class should be overwitten if another dataset is used
    """

    def __init__(self, dataset, configs):
        # Store all image IDs
        self.image_IDs = [img_filename[:-4] for img_filename in os.listdir("Data/" + dataset + "/images")]
        # Dataset directory e.g. train, val or test
        self.dataset_dir = "Data/" + dataset
        self.configs = configs
        self.name = "BBC021"

    def __len__(self):
        return len(self.image_IDs)

    def __getitem__(self, index):
        ID = self.image_IDs[index]

        # Load image
        im = Image.open(self.dataset_dir + f"/images/{ID}.jpg")

        # Apply pre-processing
        im = np.array(im) / np.max(np.array(im), axis=(0,1))

        # Convert to torch tensor
        image = torch.from_numpy(im).permute(2, 0, 1)

        # Find masks
        mask_path = self.dataset_dir + f"/masks/{ID}"

        # Count number of masks
        count = len(os.listdir(mask_path))

        # Load empty arrays for masks
        masks = np.zeros((self.configs.IMAGE_HEIGHT, self.configs.IMAGE_WIDTH, count), dtype=np.bool)

        # Initialize bounding box list
        bounding_boxes = []

        # Loop through masks fill the empty masks array and bounding_boxes list
        for i, mask in enumerate(os.listdir(mask_path)):

            # Use file name to find upper left corner coordinate of mask
            ystart = int(mask.split("_")[-2])
            xstart = int(mask.split("_")[-1][:-4])

            # Load compressed mask
            compressed_mask = np.load(mask_path + "/" + mask)

            # Find lower left corner coordinate of mask
            ystop = int(ystart + compressed_mask.shape[0])
            xstop = int(xstart + compressed_mask.shape[1])

            # Store bounding box coordinates
            bounding_boxes.append((ystart, ystop, xstart, xstop))

            # Place mask in masks array
            masks[ystart: ystart + compressed_mask.shape[0], xstart: xstart + compressed_mask.shape[1], i] = compressed_mask

        return image, masks, bounding_boxes


