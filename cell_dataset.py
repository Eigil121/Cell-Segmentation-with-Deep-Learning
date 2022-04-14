from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch

class Dataset(Dataset):

    def __init__(self, dataset, configs):
        self.image_IDs = [img_filename[:-4] for img_filename in os.listdir("Data/" + dataset + "/images")]
        self.dataset_dir = "Data/" + dataset
        self.configs = configs
        self.name = "BBC021"

    def __len__(self):
        return len(self.image_IDs)

    def __getitem__(self, index):
        ID = self.image_IDs[index]
        im = Image.open(self.dataset_dir + f"/images/{ID}.tif")
        image = torch.from_numpy(np.array(im)).permute(2, 0, 1)

        mask_path = self.dataset_dir + f"/masks/{ID}"
        count = len(os.listdir(mask_path))
        masks = np.zeros((self.configs.IMAGE_HEIGHT, self.configs.IMAGE_WIDTH, count), dtype=np.bool)

        for i, mask in enumerate(os.listdir(mask_path)):

            ystart = int(mask.split("_")[-2])
            xstart = int(mask.split("_")[-1][:-4])
            compressed_mask = np.load(mask_path + "/" + mask)

            masks[ystart: ystart + compressed_mask.shape[0], xstart: xstart + compressed_mask.shape[1], i] = compressed_mask

        return image, masks


