import os
import glob
import numpy as np
from PIL import Image
from osgeo import gdal
from tifffile import imwrite

#########################################
#Functions used for dataset construction#
#########################################

def build_npz_masks(mask_directory, npz_name):
    """
    Loads all masks from a directory into an npz structure for easy data handling
    """
    os.chdir(mask_directory)
    npz_structure = {}
    count = 0

    for image in os.listdir():
        npz_structure[image] = []

        for mask in os.listdir(image):
            instance_mask = gdal.Open(image + "/" + mask)
            instance_mask_array = instance_mask.GetRasterBand(1).ReadAsArray()
            npz_structure[image].append(instance_mask_array)
            del instance_mask

        npz_structure[image] = np.array(npz_structure[image]).astype(np.bool)

        count += 1
        print(count, " out of ", len(os.listdir()), " images handled")
    os.chdir("../")
    np.savez(npz_name, **npz_structure)
    return npz_structure

def build_compressed_masks(mask_directory):
    """
    Transforms all tiff masks to binary .npy files only containing the masks.
    Corner coordinates are saved in image file name.
    """
    os.chdir(mask_directory)
    count = 0

    for image in os.listdir():
        os.chdir(image)

        for mask in os.listdir():
            instance_mask = gdal.Open(mask)
            instance_mask_array = instance_mask.GetRasterBand(1).ReadAsArray()

            (ystart, xstart), (ystop, xstop) = np.argwhere(instance_mask_array).min(0), np.argwhere(instance_mask_array).max(0) + 1

            polygon = np.array(instance_mask_array)[ystart:ystop, xstart:xstop].astype(np.bool)

            np.save(mask[:-5] + "_" + str(ystart) + "_" + str(xstart), polygon)

            del instance_mask
            os.remove(mask)

        count += 1
        os.chdir("../")
        print(count, " out of ", len(os.listdir()), " images handled")

    os.chdir("../")
    return

def cellchannel_tiff_merger(image_directory):
    os.chdir(image_directory)
    count = 0
    total = len(glob.glob("*_w1*.tif"))

    for image in glob.glob("*_w1*.tif"):

        DNA = glob.glob(image[:-41] + "1*.tif")[0]
        Tubulin = glob.glob(image[:-41] + "2*.tif")[0]
        Actin = glob.glob(image[:-41] + "4*.tif")[0]

        array_DNA = gdal.Open(DNA)
        array_DNA = array_DNA.GetRasterBand(1).ReadAsArray()

        array_Tubulin = gdal.Open(Tubulin)
        array_Tubulin = array_Tubulin.GetRasterBand(1).ReadAsArray()

        array_Actin = gdal.Open(Actin)
        array_Actin = array_Actin.GetRasterBand(1).ReadAsArray()

        array = np.stack((array_Actin, array_Tubulin, array_DNA), axis=2)

        os.remove(DNA)
        os.remove(Tubulin)
        os.remove(Actin)

        imwrite(image, array, photometric='rgb')

        count += 1
        print(count, " out of ", total, " images handled")

    os.chdir("../")
    return

def cellchannel_tiff_merger_to_jpg(image_directory):
    os.chdir(image_directory)
    count = 0
    total = len(glob.glob("*_w1*.tif"))

    for image in glob.glob("*_w1*.tif"):

        DNA = glob.glob(image[:-41] + "1*.tif")[0]
        Tubulin = glob.glob(image[:-41] + "2*.tif")[0]
        Actin = glob.glob(image[:-41] + "4*.tif")[0]

        array_DNA = gdal.Open(DNA)
        array_DNA = array_DNA.GetRasterBand(1).ReadAsArray()

        array_Tubulin = gdal.Open(Tubulin)
        array_Tubulin = array_Tubulin.GetRasterBand(1).ReadAsArray()

        array_Actin = gdal.Open(Actin)
        array_Actin = array_Actin.GetRasterBand(1).ReadAsArray()


        array_DNA = (array_DNA.astype(int) * 255 / 65535).astype('uint8')
        array_Tubulin = (array_Tubulin.astype(int) * 255 / 65535).astype('uint8')
        array_Actin = (array_Actin.astype(int) * 255 / 65535).astype('uint8')

        array = np.stack((array_Actin, array_Tubulin, array_DNA), axis=2)

        os.remove(DNA)
        os.remove(Tubulin)
        os.remove(Actin)

        #np.save(image[:-4] + ".npy", array)
        im = Image.fromarray(array, mode="RGB")
        im.save(image[:-4] + ".jpg")

        count += 1
        print(count, " out of ", total, " images handled")

    os.chdir("../")
    return

def find_splits(mask, axis=0):
    search_array = np.sum(mask, axis)

    splits = [[0]] if search_array[0] != 0 else []

    empty = False
    if not splits:
        empty = True


    for i, contains_element in enumerate(search_array):
        if (not empty) and (not contains_element):
            empty = True
            splits[-1].append(i)
        elif empty and contains_element:
            empty = False
            splits.append([i])

    if search_array[-1] != 0:
        splits[-1].append(i+1)

    return splits

def Spilt_erroneus_masks(mask_directory):

    num_masks_total = 0
    num_erroneus_masks = 0
    os.chdir(mask_directory)
    count = 0

    for image in os.listdir():
        os.chdir(image)

        for mask in os.listdir():
            num_masks_total += 1
            instance_mask_array = np.load(mask)

            ystart = int(mask.split("_")[-2])
            xstart = int(mask.split("_")[-1][:-4])

            all_new_masks = []

            for split_x in find_splits(instance_mask_array, axis=0):
                for split_y in find_splits(instance_mask_array[:,split_x[0]:split_x[1]], axis=1):
                    new_mask = instance_mask_array[split_y[0]:split_y[1], split_x[0]:split_x[1]]
                    new_mask_ystart = split_y[0] + ystart
                    new_mask_xstart = split_x[0] + xstart
                    all_new_masks.append([new_mask, new_mask_ystart, new_mask_xstart])

            if len(all_new_masks) > 1:
                os.remove(mask)
                for file in all_new_masks:
                    np.save(mask[:60] + "_" + str(file[1]) + "_" + str(file[2]), file[0])

                num_erroneus_masks += 1

            del instance_mask_array


        count += 1
        os.chdir("../")
        print(count, " out of ", len(os.listdir()), " images handled")


    os.chdir("../")
    print(f"There were {num_erroneus_masks} erroneus masks out of {num_masks_total} in the dataset")
    return

def pre_processing(image_directory):
    os.chdir(image_directory)
    count = 0
    total = len(os.listdir())

    for image in os.listdir():
        im = Image.open(image)
        im = np.asarray(im)

        im = (im / np.max(im, axis=(0,1)) * 255).astype("uint8")

        im = Image.fromarray(im)
        im.save(image)

        count += 1
        print(count, " out of ", total, " images handled")

    os.chdir("../")
    return

def remove_from_set(dataset, removals):
    for i, defective_image in enumerate(os.listdir(removals)):
        for mask in os.listdir(dataset + "/masks/" + defective_image[:-4]):
            os.remove(dataset + "/masks/" + defective_image[:-4] + "/" + mask)
        os.remove(dataset + "/images/" + defective_image)

        print(f"{i} of {len(os.listdir(removals))}")

