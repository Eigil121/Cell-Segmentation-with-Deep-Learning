import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from configs import Configs
from matplotlib import patches

# This script is taken from https://github.com/matterport/Mask_RCNN and modified for the implementation
def load_image_np(dataset, ID, apply_prep = True, configs = Configs()):

    im = np.asarray(Image.open(f"Data/{dataset}/images/{ID}.jpg"))/255

    if apply_prep:
        # Apply pre-processing
        im = np.array(im) / np.max(im, axis=(0, 1))

    mask_path = f"Data/{dataset}/masks/{ID}"
    count = len(os.listdir(mask_path))
    masks = np.zeros((configs.IMAGE_HEIGHT, configs.IMAGE_WIDTH, count), dtype=np.bool)
    bounding_boxes = []
    for i, mask in enumerate(os.listdir(mask_path)):
        ystart = int(mask.split("_")[-2])
        xstart = int(mask.split("_")[-1][:-4])
        compressed_mask = np.load(mask_path + "/" + mask)

        ystop = int(ystart + compressed_mask.shape[0])
        xstop = int(xstart + compressed_mask.shape[1])

        bounding_boxes.append((ystart, ystop, xstart, xstop))

        masks[ystart: ystart + compressed_mask.shape[0], xstart: xstart + compressed_mask.shape[1], i] = compressed_mask
    return im, masks, bounding_boxes


def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')

        if type(image) == np.ndarray:
            plt.imshow(image, cmap=cmap,
                       norm=norm, interpolation=interpolation)

       # Handles case where image is loaded using data loader
        else:
            plt.imshow((image.permute(1,2,0).numpy()*255).astype(np.uint8), cmap=cmap,
                       norm=norm, interpolation=interpolation)

        i += 1
    plt.show()



def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors



def apply_mask(image, mask, color, alpha=0.2):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image, boxes, masks,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, plt_show=True):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = len(boxes)
    if not N:
        print("\n*** No instances to display *** \n")

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = (image*255).astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        (y1, y2, x1, x2) = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            score = scores[i] if scores is not None else None
            label = "Cell"
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        if show_mask:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)


    ax.imshow(masked_image.astype(np.uint8))
    if plt_show:
        plt.show()







