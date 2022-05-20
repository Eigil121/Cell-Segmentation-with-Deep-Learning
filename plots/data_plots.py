import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from visualize import load_image_np, display_instances
import os

script_dir = os.getcwd()

# Updating the dictionary changes which figures are plotted when
# the script is run
to_plot = {"sample_RGB": False, "quality": False, "mask_sample": False}

# Turned on to save plots as .png images
save_plot = True

if to_plot["sample_RGB"]:
    R_actin = Image.open("images/sample_R.tif")
    R_actin = np.array(R_actin)

    G_tubulin = Image.open("images/sample_G.tif")
    G_tubulin = np.array(G_tubulin)

    B_DNA = Image.open("images/sample_B.tif")
    B_DNA = np.array(B_DNA)

    RGB_image = Image.open("images/sample_RGB.jpg")
    RGB_image = np.array(RGB_image)

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    axs[0,0].imshow(R_actin, cmap='gray')
    axs[0,0].axis('off')
    axs[0, 0].set_title("F-actin")

    axs[0, 1].imshow(G_tubulin, cmap='gray')
    axs[0, 1].axis('off')
    axs[0, 1].set_title("B-tubulin")

    axs[1, 0].imshow(B_DNA, cmap='gray')
    axs[1, 0].axis('off')
    axs[1, 0].set_title("DNA")

    axs[1, 1].imshow(RGB_image)
    axs[1, 1].axis('off')
    axs[1, 1].set_title("Full color RGB")

    if save_plot:
        plt.savefig("finished plots/sample_RGB")

    plt.show()



if to_plot["quality"]:
    color_1 = Image.open("images/color_1.jpg")
    color_1 = np.array(color_1)

    color_2 = Image.open("images/color_2.jpg")
    color_2 = np.array(color_2)

    color_3 = Image.open("images/color_3.jpg")
    color_3 = np.array(color_3)


    defect_1 = Image.open("images/defect_1.jpg")
    defect_1 = np.array(defect_1)

    defect_2 = Image.open("images/defect_2.jpg")
    defect_2 = np.array(defect_2)

    defect_3 = Image.open("images/defect_3.jpg")
    defect_3 = np.array(defect_3)


    other_1 = Image.open("images/other_1.jpg")
    other_1 = np.array(other_1)

    other_2 = Image.open("images/other_2.jpg")
    other_2 = np.array(other_2)

    other_3 = Image.open("images/other_3.jpg")
    other_3 = np.array(other_3)

    fig, big_axes = plt.subplots(figsize=(15.0*1.25, 15.0), nrows=3, ncols=1, sharey=True)

    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title(["Color", "Defects", "Other"][row-1], fontsize=30)

        # Turn off axis lines and ticks of the big subplot
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    for i in range(1, 10):
        ax = fig.add_subplot(3, 3, i)
        ax.imshow([color_1, color_2, color_3, defect_1, defect_2, defect_3, other_1, other_2, other_3][i-1])
        ax.axis("off")
        ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        ax._frameon = False

    fig.set_facecolor('w')
    plt.tight_layout()
    if save_plot:
        plt.savefig("finished plots/quality")
    plt.show()


if to_plot["mask_sample"]:
    os.chdir("../../")
    example_image = "B04_s4_w1EF32DACC-2DC6-4B22-BE5D-43557FD837AD"

    image, masks, bbx = load_image_np("train", example_image, apply_prep=True)

    display_instances(image, bbx, masks, plt_show=True)

    fig, axs = plt.subplots(2, 2)

    axs[0,0].imshow(masks[:,:,10], cmap = "gray")
    axs[0,0].axis("off")
    axs[0, 0].set_title("Single instance mask")

    axs[0, 1].imshow(masks[:, :, [10]]*image)
    axs[0,1].axis("off")
    axs[0, 1].set_title("Single cell")

    axs[1,0].imshow(np.sum(masks, axis=2), cmap = "gray")
    axs[1,0].axis("off")
    axs[1, 0].set_title("All masks")

    axs[1,1].imshow(np.zeros((1024,1280)))
    axs[1,1].axis("off")
    axs[1, 1].set_title("All cells with masks")

    plt.tight_layout()


    os.chdir(script_dir)
    if save_plot:
        plt.savefig("finished plots/mask_example")

    plt.show()








