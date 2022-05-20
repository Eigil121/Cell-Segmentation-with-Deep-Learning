from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
from configs import Configs

class Dataset(Dataset):
    """Toy dataset used for the project.
    Generally handy for debugging
    """
    def __init__(self, _=None, configs=Configs()):
        self.image_IDs = list(range(100))
        self.configs = configs
        self.name = "TOYSET"

    def __len__(self):
        return len(self.image_IDs)

    def generate_cell(self):
        """Generates a single toy cell
        """

        # Size settings for the image the cells are drawn on
        image_shape = (151, 151)
        image_center = (image_shape[0] // 2, image_shape[1] // 2)

        # Initialize each color channel
        img_R = np.zeros(image_shape)
        img_G = np.zeros(image_shape)
        img_B = np.zeros(image_shape)

        # Generate color intensities for each color channel
        color_R = min(max(0.1, np.random.normal(0.3, 0.1)), 0.5)
        color_G = min(max(0.3, np.random.normal(0.5, 0.1)), 0.7)
        color_B = min(max(0.3, np.random.normal(0.5, 0.1)), 0.7)

        # Draw tubulin in the green channel
        size_G = (np.random.randint(20, 36), np.random.randint(20, 51))                             # Random axis sizes
        rotation_G = np.random.rand() * 360                                                         # Random rotation
        cv2.ellipse(img_G, image_center, size_G, rotation_G, 0, 360, color=color_G, thickness=-1)   # Draw elipse
        

        # Draw DNA in the blue channel
        cv2.circle(img_B, image_center, np.random.randint(10, np.ceil(min(size_G) * 0.8)), color_B, -1) # Draw circle

        # Draw actin in the red channel
        rotation_angle = rotation_G / 360 * 2 * np.pi                                               # Calculate cell rotation in radians
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])              # Build rotation matrix

        naive_coordinates = np.array([[np.sin(degree) * size_G[0] * 1.2 + np.random.normal(0, size_G[0] * 0.02),
                                       np.cos(degree) * size_G[1] * 1.2 + np.random.normal(0, size_G[0] * 0.02)] for
                                      degree in (np.arange(12) * 2 * np.pi / 12)])                  # Make (0,0) centered, non-rotated polygon coordinates

        fixed_coordinates = np.round(
            naive_coordinates @ rotation_matrix.T + np.array([[image_center[0], image_center[1]]])) # Translate and rotate polygon coordinats

        cv2.fillPoly(img_R, [fixed_coordinates.astype('int32')], color_R) # Draw polygon


        # Combine color channels
        cell_image = np.array([img_R, img_G, img_B]).transpose(1, 2, 0)

        # Crop image to remove zero rows and columns
        cropped_image = cell_image[np.where(np.sum(cell_image, axis=(1, 2)) != 0), :, :][0][:,
                        np.where(np.sum(cell_image, axis=(0, 2)) != 0)][:, 0, :]

        return cropped_image

    def build_cell_image(self, n_cells):
        """ Construct image of cells given a number of cells
            Return image, masks and bounding boxes
        """

        # Instanciate emty image, mask array and bounding box list
        cell_image = np.zeros((self.configs.IMAGE_HEIGHT, self.configs.IMAGE_WIDTH, 3))
        masks = np.zeros((self.configs.IMAGE_HEIGHT, self.configs.IMAGE_WIDTH, n_cells))
        bounding_boxes = []

        # Generate n cells
        for i in range(n_cells):
            # Generate new cell
            cell = self.generate_cell()

            # Generate coordinates of top left corner for generated cell
            y_start = np.random.randint(0, self.configs.IMAGE_HEIGHT - cell.shape[0])
            x_start = np.random.randint(0, self.configs.IMAGE_WIDTH - cell.shape[1])

            # Calculate buttom right cell bounding box coordinates
            y_stop = y_start + cell.shape[0]
            x_stop = x_start + cell.shape[1]

            # Save box coordinates and mask
            masks[y_start:y_stop, x_start:x_stop, i] = np.sum(cell, axis=2) > 0
            bounding_boxes.append((y_start, y_stop, x_start, x_stop))

            # Add cell to image
            cell_image[y_start:y_stop, x_start:x_stop, :] = cell_image[y_start:y_stop, x_start:x_stop, :] + cell

        return cell_image, masks, bounding_boxes

    def __getitem__(self, _=None):
        # Generate cell image
        image, masks, bounding_boxes = self.build_cell_image(np.random.randint(10, 101))

        # Apply pre-processing
        image = image/np.max(image, axis=(0,1))

        # Convert to torch tensor
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image, masks, bounding_boxes


