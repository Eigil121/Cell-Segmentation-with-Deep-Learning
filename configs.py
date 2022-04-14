class Configs:
    """Base configuration class for the standard Mask R-CNN
    """

    # Shape of the images in the dataset
    # The model assumes all images in the dataset have the same shape.
    IMAGE_HEIGHT, IMAGE_WIDTH = (1024, 1280)

    # Anchor boxes
    N_ANCHORS = 3

    # Batch size depends on GPU memory. A 12GB GPU can typically handle 2 images per batch
    BATCH_SIZE = 1
    BATCH_SHUFFLE = True

    # Dataset path
    DATA_PATH = "/Data"

    LEARNING_RATE = 0.001

    NUM_DATA_LOADER_CPU = 10