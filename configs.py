class Configs:
    """Configuration class for the standard Mask R-CNN
        Settings should be updated based on hardware
        capabilities and dataset used
        Some settings should be adjusted either with intuition
        or through trial and error
    """

    # Shape of the images in the dataset
    # The model assumes all images in the dataset have the same shape.
    IMAGE_HEIGHT, IMAGE_WIDTH = (1024, 1280)

    # Anchor boxes used by RPN
    ANCHOR_BOXES = [(24,35), (35, 24), (23, 23), (27, 27), (30, 30), (33,33)]
    N_ANCHORS = len(ANCHOR_BOXES)

    # Positive and negative IoU thresholds for anchor boxes during RPN training
    RPN_ANCHOR_POSITIVE_THRESHOLD = 0.7
    RPN_ANCHOR_NEGATIVE_THRESHOLD = 0.3

    # Maxumum number of positive instances used for training with each image
    RPN_NUM_EXAMPLES = 128

    # Hyper parameter adjusted to roughly balance the RPN loss function
    RPN_CLS_REG_WEGHTING = 50

    # Batch size depends on GPU memory. A 12GB GPU can typically handle 2 images per batch
    BATCH_SIZE = 1

    # Number of processors available for data loading
    NUM_DATA_LOADER_CPU = 5

    # Shuffle the training images for each epoch
    BATCH_SHUFFLE = True

    # Dataset path
    DATA_PATH = "/Data"

    LEARNING_RATE = 0.001



