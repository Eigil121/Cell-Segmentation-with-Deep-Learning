#########################################
#            SESSION INFO               #
#########################################

class session_info:
    """Contains settings for training or testing of models
    """
    # General settings
    DATASET = "toy"
    SEED = 100
    DEVICE = "cuda"

    # Training settings
    N_EPOCHS = 10000
    VERBOSE = True
    CHECKPOINT_FREQUENCY = 100
    MODEL_NAME = "RPN_light"
    MODEL_ID = "toy_v5"

    # Test settings
    MODEL1_NAME = "RPN_light"
    MODEL1_ID = "toy_v5"
    MODEL2_NAME = "RPN_split"
    MODEL2_ID = "toy_v3"
    N_ITERATIONS = 100






if __name__ == "__main__":
    #########################################
    #  Run script for testing or training   #
    #########################################
    from utils import train, test

    session_info = session_info()
    train(session_info)
    test(session_info)


