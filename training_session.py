#########################################
#            SESSION INFO               #
#########################################

class session_info:
    N_EPOCHS = 10
    VERBOSE = True
    CHECKPOINT_FREQUENCY = 10
    MODEL_NAME = "RPN"
    DEVICE = "cuda"
    # If none find most trained or create new if ID not recognized
    MODEL_ID = "TEST1"
    DATASET = "Debug_data"
if __name__ == "__main__":
    from utils import train

    session_info = session_info()
    train(session_info)



