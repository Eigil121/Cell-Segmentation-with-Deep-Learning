from models import RPN
import torch
import glob
from torch.utils.data import DataLoader
from configs import Configs
from cell_dataset import Dataset

# Instanciate current configs
configs = Configs()

#########################################
#           Loading functions           #
#########################################

def load_model(session_info):

    if session_info.MODEL_ID == None:
        session_info.MODEL_ID = "*"

    relevant_checkpoint = [0, None]

    # Search for best previous checkpoint
    for checkpoint in glob.glob(f'checkpoints/checkpoint_{session_info.MODEL_NAME}_{session_info.MODEL_ID}_epoch_*.pth'):
        epochs_trained = int(checkpoint.split("_")[-1][:-4])
        if epochs_trained > relevant_checkpoint[0]:
            relevant_checkpoint = [epochs_trained, checkpoint]

    if relevant_checkpoint[1]:
        selected_checkpoint = torch.load(relevant_checkpoint[1])

        if session_info.MODEL_NAME == "RPN":
            model = RPN(selected_checkpoint["model_info"]["configs"])
            model.info = selected_checkpoint["model_info"]
            model.load_state_dict(selected_checkpoint["model_state"])

            optimizer = torch.optim.SGD(model.parameters(), lr=0)
            optimizer.load_state_dict(selected_checkpoint["optim_state"])

    # A new model is created if no relevant checkpoint was found
    else:
        if session_info.MODEL_NAME == "RPN":
            model = RPN(configs, session_info.MODEL_ID)
            optimizer = torch.optim.SGD(model.parameters(), lr=configs.LEARNING_RATE)

    return model, optimizer


def load_data(dataset, configs):
    dataset = Dataset(dataset, configs)
    data_loader = DataLoader(dataset=dataset,
                          batch_size=configs.BATCH_SIZE,
                          shuffle=configs.BATCH_SHUFFLE,
                          num_workers=configs.NUM_DATA_LOADER_CPU)

    return data_loader

#########################################
#           Loss functions              #
#########################################

def RPN_loss(outputs, masks):
    return torch.sum(outputs[0])
# TODO:CLEAN CHECKPOINTS function

#########################################
#            Training loop              #
#########################################

def train(training_settings):

    # Load optimizer and model
    model, optimizer = load_model(training_settings)
    model.train()

    # Load training data
    dataloader = load_data(training_settings.DATASET, model.info["configs"])

    # The number of epochs the model has been trained for so far and the epoch count the session terminates with.
    start_epoch, end_epoch = model.info["epochs"], model.info["epochs"] + training_settings.N_EPOCHS

    # Send model to GPU or CPU
    model.to(training_settings.DEVICE)

    # Select the right loss function
    if model.info["name"] in ["RPN"]:  # TODO: Add models that use RPN criterion
        criterion = RPN_loss

    for epoch in range(start_epoch, end_epoch):
        for i, (images, masks) in enumerate(dataloader):

            images = images.float().to(training_settings.DEVICE)
            masks = masks.float().to(training_settings.DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # track loss for every training example
            model.info["train_loss"].append(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 2000 == 0 and training_settings.VERBOSE:  # TODO: Do something like % or something
                print(
                    f'Epoch [{epoch + 1 - start_epoch}/{training_settings.N_EPOCHS}], Step [{i + 1}/{len(dataloader.dataset)}], Loss: {loss.item():.4f}')


        if (epoch + 1) % training_settings.CHECKPOINT_FREQUENCY == 0 or (epoch + 1) == end_epoch:
            model.info["epochs"] = epoch + 1
            checkpoint = {
                "model_info": model.info,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict()
            }

            # Name checkpoint file using model type, ID and number of epochs trained for
            CHECKPOINT_FILENAME = f'checkpoint_{model.info["name"]}_{model.info["model_ID"]}_epoch_{str(epoch + 1)}.pth'

            # Save checkpoint file
            torch.save(checkpoint, "checkpoints/" + CHECKPOINT_FILENAME)


