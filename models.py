import torch
import torch.nn as nn
import uuid


class RPN(nn.Module):
    """ torch implimentation of a region proposal network
        expecting a regular RGB-image in the torch.tensor format.
    """

    def __init__(self, configs, model_ID=None):
        super(RPN, self).__init__()
        self.info = {"name": "RPN", "model_ID": model_ID if model_ID else str(uuid.uuid4())[:6], "configs": configs, "epochs": 0, "train_loss": []}

        # Functions used in VGG-16
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.functional.relu

        # Convolutional layers used in VGG-16
        self.conv_1_1 = nn.Conv2d(3, 64, 3, padding='same')
        self.conv_1_2 = nn.Conv2d(64, 64, 3, padding='same')

        self.conv_2_1 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv_2_2 = nn.Conv2d(128, 128, 3, padding='same')

        self.conv_3_1 = nn.Conv2d(128, 256, 3, padding='same')
        self.conv_3_2 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv_3_3 = nn.Conv2d(256, 256, 3, padding='same')

        self.conv_4_1 = nn.Conv2d(256, 512, 3, padding='same')
        self.conv_4_2 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv_4_3 = nn.Conv2d(512, 512, 3, padding='same')

        self.conv_5_1 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv_5_2 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv_5_3 = nn.Conv2d(512, 512, 3, padding='same')

        # Convolution layer for used by objectness classification and region proposal box regression
        self.intermediate = nn.Conv2d(512, 512, 3, padding='same')

        # Convolutional implimentation of objectness score classification
        self.conv_cls = nn.Conv2d(512, configs.N_ANCHORS, 1)

        # Convolutional implimentation of region proposal box regression
        self.conv_reg = nn.Conv2d(512, 4 * configs.N_ANCHORS, 1)


    def forward(self, x):

        # At this point x is the raw input image
        # Below are the first 13 convolutional layers of VGG-16

        x = self.relu(self.conv_1_1(x))
        x = self.relu(self.conv_1_2(x))
        x = self.pool(x)
        x = self.relu(self.conv_2_1(x))
        x = self.relu(self.conv_2_2(x))
        x = self.pool(x)
        x = self.relu(self.conv_3_1(x))
        x = self.relu(self.conv_3_2(x))
        x = self.relu(self.conv_3_3(x))
        x = self.pool(x)
        x = self.relu(self.conv_4_1(x))
        x = self.relu(self.conv_4_2(x))
        x = self.relu(self.conv_4_3(x))
        x = self.pool(x)
        x = self.relu(self.conv_5_1(x))
        x = self.relu(self.conv_5_2(x))
        x = self.relu(self.conv_5_3(x))

        # End of VGG-16 sharable layers
        # At this point x is a large number of feature maps

        x = self.relu(self.intermediate(x))

        # Calculate a 2*N_ANCHORS dimensional vector containing probabilities of each anchor box in each location

        cls = torch.sigmoid(self.conv_cls(x))

        # Calculate a 4*N_ANCHORS dimensional vector containing probabilities of each anchor box in each location

        reg = self.conv_reg(x)

        return cls, reg


