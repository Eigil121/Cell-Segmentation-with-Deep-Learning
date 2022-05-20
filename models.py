import torch
import torch.nn as nn
import uuid

#########################################
#           Standard model              #
#########################################
class RPN(nn.Module):
    """ torch implimentation of a region proposal network
        with a VGG-16 backbone.
        expecting a regular RGB-image in the torch.tensor format.
    """

    def __init__(self, configs, model_ID=None):
        super(RPN, self).__init__()
        self.info = {"name": "RPN", "model_ID": model_ID if model_ID else str(uuid.uuid4())[:6], "configs": configs, "epochs": 0, "train_loss": []}

        # Functions used in VGG-16
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.functional.relu
        self.softplus = nn.Softplus()

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

        # Convolution layer for objectness classification and region proposal box regression
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

        reg[0,[4*k+i for i in [2,3] for k in range(self.info['configs'].N_ANCHORS)]] = self.softplus(reg[0,[4*k+i for i in [2,3] for k in range(self.info['configs'].N_ANCHORS)]])

        return cls, reg


class RPN_light(nn.Module):
    """ A lighter version of the model defined above
        expecting a regular RGB-image in the torch.tensor format.
    """

    def __init__(self, configs, model_ID=None):
        super(RPN_light, self).__init__()
        self.info = {"name": "RPN_light", "model_ID": model_ID if model_ID else str(uuid.uuid4())[:6], "configs": configs, "epochs": 0, "train_loss": []}

        # Functions used in VGG-16
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.functional.relu
        self.softplus = nn.Softplus()

        # Convolutional layers used in VGG-16
        self.conv_1_1 = nn.Conv2d(3, 10, 3, padding='same')

        self.conv_2_1 = nn.Conv2d(10, 60, 3, padding='same')

        self.conv_3_1 = nn.Conv2d(60, 20, 3, padding='same')

        self.conv_4_1 = nn.Conv2d(20, 120, 3, padding='same')
        self.conv_4_2 = nn.Conv2d(120, 120, 3, padding='same')
        self.conv_4_3 = nn.Conv2d(120, 120, 3, padding='same')

        self.conv_5_1 = nn.Conv2d(120, 120, 3, padding='same')

        # Convolution layer for used by objectness classification and region proposal box regression
        self.intermediate = nn.Conv2d(120, 120, 3, padding='same')

        # Convolutional implimentation of objectness score classification
        self.conv_cls = nn.Conv2d(120, configs.N_ANCHORS, 1)

        # Convolutional implimentation of region proposal box regression
        self.conv_reg = nn.Conv2d(120, 4 * configs.N_ANCHORS, 1)


    def forward(self, x):

        # At this point x is the raw input image
        # Below are the first 13 convolutional layers of VGG-16

        x = self.relu(self.conv_1_1(x))
        x = self.pool(x)
        x = self.relu(self.conv_2_1(x))
        x = self.pool(x)
        x = self.relu(self.conv_3_1(x))
        x = self.pool(x)
        x = self.relu(self.conv_4_1(x))
        x = self.relu(self.conv_4_2(x))
        x = self.relu(self.conv_4_3(x))
        x = self.pool(x)
        x = self.relu(self.conv_5_1(x))

        # End of sharable layers
        # At this point x is a large number of feature maps

        x = self.relu(self.intermediate(x))

        # Calculate a 2*N_ANCHORS dimensional vector containing probabilities of each anchor box in each location

        cls = torch.sigmoid(self.conv_cls(x))

        # Calculate a 4*N_ANCHORS dimensional vector containing probabilities of each anchor box in each location

        reg = self.conv_reg(x)

        reg[0,[4*k+i for i in [2,3] for k in range(self.info['configs'].N_ANCHORS)]] = self.softplus(reg[0,[4*k+i for i in [2,3] for k in range(self.info['configs'].N_ANCHORS)]])

        return cls, reg


class RPN_split(nn.Module):
    """ torch implimentation of a region proposal network
        expecting a regular RGB-image in the torch.tensor format.
    """

    def __init__(self, configs, model_ID=None):
        super(RPN_split, self).__init__()
        self.info = {"name": "RPN_split", "model_ID": model_ID if model_ID else str(uuid.uuid4())[:6], "configs": configs, "epochs": 0, "train_loss": []}

        # Functions used in VGG-16
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.functional.relu
        self.softplus = nn.Softplus()

        # Convolutional layers used in VGG-16
        self.conv_1_R = nn.Conv2d(1, 10, 3, padding='same')
        self.conv_1_G = nn.Conv2d(1, 10, 3, padding='same')
        self.conv_1_B = nn.Conv2d(1, 10, 3, padding='same')

        self.conv_2_R = nn.Conv2d(10, 20, 3, padding='same')
        self.conv_2_G = nn.Conv2d(10, 20, 3, padding='same')
        self.conv_2_B = nn.Conv2d(10, 20, 3, padding='same')

        self.conv_3_R = nn.Conv2d(20, 20, 3, padding='same')
        self.conv_3_G = nn.Conv2d(20, 20, 3, padding='same')
        self.conv_3_B = nn.Conv2d(20, 20, 3, padding='same')

        self.conv_4_R = nn.Conv2d(20, 40, 3, padding='same')
        self.conv_4_G = nn.Conv2d(20, 40, 3, padding='same')
        self.conv_4_B = nn.Conv2d(20, 40, 3, padding='same')
        self.conv_4_integrated1 = nn.Conv2d(120, 120, 3, padding='same')
        self.conv_4_integrated2 = nn.Conv2d(120, 120, 3, padding='same')

        self.conv_5_integrated = nn.Conv2d(120, 120, 3, padding='same')

        # Convolution layer for used by objectness classification and region proposal box regression
        self.intermediate = nn.Conv2d(120, 120, 3, padding='same')

        # Convolutional implimentation of objectness score classification
        self.conv_cls = nn.Conv2d(120, configs.N_ANCHORS, 1)

        # Convolutional implimentation of region proposal box regression
        self.conv_reg = nn.Conv2d(120, 4 * configs.N_ANCHORS, 1)


    def forward(self, x):

        # At this point x is the raw input image
        # Below are the first 13 convolutional layers of VGG-16
        R = x[:,[0],:,:]
        G = x[:,[1],:,:]
        B = x[:,[2],:,:]

        R = self.relu(self.conv_1_R(R))
        G = self.relu(self.conv_1_G(G))
        B = self.relu(self.conv_1_B(B))
        R = self.pool(R)
        G = self.pool(G)
        B = self.pool(B)
        R = self.relu(self.conv_2_R(R))
        G = self.relu(self.conv_2_G(G))
        B = self.relu(self.conv_2_B(B))
        R = self.pool(R)
        G = self.pool(G)
        B = self.pool(B)
        R = self.relu(self.conv_3_R(R))
        G = self.relu(self.conv_3_G(G))
        B = self.relu(self.conv_3_B(B))
        R = self.pool(R)
        G = self.pool(G)
        B = self.pool(B)
        R = self.relu(self.conv_4_R(R))
        G = self.relu(self.conv_4_G(G))
        B = self.relu(self.conv_4_B(B))

        # Integrate layers
        x = torch.concat((R,G,B), dim=1)

        x = self.relu(self.conv_4_integrated1(x))
        x = self.relu(self.conv_4_integrated2(x))
        x = self.pool(x)
        x = self.relu(self.conv_5_integrated(x))



        # End of VGG-16 sharable layers
        # At this point x is a large number of feature maps

        x = self.relu(self.intermediate(x))

        # Calculate a 2*N_ANCHORS dimensional vector containing probabilities of each anchor box in each location

        cls = torch.sigmoid(self.conv_cls(x))

        # Calculate a 4*N_ANCHORS dimensional vector containing probabilities of each anchor box in each location

        reg = self.conv_reg(x)

        reg[0,[4*k+i for i in [2,3] for k in range(self.info['configs'].N_ANCHORS)]] = self.softplus(reg[0,[4*k+i for i in [2,3] for k in range(self.info['configs'].N_ANCHORS)]])

        return cls, reg