from models import RPN, RPN_light, RPN_split
import torch
import glob
from torch.utils.data import DataLoader
from configs import Configs
from toy_dataset import Dataset as toyset
from cell_dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from visualize import display_instances

# instantiate current configs
configs = Configs()

#########################################
#           Loading functions           #
#########################################

def load_model(session_info):
    """ Loads or creates relevant model or optimizer.
    """

    # Handle case where no ID is specified
    if session_info.MODEL_ID == None:
        session_info.MODEL_ID = "*"

    # Store the most relevant checkpoint
    relevant_checkpoint = [0, None]

    # Search for best previous checkpoint
    for checkpoint in glob.glob(f'checkpoints/checkpoint_{session_info.MODEL_NAME}_{session_info.MODEL_ID}_epoch_*.pth'):
        epochs_trained = int(checkpoint.split("_")[-1][:-4])
        if epochs_trained > relevant_checkpoint[0]:
            relevant_checkpoint = [epochs_trained, checkpoint]

    # Load the most relevant checkpoint
    if relevant_checkpoint[1]:
        selected_checkpoint = torch.load(relevant_checkpoint[1])

        if session_info.MODEL_NAME == "RPN":
            model = RPN(selected_checkpoint["model_info"]["configs"])
            model.info = selected_checkpoint["model_info"]
            model.load_state_dict(selected_checkpoint["model_state"])

            optimizer = torch.optim.SGD(model.parameters(), lr=0)
            optimizer.load_state_dict(selected_checkpoint["optim_state"])

        if session_info.MODEL_NAME == "RPN_light":
            model = RPN_light(selected_checkpoint["model_info"]["configs"])
            model.info = selected_checkpoint["model_info"]
            model.load_state_dict(selected_checkpoint["model_state"])

            optimizer = torch.optim.SGD(model.parameters(), lr=0)
            optimizer.load_state_dict(selected_checkpoint["optim_state"])

        if session_info.MODEL_NAME == "RPN_split":
            model = RPN_split(selected_checkpoint["model_info"]["configs"])
            model.info = selected_checkpoint["model_info"]
            model.load_state_dict(selected_checkpoint["model_state"])

            optimizer = torch.optim.SGD(model.parameters(), lr=0)
            optimizer.load_state_dict(selected_checkpoint["optim_state"])

    # A new model is created if no relevant checkpoint was found
    else:
        if session_info.MODEL_NAME == "RPN":
            model = RPN(configs, session_info.MODEL_ID)
            optimizer = torch.optim.SGD(model.parameters(), lr=configs.LEARNING_RATE)

        if session_info.MODEL_NAME == "RPN_light":
            model = RPN_light(configs, session_info.MODEL_ID)
            optimizer = torch.optim.SGD(model.parameters(), lr=configs.LEARNING_RATE)

        if session_info.MODEL_NAME == "RPN_split":
            model = RPN_split(configs, session_info.MODEL_ID)
            optimizer = torch.optim.SGD(model.parameters(), lr=configs.LEARNING_RATE)

    return model, optimizer


def load_data(dataset, configs):
    """Construct the dataloader object with the relevant dataset
    """
    if dataset == "toy":
        dataset = toyset()
    else:
        dataset = Dataset(dataset, configs)
    data_loader = DataLoader(dataset=dataset,
                          batch_size=configs.BATCH_SIZE,
                          shuffle=configs.BATCH_SHUFFLE,
                          num_workers=configs.NUM_DATA_LOADER_CPU)

    return data_loader

#########################################
#           Helper functions            #
#########################################
# Functions implemented to simplify code

def to_Fmap_coordinates(image_coordinate):
    return (image_coordinate - 8) / 16

def to_image_coordinates(Fmap_coordinate):
    return Fmap_coordinate * 16 + 8

def IoU(box_A, box_B):
    """Calculate IoU given two sets of box coordinates (y1, y2, x1, x2)"""

    # Return 0 if boxes are not overlapping
    if box_A[0] > box_B[1] or box_A[1] < box_B[0] or box_A[2] > box_B[3] or box_A[3] < box_B[2]:
        return 0

    # Sort coordinates for calculation of intersection
    sorted_y = sorted([box_A[0], box_A[1], box_B[0], box_B[1]])
    sorted_x = sorted([box_A[2], box_A[3], box_B[2], box_B[3]])

    # Calculate intersection
    intersection = (sorted_y[2] - sorted_y[1]) * (sorted_x[2] - sorted_x[1])

    # Calculate box areas
    area_A = (box_A[1] - box_A[0]) * (box_A[3] - box_A[2])
    area_B = (box_B[1] - box_B[0]) * (box_B[3] - box_B[2])

    # Calculate box union
    union = area_A + area_B - intersection

    # Return intersection over union
    return intersection / union

def NMS(potential_proposals, confidence, IoU_threshold):
    """implementation of non-maximum suppression
    """

    final_proposals = []
    # Find all final proposals
    while potential_proposals:
        max_idx = np.argmax(confidence)
        final_proposals.append(potential_proposals[max_idx])
        potential_proposals.pop(max_idx)
        confidence.pop(max_idx)

        idx_overlaps = []
        for i, box in enumerate(potential_proposals):
            if IoU(box, final_proposals[-1]) >= IoU_threshold:
                idx_overlaps.append(i)

        for idx in sorted(idx_overlaps, reverse=True):
            confidence.pop(idx)
            potential_proposals.pop(idx)

    return final_proposals

def detected(pred_bbx, GT_bbx):
    # Counts number of "correctly" detected instances from a set of region proposals
    detected = 0
    for GT_box in GT_bbx:
        for pred_box in pred_bbx:
            if IoU(GT_box, pred_box) >= 0.95:
                detected += 1
                break

    return detected


#########################################
#           Loss functions              #
#########################################

def RPN_loss(outputs, _, mask_bounding_boxes, configs, training_settings):

    # Initialize arrays for collecting information about every anchor
    # Anchor_IoU stores the maximum IoU scored by each anchor
    anchor_IoU = np.zeros((int(configs.IMAGE_HEIGHT/16), int(configs.IMAGE_WIDTH/16), configs.N_ANCHORS))
    # Anchor_best_GT notes which instance scored the highest IoU with the anchor
    anchor_best_GT = np.full((int(configs.IMAGE_HEIGHT/16), int(configs.IMAGE_WIDTH/16), configs.N_ANCHORS), None)

    # Find the instance most relevant to each anchor without affecting the gradient
    with torch.no_grad():
        # Start from instances and find relevant anchors.
        for GT_number, GT_bounding_box in enumerate(mask_bounding_boxes):
            GT_Height = GT_bounding_box[1].item()-GT_bounding_box[0].item()
            GT_Width = GT_bounding_box[3].item()-GT_bounding_box[2].item()
            GT_Center = ((GT_bounding_box[1].item() + GT_bounding_box[0].item()) / 2, (GT_bounding_box[3].item() + GT_bounding_box[2].item()) / 2)
            for anchor_number, anchor in enumerate(configs.ANCHOR_BOXES):

                # Skip anchor if possible IoU is too low.
                if not 3/10 < 4*anchor[0]*anchor[1]/(GT_Height*GT_Width) < 10/3:
                    break

                # Find relevant anchor box coordinates
                # The if statements help minimize the number of anchors visited
                if (GT_bounding_box[1]-GT_bounding_box[0]) > 2*anchor[0]:
                    min_y_reachable = GT_bounding_box[0] + np.round(GT_Height*0.3) - anchor[0]
                    max_y_reachable = GT_bounding_box[1] - np.round(GT_Height*0.3) + anchor[0]

                else:
                    min_y_reachable = GT_bounding_box[0] + np.round(GT_Width*0.3) - np.round(anchor[0]*0.7)
                    max_y_reachable = GT_bounding_box[1] - np.round(GT_Width*0.3) + np.round(anchor[0]*0.7)


                if (GT_bounding_box[3] - GT_bounding_box[2]) > 2 * anchor[1]:
                    min_x_reachable = GT_bounding_box[2] - anchor[1]
                    max_x_reachable = GT_bounding_box[3] + anchor[1]

                else:
                    min_x_reachable = GT_bounding_box[2] - np.round(anchor[1]*0.7)
                    max_x_reachable = GT_bounding_box[3] + np.round(anchor[1]*0.7)


                min_y_reachable = int(max(np.ceil(to_Fmap_coordinates(min_y_reachable)), 0))
                max_y_reachable = int(min(np.floor(to_Fmap_coordinates(max_y_reachable)) , to_Fmap_coordinates(configs.IMAGE_HEIGHT)))
                min_x_reachable = int(max(np.ceil(to_Fmap_coordinates(min_x_reachable)), 0))
                max_x_reachable = int(min(np.floor(to_Fmap_coordinates(max_x_reachable)), to_Fmap_coordinates(configs.IMAGE_WIDTH)))


                # Anchor box locations in image coordinates
                Anc_x = np.array([[7.5 + 16 * x for x in range(min_x_reachable, max_x_reachable+1)]])
                Anc_y = np.array([[7.5 + 16 * y for y in range(min_y_reachable, max_y_reachable + 1)]])

                # Partial vectorization of ground truth and anchor box intersection calculation
                IoUX = np.vstack((np.floor(Anc_x-anchor[1]), np.floor(Anc_x + anchor[1]), np.ones((1,len(Anc_x[0])))*GT_bounding_box[2].item(), np.ones((1,len(Anc_x[0])))*GT_bounding_box[3].item()))
                IoUY = np.vstack((np.floor(Anc_y - anchor[0]), np.floor(Anc_y + anchor[0]), np.ones((1, len(Anc_y[0]))) * GT_bounding_box[0].item(), np.ones((1, len(Anc_y[0]))) * GT_bounding_box[1].item()))

                IoUX_sort = np.sort(IoUX, axis=0)
                IoUY_sort = np.sort(IoUY, axis=0)

                intersection = (IoUY_sort[[2],:] - IoUY_sort[[1],:]).T @ (IoUX_sort[[2],:] - IoUX_sort[[1],:])

                # IoUs calculated using intersections
                doneIoU = intersection / (4*anchor[0]*anchor[1] + GT_Height*GT_Width - intersection)

                # Calculate new IoU matrix
                IoU_update = np.max(np.vstack((np.array([doneIoU]), np.array([anchor_IoU[min_y_reachable:max_y_reachable+1, min_x_reachable:max_x_reachable+1, anchor_number]]))), axis=0)

                # Find updated indexes
                updated_indexes = np.where(IoU_update != anchor_IoU[min_y_reachable:max_y_reachable+1, min_x_reachable:max_x_reachable+1, anchor_number])

                # Update IoU
                anchor_IoU[min_y_reachable:max_y_reachable + 1, min_x_reachable:max_x_reachable + 1, anchor_number] = IoU_update

                # Only update best ground truth - anchor match for updated anchors
                anchor_best_GT[updated_indexes[0] + min_y_reachable, updated_indexes[1] + min_x_reachable, anchor_number] = GT_number

        # Identify all positive and negative anchors
        negative_idx = np.array(np.where(anchor_IoU <= configs.RPN_ANCHOR_NEGATIVE_THRESHOLD))
        positive_idx = np.array(np.where(anchor_IoU >= configs.RPN_ANCHOR_POSITIVE_THRESHOLD))

        # Return error if no positive instances were found.
        if not len(positive_idx[0]):
            return "ERROR"

        # Select anchor boxes to be used for training
        negative_choice = np.random.permutation(negative_idx.shape[1])[:positive_idx.shape[1]] if positive_idx.shape[1] < 125 else np.random.permutation(negative_idx.shape[1])[:125]
        positive_choice = np.random.permutation(positive_idx.shape[1]) if positive_idx.shape[1] < 125 else np.random.permutation(positive_idx.shape[1])[:125]

        negative_idx = negative_idx[:, negative_choice]
        positive_idx = positive_idx[:, positive_choice]

    # Initialize prediction loss
    log_loss1 = torch.nn.BCELoss()
    log_loss2 = torch.nn.BCELoss()

    # Initialize box regression loss
    smooth_L1_loss1 = torch.nn.SmoothL1Loss()
    smooth_L1_loss2 = torch.nn.SmoothL1Loss()
    smooth_L1_loss3 = torch.nn.SmoothL1Loss()
    smooth_L1_loss4 = torch.nn.SmoothL1Loss()

    # Find model outputs for chosen training instances
    negative_output = outputs[0][0,negative_idx[2,:], negative_idx[0,:], negative_idx[1,:]]
    positive_output = outputs[0][0, positive_idx[2, :], positive_idx[0, :], positive_idx[1, :]]

    # Calculate object detection loss
    cls_loss_negative = log_loss1(negative_output.reshape(-1,1), torch.zeros((positive_idx.shape[1],1)).to(training_settings.DEVICE))
    cls_loss_positive = log_loss2(positive_output.reshape(-1,1), torch.ones((positive_idx.shape[1],1)).to(training_settings.DEVICE))

    # Calculate transformation used for box regression
    Wa = torch.tensor([configs.ANCHOR_BOXES[i][0] for i in positive_idx[2, :]]).to(training_settings.DEVICE)
    Ha = torch.tensor([configs.ANCHOR_BOXES[i][1] for i in positive_idx[2, :]]).to(training_settings.DEVICE)

    Ya = torch.tensor(to_image_coordinates(positive_idx[0, :])).to(training_settings.DEVICE)
    Xa = torch.tensor(to_image_coordinates(positive_idx[1, :])).to(training_settings.DEVICE)

    # Top left corner coordinates
    ty = ((outputs[1][0, positive_idx[2, :] * 4, positive_idx[0, :], positive_idx[1, :]]) / (2 * Ha))
    tx = ((outputs[1][0, positive_idx[2, :] * 4 + 1, positive_idx[0, :], positive_idx[1, :]]) / (2 * Wa))

    th = torch.log(outputs[1][0, positive_idx[2, :] * 4 + 2, positive_idx[0, :], positive_idx[1, :]] / Ha)
    tw = torch.log(outputs[1][0, positive_idx[2, :] * 4 + 3, positive_idx[0, :], positive_idx[1, :]] / Wa)





    # Top left corner coordinates
    positive_boxes_GT_idx = anchor_best_GT[positive_idx[0, :], positive_idx[1, :], positive_idx[2, :]].astype(int)

    # Calculate transformations for prediction targets for box regression
    mask_Ys = torch.tensor([(m[0] + m[1]) / 2 for m in mask_bounding_boxes]).to(training_settings.DEVICE)
    mask_Xs = torch.tensor([(m[2] + m[3]) / 2 for m in mask_bounding_boxes]).to(training_settings.DEVICE)
    mask_Hs = torch.tensor([(m[1] - m[0]) / 2 for m in mask_bounding_boxes]).to(training_settings.DEVICE)
    mask_Ws = torch.tensor([(m[3] - m[2]) / 2 for m in mask_bounding_boxes]).to(training_settings.DEVICE)

    tys = ((mask_Ys[positive_boxes_GT_idx] - Ya) / (2 * Ha))
    txs = ((mask_Xs[positive_boxes_GT_idx] - Xa) / (2 * Wa))

    ths = torch.log(mask_Hs[positive_boxes_GT_idx] / Ha)
    tws = torch.log(mask_Ws[positive_boxes_GT_idx] / Wa)

    # Calculate box regression loss
    reg_loss = smooth_L1_loss1(ty, tys) + smooth_L1_loss2(tx, txs) + smooth_L1_loss3(th, ths) + smooth_L1_loss4(tw, tws)

    # Assemble final loss
    loss = (cls_loss_negative + cls_loss_positive) + reg_loss * configs.RPN_CLS_REG_WEGHTING

    return loss


#########################################
#            Training loop              #
#########################################

def train(training_settings):

    # Set seed for reproducibility
    torch.manual_seed(training_settings.SEED)

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
    if model.info["name"] in ["RPN", "RPN_light", "RPN_split"]:
        criterion = RPN_loss

    for epoch in range(start_epoch, end_epoch):
        for i, (images, masks, bounding_boxes) in enumerate(dataloader):

            # Move training examples to device
            images = images.float().to(training_settings.DEVICE)
            masks = masks.float().to(training_settings.DEVICE)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, masks, bounding_boxes, model.info["configs"], training_settings)

            # Skip given loss error
            if loss == "ERROR":
                continue

            # track loss for every training example
            model.info["train_loss"].append(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if i == 99:
                print("Epoch number:  ", epoch)

        # Print and display training loss to check convergence
        if training_settings.VERBOSE and epoch % 10==0:
            print( f'Epoch [{epoch + 1 - start_epoch}/{training_settings.N_EPOCHS}]')

            plt.plot(model.info['train_loss'][500:])
            plt.show()
            print("current loss: ", np.mean(model.info['train_loss'][-100:]), "  Previous loss: ", np.mean(model.info['train_loss'][-200:-100]), " Improved by: ", 1 - np.mean(model.info['train_loss'][-100:])/np.mean(model.info['train_loss'][-200:-100]))

        # Save checkpoint
        if (epoch + 1) % training_settings.CHECKPOINT_FREQUENCY == 0 or (epoch + 1) == end_epoch:
            #if (1 - np.mean(model.info['train_loss'][-100:])/np.mean(model.info['train_loss'][-200:-100])) < (-0.005):
            #    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2


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


#########################################
#             Experiment                #
#########################################

def test(test_settings):

    # Set seed for reproducibility
    torch.manual_seed(test_settings.SEED)


    # Load models
    test_settings.MODEL_NAME = test_settings.MODEL1_NAME
    test_settings.MODEL_ID = test_settings.MODEL1_ID
    model1, _ = load_model(test_settings)
    model1.eval()

    test_settings.MODEL_NAME = test_settings.MODEL2_NAME
    test_settings.MODEL_ID = test_settings.MODEL2_ID
    model2, _ = load_model(test_settings)
    model2.eval()

    # Load training data
    dataloader = load_data(test_settings.DATASET, model1.info["configs"])

    # Send models to GPU or CPU
    model1.to(test_settings.DEVICE)
    model2.to(test_settings.DEVICE)

    # Select the right loss function
    if model1.info["name"] in ["RPN", "RPN_light", "RPN_split"]:
        criterion = RPN_loss

    # Track loss
    model1_loss = []
    model2_loss = []

    # Track prediction accuracy given 95% threshold
    total_instances = []
    model1_detected = []
    model2_detected = []

    for epoch in range(test_settings.N_ITERATIONS):
        for i, (images, masks, bounding_boxes) in enumerate(dataloader):

            # Move image to device
            images = images.float().to(test_settings.DEVICE)

            # Forward pass model 1
            output1 = model1(images)
            loss1 = criterion(output1, masks, bounding_boxes, model1.info["configs"], test_settings)
            if loss1 == "ERROR":
                continue
            model1_loss.append(loss1.item())

            # Extract proposals
            pos1 = np.where(output1[0].detach().to('cpu').numpy() > 0.5)
            
            ys1 = [output1[1][pos1[0][i], 4 * pos1[1][i] + 0, pos1[2][i], pos1[3][i]].item() + 7.5 + 16 * pos1[2][i] for i in range(len(pos1[0]))]
            xs1 = [output1[1][pos1[0][i], 4 * pos1[1][i] + 1, pos1[2][i], pos1[3][i]].item() + 7.5 + 16 * pos1[3][i] for i in range(len(pos1[0]))]
            hs1 = [output1[1][pos1[0][i], 4 * pos1[1][i] + 2, pos1[2][i], pos1[3][i]].item() for i in range(len(pos1[0]))]
            ws1 = [output1[1][pos1[0][i], 4 * pos1[1][i] + 3, pos1[2][i], pos1[3][i]].item() for i in range(len(pos1[0]))]
            c1 = [output1[0][pos1[0][i], pos1[1][i], pos1[2][i], pos1[3][i]].item() for i in range(len(pos1[0]))]
            bbx1 = [(ys1[i] - hs1[i], ys1[i] + hs1[i], xs1[i] - ws1[i], xs1[i] + ws1[i]) for i in range(len(pos1[0]))]
            del ys1, xs1, hs1, ws1, output1
            bbx1 = NMS(bbx1, c1, 0.4)
            model1_detected.append(detected(bbx1, bounding_boxes))
            del bbx1, c1

            # Forward pass model 2
            output2 = model2(images)
            loss2 = criterion(output2, masks, bounding_boxes, model1.info["configs"], test_settings)

            model2_loss.append(loss2.item())

            # Extract proposals
            pos2 = np.where(output2[0].detach().to('cpu').numpy() > 0.5)

            ys2 = [output2[1][pos2[0][i], 4 * pos2[1][i] + 0, pos2[2][i], pos2[3][i]].item() + 7.5 + 16 * pos2[2][i] for i in range(len(pos2[0]))]
            xs2 = [output2[1][pos2[0][i], 4 * pos2[1][i] + 1, pos2[2][i], pos2[3][i]].item() + 7.5 + 16 * pos2[3][i] for i in range(len(pos2[0]))]
            hs2 = [output2[1][pos2[0][i], 4 * pos2[1][i] + 2, pos2[2][i], pos2[3][i]].item() for i in range(len(pos2[0]))]
            ws2 = [output2[1][pos2[0][i], 4 * pos2[1][i] + 3, pos2[2][i], pos2[3][i]].item() for i in range(len(pos2[0]))]
            c2 = [output2[0][pos2[0][i], pos2[1][i], pos2[2][i], pos2[3][i]].item() for i in range(len(pos2[0]))]

            bbx2 = [(ys2[i] - hs2[i], ys2[i] + hs2[i], xs2[i] - ws2[i], xs2[i] + ws2[i]) for i in range(len(pos2[0]))]
            del ys2, xs2, hs2, ws2, output2
            bbx2 = NMS(bbx2, c2, 0.4)
            model2_detected.append(detected(bbx2, bounding_boxes))
            del bbx2, c2

            total_instances.append(len(bounding_boxes))



        print(epoch)

    p1 = np.array(model1_detected)/np.array(total_instances)
    p2 = np.array(model2_detected) / np.array(total_instances)

    print("Average image accuracy for early integrating model: ", np.mean(p1))
    print("Average image accuracy for late integrating model: ", np.mean(p2))

    print("Overall accuracy for early integrating model: ", np.sum(model1_detected)/np.sum(total_instances))
    print("Overall accuracy for late integrating model: ", np.sum(model2_detected)/np.sum(total_instances))



    # Plot model output examples
    for i, (images, masks, bounding_boxes) in enumerate(dataloader):
        images = images.float().to(test_settings.DEVICE)

        output1 = model1(images)

        pos1 = np.where(output1[0].detach().to('cpu').numpy() > 0.5)

        ys1 = [output1[1][pos1[0][i], 4 * pos1[1][i] + 0, pos1[2][i], pos1[3][i]].item() + 7.5 + 16 * pos1[2][i] for i
               in range(len(pos1[0]))]
        xs1 = [output1[1][pos1[0][i], 4 * pos1[1][i] + 1, pos1[2][i], pos1[3][i]].item() + 7.5 + 16 * pos1[3][i] for i
               in range(len(pos1[0]))]
        hs1 = [output1[1][pos1[0][i], 4 * pos1[1][i] + 2, pos1[2][i], pos1[3][i]].item() for i in range(len(pos1[0]))]
        ws1 = [output1[1][pos1[0][i], 4 * pos1[1][i] + 3, pos1[2][i], pos1[3][i]].item() for i in range(len(pos1[0]))]
        c1 = [output1[0][pos1[0][i], pos1[1][i], pos1[2][i], pos1[3][i]].item() for i in range(len(pos1[0]))]
        bbx1 = [(ys1[i] - hs1[i], ys1[i] + hs1[i], xs1[i] - ws1[i], xs1[i] + ws1[i]) for i in range(len(pos1[0]))]
        bbx1 = NMS(bbx1, c1, 0.4)

        output2 = model2(images)

        pos2 = np.where(output2[0].detach().to('cpu').numpy() > 0.5)


        ys2 = [output2[1][pos2[0][i], 4 * pos2[1][i] + 0, pos2[2][i], pos2[3][i]].item() + 7.5 + 16 * pos2[2][i] for i
               in range(len(pos2[0]))]
        xs2 = [output2[1][pos2[0][i], 4 * pos2[1][i] + 1, pos2[2][i], pos2[3][i]].item() + 7.5 + 16 * pos2[3][i] for i
               in range(len(pos2[0]))]
        hs2 = [output2[1][pos2[0][i], 4 * pos2[1][i] + 2, pos2[2][i], pos2[3][i]].item() for i in range(len(pos2[0]))]
        ws2 = [output2[1][pos2[0][i], 4 * pos2[1][i] + 3, pos2[2][i], pos2[3][i]].item() for i in range(len(pos2[0]))]
        c2 = [output2[0][pos2[0][i], pos2[1][i], pos2[2][i], pos2[3][i]].item() for i in range(len(pos2[0]))]

        bbx2 = [(ys2[i] - hs2[i], ys2[i] + hs2[i], xs2[i] - ws2[i], xs2[i] + ws2[i]) for i in range(len(pos2[0]))]
        del ys2, xs2, hs2, ws2, output2
        bbx2 = NMS(bbx2, c2, 0.4)


        image1 = images[0].to('cpu').detach().permute(1, 2, 0).numpy()
        image2 = images[0].to('cpu').detach().permute(1, 2, 0).numpy()

        display_instances(image1, bbx1, False, show_mask=False)
        display_instances(image2, bbx2, False, show_mask=False)

        if i == 1:
            break

    p1 = np.sum(model1_detected) / np.sum(total_instances)
    p2 = np.sum(model2_detected) / np.sum(total_instances)
    p = (np.sum(model2_detected) + np.sum(model1_detected))/(2 * np.sum(total_instances))


    Zobs = (p1-p2)/np.sqrt(p*(1-p)*(2/(np.sum(total_instances))))

    from scipy import stats
    p_values = stats.norm.sf(abs(Zobs)) * 2

    print("p-value: ", p_values)

    fig, axs = plt.subplots(2)

    axs[0].plot(model2.info['train_loss'][500:179987], label="Early integration")
    axs[0].plot(model1.info['train_loss'][500:], label="Late integration")
    axs[0].legend()

    axs[1].plot(np.convolve(model2.info['train_loss'][500:179987], np.ones(50)/50, mode='valid'))
    axs[1].plot(np.convolve(model1.info['train_loss'][500:], np.ones(50) / 50, mode='valid'))

    plt.show()

