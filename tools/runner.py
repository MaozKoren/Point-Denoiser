import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from datasets.noise import Noise
import cv2
import numpy as np
from models.Point_MAE import Point_Denoiser
from tools.helper_functions import plot_point_clouds, add_noise_in_sphere, calc_confusion_matrix, plot_classification_logits_heatmap

def perturb_points(point_cloud, N, std=0.04):
    # Ensure that the input tensor is of shape [1, 1024, 3]
    assert point_cloud.shape == (1, 1024, 3), "Input tensor must be of shape [1, 1024, 3]"

    # Flatten the first dimension for easier manipulation
    point_cloud = point_cloud[0]  # Now the shape is [1024, 3]

    # Randomly select N unique indices to perturb
    indices = torch.randperm(point_cloud.size(0))[:N]
    # Create random perturbations with values in the range [-offset, offset]
    perturbations = torch.normal(mean=0, std=std, size=(N, 3), device=point_cloud.device)

    # Apply the perturbations to the selected points
    point_cloud[indices] += perturbations

    # Create a noise indices tensor, initially all zeros
    noise_indices = torch.zeros(point_cloud.shape[0], 1, device=point_cloud.device)
    noise_indices[indices] = 1  # Set the selected points to 1
    # Reshape both tensors to include the batch dimension again
    return point_cloud.unsqueeze(0), noise_indices.unsqueeze(0)

def change_one_point_to_noise(neighborhood, offset=0.15):
    # print(neighborhood.shape)
    batch_size, num_clusters, num_points, num_coords = neighborhood.shape

    # Calculate the maximum x, y, z values within each cluster
    max_values = neighborhood.max(dim=2, keepdim=True).values  # Shape: (batch_size, num_clusters, 1, 3)

    # Generate random offsets slightly larger than the max values
    random_offsets = torch.rand(batch_size, num_clusters, 1, num_coords, device=neighborhood.device) * offset

    # Create the noise point by adding the random offsets to the max values
    noise_points = max_values + random_offsets  # Shape: (batch_size, num_clusters, 1, 3)

    # Generate random indices for each batch and cluster
    random_indices = torch.randint(0, num_points, (batch_size, num_clusters), device=neighborhood.device)

    # Iterate through each batch and cluster to replace points
    for b in range(batch_size):
        for c in range(num_clusters):
            idx = random_indices[b, c]
            neighborhood[b, c, idx, :] = noise_points[b, c, 0, :]

    return neighborhood


def reshape_tokens(input_tensor):
    # Ensure the input tensor has the expected shape
    assert input_tensor.shape == (1, 64, 32, 3), "Input tensor must have shape [1, 64, 32, 3]"

    # Reshape the tensor to [1024, 3] for easier manipulation
    reshaped_tensor = input_tensor.view(6144, 3)

    # Remove duplicate points
    unique_tensor = torch.unique(reshaped_tensor, dim=0)

    # Ensure the output tensor has the correct shape after removing duplicates
    if unique_tensor.shape[0] != 1024:
        # If duplicates were removed, pad the tensor with zeros to maintain the size
        padding_size = 1024 - unique_tensor.shape[0]
        padding_tensor = torch.zeros(padding_size, 3)
        unique_tensor = torch.cat((unique_tensor, padding_tensor), dim=0)

    # Reshape back to [1, 1024, 3]
    output_tensor = unique_tensor.view(1, 1024, 3)

    return output_tensor


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    _, ModelNet40_dataloader = builder.dataset_builder(args, config.dataset.ModelNet)

    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    useTestDatasets = False
    if useTestDatasets:
        combCdl2, combCdl2Std, combAccuracyOutside, combAccuracyStdOutside = createTestDataset(test_dataloader, ModelNet40_dataloader, base_model)
        print(f'mean cdl2 of combined dataset (shapenet, modlenet): {combCdl2}')
        print(f'cdl2 standard deviation of combined dataset (shapenet, modlenet): {combCdl2Std}')
        print(f'mean accuracy of combined dataset (shapenet, modlenet): {combAccuracyOutside}')
        print(f'accuracy standard deviation of combined dataset (shapenet, modlenet): {combAccuracyStdOutside}')
    test(base_model, test_dataloader, args, config, logger=logger)


# Pu-Net (test) Dataset Creation
def createTestDataset(shapenet_dataloader, modelnet_dataloader, base_model):
    print('in createTestDataset, test function')
    base_model.eval()  # set model to eval mode
    npoints = 1024
    classification_acc = []
    classification_acc_outside = []
    confusion = []
    confusion_outside = []
    cdl2_array = []

    with torch.no_grad():
        print('in torch no grad')
        for idx, (taxonomy_id, model_id, (points, label)) in enumerate(modelnet_dataloader):
            print(taxonomy_id[0])
            print(f'ModelNet40 label: {label[0]}')
            points_noise = points.clone().cuda()
            points_noise = misc.fps(points_noise, npoints)
            print(f'ModelNet40 points shape: {points_noise.shape}')
            # points_noise, labels, inside_mask_full = add_noise_in_sphere(points_noise, N=700, check_convex=True)
            points_noise, labels = perturb_points(points_noise, N=256, std=0.01)
            vis_points, centers, acc, TP, TN, FP, FN, cdl2, neighborhood_vis, classification_logits = base_model(pts=points_noise, labels=labels, vis=True)
            # vis_points, centers, acc, acc_outside, TP, TN, FP, FN, cdl2, neighborhood_vis, classification_logits = base_model(pts=points_noise, labels=labels, inside_mask=inside_mask_full, vis=True)
            cdl2_array.append(cdl2)
            # classification_acc_outside.append(acc_outside)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(shapenet_dataloader):
            print(taxonomy_ids[0])
            points_noise = data.clone().cuda()
            # points_noise, labels, inside_mask_full = add_noise_in_sphere(points_noise, N=700, check_convex=True)
            # vis_points, centers, acc, acc_outside, TP, TN, FP, FN, cdl2, neighborhood_vis, classification_logits = base_model(pts=points_noise, labels=labels, inside_mask=inside_mask_full, vis=True)
            points_noise, labels = perturb_points(points_noise, N=256, std=0.01)
            vis_points, centers, acc, TP, TN, FP, FN, cdl2, neighborhood_vis, classification_logits = base_model(pts=points_noise, labels=labels, vis=True)
            # classification_acc_outside.append(acc_outside)
            cdl2_array.append(cdl2)

    combCdl2 = sum(cdl2_array) / len(cdl2_array)
    combCdl2Std = np.std(torch.stack(cdl2_array).cpu().numpy(), ddof=0)
    print(f'mean cdl2 of combined dataset (shapenet, modlenet): {sum(cdl2_array) / len(cdl2_array)}')
    print(f'cdl2 standard deviation of combined dataset (shapenet, modlenet): {np.std(torch.stack(cdl2_array).cpu().numpy(), ddof=0)}')
    combAccuracyOutside = sum(classification_acc_outside) / len(classification_acc_outside)
    combAccuracyStdOutside = np.std(torch.stack(classification_acc_outside).cpu().numpy(), ddof=0)
    print(f'mean cdl2 of combined dataset (shapenet, modlenet): {sum(cdl2_array) / len(cdl2_array)}')
    print(f'cdl2 standard deviation of combined dataset (shapenet, modlenet): {np.std(torch.stack(cdl2_array).cpu().numpy(), ddof=0)}')
    return combCdl2, combCdl2Std, combAccuracyOutside, combAccuracyStdOutside

# visualization
def test(base_model, test_dataloader, args, config, logger = None):
    print('in runner.py, test function')
    base_model.eval()  # set model to eval mode

    target = './vis'
    useful_cate = [
        "02691156", #plane
        "04379243",  #table
        "03790512", #motorbike
        "03948459", #pistol
        "03642806", #laptop
        "03467517",     #guitar
        "03261776", #earphone
        "03001627", #chair
        "02958343", #car
        "04090263", #rifle
        "03759954", # microphone
    ]

    classification_acc = []
    classification_acc_outside = []
    confusion = []
    confusion_outside = []
    cdl2_array = []

    with torch.no_grad():
        print('in torch no grad')
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # print('in test dataloader for loop')
            # import pdb; pdb.set_trace()
            # print(f' taxonomy_ids[0] is: {taxonomy_ids[0]}')
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                if taxonomy_ids[0] not in useful_cate:
                    continue
                if taxonomy_ids[0] == "02691156":
                    a, b= 90, 135
                elif taxonomy_ids[0] == "04379243":
                    a, b = 30, 30
                elif taxonomy_ids[0] == "03642806":
                    a, b = 30, -45
                elif taxonomy_ids[0] == "03467517":
                    a, b = 0, 90
                elif taxonomy_ids[0] == "03261776":
                    a, b = 0, 75
                elif taxonomy_ids[0] == "03001627":
                    a, b = 30, -45
                else:
                    a, b = 0, 0
            elif dataset_name == 'ModelNet':
                a, b = 0, 0

            # a, b = 0, -90  # orientation override
            # a, b = 0, 0  # orientation override

            # print(f'dataset name: {dataset_name}')
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                print('datadet name is modelnet')
                # points = data[0].cuda()
                points = data[0]
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            print(f'original points shape: {points.shape}')
            # dense_points, vis_points = base_model(points, vis=True)
            if config.ADD_NOISE.BOOL:
                # for param in base_model.MAE_encoder.parameters():
                #     param.requires_grad = False

                if config.ADD_NOISE.TYPE == 'NORMAL':
                    NOISE_TYPE = config.ADD_NOISE.TYPE
                    INTENSITY = config.ADD_NOISE.INTENSITY
                    points_noise = points.clone()
                    points_noise, labels = perturb_points(points_noise, N=256, std=0.02)
                    # points_noise, labels, inside_mask_full = add_noise_in_sphere(points_noise, N=700, check_convex=True)
                    if config.ADD_NOISE.CLASSIFICATION == False:
                        vis_points, centers, acc, TP, TN, FP, FN, cdl2, neighborhood_vis, classification_logits, vis_labels = base_model(pts=points_noise, labels=labels, vis=True)
                        # vis_points, centers, acc, acc_outside, TP, TN, FP, FN, cdl2, neighborhood_vis, classification_logits, vis_labels = base_model(pts=points_noise, labels=labels, inside_mask=inside_mask_full, vis=True)
                        print(f'id: {taxonomy_ids[0]}')
                        clean_points = vis_points
                        classification_acc.append(acc)
                        # classification_acc_outside.append(acc_outside)
                        cdl2_array.append(cdl2)
                    else:
                        # Call Denoising module
                        model = Point_Denoiser(config.model, encoder=base_model.MAE_encoder, load_weights=True).cuda()
                        model.eval()
                        clean_points, centers, acc, TP, TN, FP, FN = model(points_noise, labels=labels, vis=True)
                        classification_acc.append(acc)
                        vis_points = clean_points
                elif config.ADD_NOISE.TYPE == 'MIRROR':
                    NOISE_TYPE = config.ADD_NOISE.TYPE
                    noise = Noise(type=NOISE_TYPE)
                    points = noise.addNoise(points, mirror_plane=config.ADD_NOISE.MIRROR_PLANE)
                    dense_points, vis_points, centers, TP, TN, FP, FN, neighborhood_vis, classification_logits, vis_labels = base_model(points, vis=True)
            else:
                print('No Denoising')
                dense_points, vis_points, centers, TP, TN, FP, FN = base_model(points, vis=True)

            final_image = []
            confusion.append((TP, TN, FP, FN))

            data_path = f'./vis/{taxonomy_ids[0]}_{idx}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            if config.ADD_NOISE.BOOL:
                points = points.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path,'gt.txt'), points, delimiter=';')
                points = misc.get_ptcloud_img(points, a, b)
                final_image.append(points[150:650,150:675,:])
                #
                img = np.concatenate(final_image, axis=1)
                img_path = os.path.join(data_path, f'plot_original_points.jpg')
                cv2.imwrite(img_path, img)

                final_image = []

                points_noise = points_noise.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path, 'points_noise.txt'), points_noise, delimiter=';')
                points_noise = misc.get_ptcloud_img(points_noise, a, b)
                final_image.append(points_noise[150:650,150:675,:])

                img = np.concatenate(final_image, axis=1)
                img_path = os.path.join(data_path, f'plot_points_noise.jpg')
                cv2.imwrite(img_path, img)

                final_image = []

                clean_points = clean_points.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path, 'clean.txt'), clean_points, delimiter=';')
                clean_points = misc.get_ptcloud_img(clean_points, a, b)
                final_image.append(clean_points[150:650,150:675,:])

                img = np.concatenate(final_image, axis=1)
                img_path = os.path.join(data_path, f'plot_clean_points.jpg')
                cv2.imwrite(img_path, img)

                # Plot Heatmap of point cloud along with classification_logits values to understand what the classification net predicts
                plot_classification_logits_heatmap(neighborhood_vis,
                                                   classification_logits,
                                                   vis_labels,
                                                   os.path.join(data_path, f'cls_logits_heatmap.png'),
                                                   a, b)
            else:
                points = points.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path, 'gt.txt'), points, delimiter=';')
                points = misc.get_ptcloud_img(points, a, b)
                final_image.append(points[150:650, 150:675, :])

                vis_points = vis_points.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
                vis_points = misc.get_ptcloud_img(vis_points, a, b)

                final_image.append(vis_points[150:650, 150:675, :])

                dense_points = dense_points.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path, 'dense_points.txt'), dense_points, delimiter=';')
                dense_points = misc.get_ptcloud_img(dense_points, a, b)
                final_image.append(dense_points[150:650, 150:675, :])

                img = np.concatenate(final_image, axis=1)
                img_path = os.path.join(data_path, f'plot.jpg')
                cv2.imwrite(img_path, img)

            if idx > 1500:
                print(f'mean classification accuracy: {sum(classification_acc) / len(classification_acc)}')
                print(f'Standard deviation of accuracy: {np.std(torch.stack(classification_acc).cpu().numpy(), ddof=0)}')
                # print(f'mean classification accuracy of outside shape points only: {sum(classification_acc_outside) / len(classification_acc_outside)}')
                print(f'mean Chamfer Distance L2: {sum(cdl2_array) / len(cdl2_array)}')
                calc_confusion_matrix(confusion)
                break
        # if config.ADD_NOISE.CLASSIFICATION:

        return
