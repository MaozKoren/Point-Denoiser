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
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, Delaunay
from sklearn.neighbors import NearestNeighbors


def plot_filtered_points(neighborhood, filename="labels_test.png", plot_type="all"):
    """
    Plot 2D projection of 3D points and save the plot to a file only if it doesn't already exist.

    Parameters:
    - neighborhood: torch.Tensor of shape [x, y, 3] representing 3D points.
    - labels: torch.Tensor of shape [x, y] containing 0 for GT and 1 for noise.
    - filename: The name of the file to save the plot.
    - plot_type: "all" to plot both GT and noise, "gt" to plot only ground truth points, "noise" to plot only noise points.
    """
    # Check if the file already exists
    if os.path.exists(filename):
        # print(f"File '{filename}' already exists. Skipping save.")
        return

    # print(neighborhood)

    # Convert tensors to numpy arrays
    neighborhood_np = neighborhood.view(-1, 3).cpu().numpy()


    # Extract x and y coordinates for the 2D plot
    x_coords = neighborhood_np[:, 0]
    y_coords = neighborhood_np[:, 1]
    z_coords = neighborhood_np[:, 2]

    # Create scatter plot with smaller dots (s=10)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_coords, y_coords, cmap='coolwarm', alpha=0.6, marker='o', s=1)

    # Customize plot
    plt.colorbar(scatter, label='Label (0 = GT, 1 = Noise)')
    plt.title(f"2D Projection of 3D Points ({plot_type.capitalize()} Points)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.4)

    # Save plot to file
    plt.savefig(filename)
    plt.close()  # Close plot to free up memory
    print(f"Plot saved as '{filename}'.")


def plot_neighborhood_lables(neighborhood, labels, all_vis_points, filename="labels_test.png"):
    """
    Plot 2D projection of 3D points in three subplots:
    - The first subplot contains both GT and noise points.
    - The second subplot contains only GT points.
    - The third subplot contains all visualized points.

    Parameters:
    - neighborhood: torch.Tensor of shape [x, y, 3] representing 3D points.
    - labels: torch.Tensor of shape [x, y, 1] containing 0 for GT and 1 for noise.
    - all_vis_points: torch.Tensor of shape [z, 3] representing all visualization points.
    - filename: The name of the file to save the plot.
    """
    # Check if the file already exists
    if os.path.exists(filename):
        return

    # Convert tensors to numpy arrays
    neighborhood_np = neighborhood.view(-1, 3).cpu().numpy()
    labels_np = labels.view(-1).cpu().numpy()
    all_vis_points_np = all_vis_points.view(-1, 3).cpu().numpy()

    # Extract x and y coordinates for the 2D plot
    x_coords = neighborhood_np[:, 0]
    y_coords = neighborhood_np[:, 1]

    # Extract x and y coordinates for all visualized points
    x_coords_vis = all_vis_points_np[:, 0]
    y_coords_vis = all_vis_points_np[:, 1]

    # Create figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot all visualized points
    scatter3 = axes[0].scatter(x_coords_vis, y_coords_vis, c='green', alpha=0.6, marker='o', s=10)
    axes[0].set_title("All Points")
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Y Coordinate")
    axes[0].grid(True, linestyle='--', alpha=0.4)

    # Plot all points (GT + Noise)
    scatter1 = axes[1].scatter(x_coords, y_coords, c=labels_np, cmap='coolwarm', alpha=0.6, marker='o', s=10)
    axes[1].set_title("All Points Predicted as GT")
    axes[1].set_xlabel("X Coordinate")
    axes[1].set_ylabel("Y Coordinate")
    fig.colorbar(scatter1, ax=axes[1], label='Label (0 = GT, 1 = Noise)')
    axes[1].grid(True, linestyle='--', alpha=0.4)

    # Plot only GT points
    gt_mask = labels_np == 0  # Ground truth mask
    scatter2 = axes[2].scatter(x_coords[gt_mask], y_coords[gt_mask], c='blue', alpha=0.6, marker='o', s=10)
    axes[2].set_title("Ground Truth (GT) Points Only")
    axes[2].set_xlabel("X Coordinate")
    axes[2].set_ylabel("Y Coordinate")
    axes[2].grid(True, linestyle='--', alpha=0.4)

    # Save plot to file
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as '{filename}'.")

# def plot_neighborhood_lables(neighborhood, labels, filename="labels_test.png", plot_type="all"):
#     """
#     Plot 2D projection of 3D points and save the plot to a file only if it doesn't already exist.
#
#     Parameters:
#     - neighborhood: torch.Tensor of shape [x, y, 3] representing 3D points.
#     - labels: torch.Tensor of shape [x, y] containing 0 for GT and 1 for noise.
#     - filename: The name of the file to save the plot.
#     - plot_type: "all" to plot both GT and noise, "gt" to plot only ground truth points, "noise" to plot only noise points.
#     """
#     # Check if the file already exists
#     if os.path.exists(filename):
#         # print(f"File '{filename}' already exists. Skipping save.")
#         return
#
#     # Convert tensors to numpy arrays
#     neighborhood_np = neighborhood.view(-1, 3).cpu().numpy()
#
#
#     # Extract x and y coordinates for the 2D plot
#     x_coords = neighborhood_np[:, 0]
#     y_coords = neighborhood_np[:, 1]
#
#     labels_np = labels.view(-1).cpu().numpy()
#
#     # Filter points based on plot_type
#     if plot_type == "gt":
#         mask = labels_np == 0  # Only ground truth points
#     elif plot_type == "noise":
#         mask = labels_np == 1  # Only noise points
#     else:
#         mask = np.ones_like(labels_np, dtype=bool)  # All points
#
#
#
#     # Apply mask to coordinates and labels
#     x_coords = x_coords[mask]
#     y_coords = y_coords[mask]
#     filtered_labels = labels_np[mask]
#
#     # Create scatter plot with smaller dots (s=10)
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(x_coords, y_coords, c=filtered_labels, cmap='coolwarm', alpha=0.6, marker='o', s=10)
#
#     # Customize plot
#     plt.colorbar(scatter, label='Label (0 = GT, 1 = Noise)')
#     plt.title(f"2D Projection of 3D Points ({plot_type.capitalize()} Points)")
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.grid(True, linestyle='--', alpha=0.4)
#
#     # Save plot to file
#     plt.savefig(filename)
#     plt.close()  # Close plot to free up memory
#     print(f"Plot saved as '{filename}'.")


def plot_ROC_curve(TP, TN, FP, FN):
    # Convert lists to numpy arrays
    TP = np.array(TP)
    TN = np.array(TN)
    FP = np.array(FP)
    FN = np.array(FN)

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Calculate TPR and FPR
    tpr = TP / (TP + FN)  # True Positive Rate (Recall)
    fpr = FP / (FP + TN)  # False Positive Rate

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='o', label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')  # Diagonal for random guess
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend()

    # Calculate and display AUC
    roc_auc = auc(fpr, tpr)
    plt.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', fontsize=12)
    plt.savefig(f'ROC_curve {timestamp}.png', dpi=300)


def plot_training_curve_loss(lossesDict):
    plt.figure(figsize=(8, 6))
    for name, losses in lossesDict.items():
        # Ensure tensor is on CPU before converting to NumPy
        if isinstance(losses, torch.Tensor):
            losses = losses.cpu().numpy()
        losses = np.array(losses) / 1000  # Convert to NumPy and divide

        plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', label=name)

    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}-training_loss_plot.png"

    # Save the plot to a file
    plt.savefig(filename, dpi=300)


def plot_training_curve_acc(accDict):
    plt.figure(figsize=(8, 6))

    for name, losses in accDict.items():
        # Flatten & convert everything to float
        flat_losses = []
        for l in losses:
            if isinstance(l, list):
                if len(l) == 1:
                    l = l[0]
                else:
                    print(f"Warning: Nested list with more than 1 element in {name}: {l}")
                    continue
            if isinstance(l, torch.Tensor):
                l = l.detach().cpu().item()
            try:
                flat_losses.append(float(l))
            except Exception as e:
                print(f"Skipping invalid loss value: {l}, error: {e}")

        losses = np.array(flat_losses)   # Normalize if needed
        plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', label=name)

    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}-Training_and_Validation_Accuracy_Over_Epochs_plot.png"
    plt.savefig(filename, dpi=300)


def add_boundaries_and_axes(image, thickness=2, color=(0, 0, 255)):
    # Add boundaries
    image = cv2.copyMakeBorder(image, thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT, value=color)

    # Add x and y axes
    height, width = image.shape[:2]

    # Draw x axis
    cv2.line(image, (0, height - thickness), (width, height - thickness), color, thickness)
    # Draw y axis
    cv2.line(image, (thickness, 0), (thickness, height), color, thickness)

    # Add values on axes
    font_scale = 0.5
    font_thickness = 1

    # Label x-axis
    for i in range(0, width, 50):
        label = str(i)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        label_x = i + thickness - label_size[0] // 2
        label_y = height - 5  # Adjust label position slightly above the border
        cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)

    # Label y-axis
    for j in range(0, height, 50):
        label = str(j)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        label_x = 5  # Adjust label position slightly to the right of the border
        label_y = height - j + thickness + label_size[1] // 2
        cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)

    return image


def save_training_img(mun_iter, gt, unmasked, rebuild, denoise=False):
    a, b = 0, 0
    final_image = []
    data_path = f'./training_vis/{mun_iter}'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    points = gt.squeeze().detach().cpu().numpy()
    # np.savetxt(os.path.join(data_path, 'gt.txt'), points, delimiter=';')
    points = misc.get_ptcloud_img(points, a, b)
    final_image.append(points[150:650, 150:675, :])

    points_gt = unmasked.squeeze().detach().cpu().numpy()
    # np.savetxt(os.path.join(data_path, 'gt.txt'), points, delimiter=';')
    points_gt = misc.get_ptcloud_img(points_gt, a, b)
    final_image.append(points_gt[150:650, 150:675, :])

    # centers = centers.squeeze().detach().cpu().numpy()
    # np.savetxt(os.path.join(data_path,'center.txt'), centers, delimiter=';')
    # centers = misc.get_ptcloud_img(centers)
    # final_image.append(centers)

    # vis_points = unmasked.squeeze().detach().cpu().numpy()
    # # np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
    # vis_points = misc.get_ptcloud_img(vis_points, a, b)
    # final_image.append(vis_points[150:650, 150:675, :])

    dense_points = rebuild.squeeze().detach().cpu().numpy()
    # np.savetxt(os.path.join(data_path, 'dense_points.txt'), dense_points, delimiter=';')
    dense_points = misc.get_ptcloud_img(dense_points, a, b)
    final_image.append(dense_points[150:650, 150:675, :])

    # Add boundaries and axes to each image
    final_image = [add_boundaries_and_axes(img) for img in final_image]

    img = np.concatenate(final_image, axis=1)
    if denoise:
        img_path = os.path.join(data_path, f'{mun_iter}_denoise.jpg')
    else:
        img_path = os.path.join(data_path, f'{mun_iter}.jpg')
    cv2.imwrite(img_path, img)


# def perturb_points(point_cloud, N, offset=0.15):
#     print(point_cloud[0])
#     print(point_cloud[2])
#     # Ensure that the input tensor is of shape [1, 1024, 3]
#     assert point_cloud.shape == (1, 1024, 3), "Input tensor must be of shape [1, 1024, 3]"
#
#     # Flatten the first dimension for easier manipulation
#     point_cloud = point_cloud[0]  # Now the shape is [1024, 3]
#
#     # Randomly select N unique indices to perturb
#     indices = torch.randperm(point_cloud.size(0))[:N]
#
#     # Create random perturbations with values in the range [-offset, offset]
#     perturbations = (torch.rand((N, 3)) - 0.5) * 2 * offset
#     perturbations = perturbations.cuda()
#     # Apply the perturbations to the selected points
#     point_cloud[indices] += perturbations
#
#     # Create a mask tensor of zeros (GT points) and set the selected indices (noise points) to 1
#     noise_mask = torch.zeros(point_cloud.size(0), device=point_cloud.device)  # Shape [1024]
#     noise_mask[indices] = 1  # Set noise points to 1
#
#     # Ensure that the point cloud still has shape [1, 1024, 3] when returned
#     return point_cloud.unsqueeze(0), noise_mask.unsqueeze(0)  # Returning noise mask with shape [1, 1024]

def perturb_points(point_cloud, N, std=0.04):
    """
    Adds Gaussian noise to a random subset of points in the point cloud.

    Args:
        point_cloud (torch.Tensor): Input point cloud of shape [128, 1024, 3].
        N (int): Number of points to perturb in each batch.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        perturbed_point_cloud (torch.Tensor): Point cloud with added Gaussian noise.
        noise_mask (torch.Tensor): Binary mask indicating the perturbed points (1 for perturbed, 0 otherwise).
    """
    # Ensure the input tensor is of the correct shape
    assert point_cloud.shape == (128, 1024, 3), "Input tensor must be of shape [128, 1024, 3]"

    # Create a copy of the point cloud to apply perturbations
    perturbed_point_cloud = point_cloud.clone()

    # Create an empty noise mask for all batches, shape [128, 1024]
    noise_mask = torch.zeros((128, 1024), device=point_cloud.device)

    # Iterate through each batch
    for i in range(point_cloud.shape[0]):  # Loop over the 128 batches
        # Randomly select N unique indices to perturb in the current batch
        indices = torch.randperm(1024)[:N]

        # Create Gaussian noise with mean=0 and std=std
        perturbations = torch.normal(mean=0, std=std, size=(N, 3), device=point_cloud.device)

        # Apply the perturbations to the selected points in the current batch
        perturbed_point_cloud[i, indices] += perturbations

        # Set the noise points in the noise mask for the current batch
        noise_mask[i, indices] = 1

    # Return the perturbed point cloud and the noise mask
    return perturbed_point_cloud, noise_mask


def add_noise_in_sphere(point_cloud, N, noise_density_factor=0.5, std=0.04, fixed=None, check_convex=False):
    """
    Adds noise throughout the entire unit sphere of the point cloud.
    The noise is less dense than the point cloud to keep it visible.

    Args:
        point_cloud (torch.Tensor): Input point cloud tensor of shape [B, 1024, 3], where B is the batch size.
        N (int): Number of noise points to add in the unit sphere per batch.
        noise_density_factor (float): Factor to reduce the density of the noise points.
        std (float): Standard deviation for perturbation.

    Returns:
        torch.Tensor: Perturbed point cloud with added noise of shape [B, 1024 + N, 3].
        torch.Tensor: Noise indicator tensor (1 for noise points, 0 for original points) of shape [B, 1024 + N, 1].
    """
    # Ensure the input tensor has the correct dimensions
    assert point_cloud.dim() == 3 and point_cloud.size(2) == 3, "Input tensor must be of shape [B, 1024, 3]"

    B, P, C = point_cloud.shape  # Batch size, number of points, and coordinate dimensions
    assert P == 1024, "Point cloud must have exactly 1024 points per batch"

    if fixed is None:
        # 1. Generate random noise points within the unit sphere for each batch
        phi = torch.rand(B, N, 1, device=point_cloud.device) * 2 * torch.pi  # Random angles
        costheta = torch.rand(B, N, 1, device=point_cloud.device) * 2 - 1  # Random cos(theta)
        theta = torch.acos(costheta)  # theta from cos(theta)
        r = torch.rand(B, N, 1, device=point_cloud.device) ** (1 / 3)  # Uniformly distributed radius

        # Convert to Cartesian coordinates (x, y, z) in 3D space
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)

        # Stack to form noise points within the unit sphere
        noise_points = torch.cat((x, y, z), dim=2)  # Shape: [B, N, 3]
    else:
        noise_points = fixed

    # 2. Add random perturbation to the original points
    # perturbations = torch.normal(mean=0, std=std, size=(B, P, C), device=point_cloud.device)
    # perturbed_point_cloud = point_cloud + perturbations

    # 3. Concatenate the original point cloud and noise points
    # perturbed_point_cloud = torch.cat([perturbed_point_cloud, noise_points], dim=1)  # Shape: [B, 1024 + N, 3]
    perturbed_point_cloud = torch.cat([point_cloud, noise_points], dim=1)  # Shape: [B, 1024 + N, 3] # temp

    # 4. Create a noise mask
    noise_mask = torch.zeros(B, 1024 + N, 1, device=point_cloud.device)
    noise_mask[:, 1024:] = 1  # Mark noise points
    # print(f'point cloud shape: {point_cloud.shape}')
    # ratios = []
    # for i in range(B):
    #     ratios.append(percentage_noise_inside_shape_convex(point_cloud=point_cloud[i], noise_points=noise_points[i]))
    # # 5. Return the perturbed point cloud and the noise mask
    # mean_ratio = sum(ratios) / len(ratios)
    # print(f'Ratio of noise points in shape: {mean_ratio}')
    if check_convex:
        inside_mask = mask_noise_inside_shape_convex(point_cloud=point_cloud[0], noise_points=noise_points[0])
        inside_mask_full = torch.zeros(B, 1024 + N, 1, device=point_cloud.device)
        inside_mask = torch.from_numpy(inside_mask).to(device=inside_mask_full.device, dtype=inside_mask_full.dtype)
        inside_mask = inside_mask.unsqueeze(0).unsqueeze(-1)  # Shape: [1, 700, 1]
        inside_mask_full[:, 1024:] = inside_mask
        return perturbed_point_cloud, noise_mask, inside_mask_full  # True means inside
    else:
        return perturbed_point_cloud, noise_mask



def filter_centers_out_of_shape(neighborhood, center, cell_radius):
    """
    Filters center points that are outside the given point cloud based on a specified radius.

    Args:
        neighborhood (torch.Tensor): Tensor of shape [1, 175, 32, 3] representing the neighborhoods of center points.
        center (torch.Tensor): Tensor of shape [1, 175, 3] representing the center points.
        cell_radius (float): Radius to check for points in neighborhood around each center.

    Returns:
        torch.Tensor: Boolean mask of shape [1, 175, 1], where True indicates the center has at least one
                      neighboring point within cell_radius.
    """
    # Remove the batch dimension for easier computation
    neighborhood = neighborhood.squeeze(0)  # Shape: [175, 32, 3]
    center = center.squeeze(0)  # Shape: [175, 3]

    # Compute distances between each center point and its corresponding neighborhood
    distances = torch.norm(neighborhood - center.unsqueeze(1), dim=-1)  # Shape: [175, 32]

    # Check if any distance is less than or equal to the cell_radius for each center point
    mask = (distances <= cell_radius).any(dim=1, keepdim=True)  # Shape: [175, 1]

    # Add the batch dimension back to the mask
    mask = mask.unsqueeze(0)  # Shape: [1, 175, 1]

    return mask


def get_intermediate_output(model, neighborhood, center, layer_idx):
    """
    Extract the output from the N'th layer of the encoder using a forward hook.

    Args:
        model: The trained Point-MAE encoder model.
        neighborhood: The input neighborhood point cloud tensor.
        center: The center point tensor.
        layer_idx: The index of the layer to extract.

    Returns:
        The latent space of the specified layer.
    """
    activation = {}

    def hook_fn(module, input, output):
        # Store the output of the hooked layer
        activation['output'] = output.detach()

    # Register the hook to the N'th transformer block inside the encoder
    hook_handle = model.blocks.blocks[layer_idx].register_forward_hook(hook_fn)

    # Forward pass through the encoder
    with torch.no_grad():
        _ = model(neighborhood, center)

    # Remove the hook after the forward pass
    hook_handle.remove()

    return activation['output']


from collections import Counter


def count_repeated_values(filtered_idx):
    """
    Counts how many values occur more than once in each batch of the filtered_idx tensor
    and prints the count along with how many times they occur.

    Args:
        filtered_idx: Tensor of shape [128, X, 32] containing filtered values for each batch.
    """
    batch_size = filtered_idx.size(0)  # Number of batches (128)

    for batch_idx in range(batch_size):
        batch_values = filtered_idx[batch_idx].view(-1).tolist()  # Flatten the batch into a list
        value_counts = Counter(batch_values)  # Count occurrences of each value

        # Find values that occur more than once
        repeated_values = {val: count for val, count in value_counts.items() if count > 1}

        # Print out the results for this batch
        print(f"Batch {batch_idx}: {len(repeated_values)} values occur more than once")
        for val, count in repeated_values.items():
            print(f"Value {val} occurs {count} times")
        print('-' * 40)


import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# Define the save directory

def save_latent_pic(x_vis, labels):
    save_dir = 'latent_pictures'
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    save_path_data = os.path.join(save_dir, 'x_vis_pca_data.npy')
    save_path_labels = os.path.join(save_dir, 'x_vis_pca_labels.npy')
    save_path_fig = os.path.join(save_dir, 'x_vis_pca_visualization.png')

    # Flatten x_vis along batch and point dimensions
    x_flat = x_vis.view(-1, 384).cpu().numpy()  # shape (26, 384)

    # Reduce labels to match x_vis shape
    # Assuming x_vis corresponds to 26 groups (subset of neighborhoods)
    labels_flat = labels.view(-1, 32).mean(dim=1).cpu().numpy()[:26]
    labels_flat = (labels_flat > 0.5).astype(int)  # Convert to binary (0: non-noise, 1: noise)

    # Apply PCA to reduce x_vis to 2D
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_flat)

    # Load existing PCA data and labels if they exist
    if os.path.exists(save_path_data):
        existing_data = np.load(save_path_data)
        existing_labels = np.load(save_path_labels)
        all_data = np.vstack((existing_data, x_pca))
        all_labels = np.hstack((existing_labels, labels_flat))
    else:
        all_data = x_pca
        all_labels = labels_flat

    # Save the updated data and labels
    np.save(save_path_data, all_data)
    np.save(save_path_labels, all_labels)

    # Plot the accumulated PCA points with colors based on labels
    plt.figure(figsize=(10, 8))
    plt.scatter(all_data[all_labels == 0, 0], all_data[all_labels == 0, 1], s=5, alpha=0.6, c='blue', label='Non-Noise')
    plt.scatter(all_data[all_labels == 1, 0], all_data[all_labels == 1, 1], s=5, alpha=0.6, c='red', label='Noise')

    plt.title("PCA Visualization of x_vis (Noise vs Non-Noise)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(save_path_fig)
    plt.close()
# def save_latent_pic(x_vis, labels):
#     save_dir = 'latent_pictures'
#     os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
#
#     # Flatten along batch and point dimensions
#     x_flat = x_vis.view(-1, 384).cpu().numpy()  # shape (128 * 26, 384)
#
#     # Apply PCA
#     pca = PCA(n_components=2)
#     x_pca = pca.fit_transform(x_flat)
#
#     # Plot the results
#     plt.figure(figsize=(10, 8))
#     plt.scatter(x_pca[:, 0], x_pca[:, 1], s=5, alpha=0.6)
#     plt.title("2D PCA Visualization of x_vis")
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#
#     # # Initialize t-SNE with 2 components for visualization
#     # tsne = TSNE(n_components=2, random_state=42)
#     # embeddings_2d = tsne.fit_transform(x_flat)
#     # labels_np = labels.cpu().numpy()
#     #
#     # # Plot the results
#     # plt.figure(figsize=(10, 8))
#     # plt.scatter(embeddings_2d[labels_np == 0, 0], embeddings_2d[labels_np == 0, 1],
#     #             c='blue', label='Non-Noise', alpha=0.6)
#     # plt.scatter(embeddings_2d[labels_np == 1, 0], embeddings_2d[labels_np == 1, 1],
#     #             c='red', label='Noise', alpha=0.6)
#     # plt.legend()
#     # plt.title("t-SNE Visualization of Noise vs. Non-Noise Point Embeddings")
#     # plt.xlabel("t-SNE Dimension 1")
#     # plt.ylabel("t-SNE Dimension 2")
#
#     # Save the plot as PNG
#     save_path = os.path.join(save_dir, 'x_vis_pca_visualization.png')
#     # Check if the file already exists
#     if not os.path.exists(save_path):
#         # Plot the results
#         plt.figure(figsize=(10, 8))
#         plt.scatter(x_pca[:, 0], x_pca[:, 1], s=5, alpha=0.6)
#         plt.title("PCA Visualization of x_vis")
#         plt.xlabel("PCA Component 1")
#         plt.ylabel("PCA Component 2")
#
#         # Save the plot as PNG
#         plt.savefig(save_path, format='png')
#         plt.close()  # Close the plot to free memory
#     # else:
#     #     print(f"File already exists at {save_path}, skipping save.")


def plot_point_clouds(tensor1, tensor2, tensor3, roll=30, pitch=30):
    """
    Plots three 3D point clouds with adjustable roll and pitch angles and returns the figure.

    Args:
        tensor1 (torch.Tensor): First tensor of shape [1, 1024, 3].
        tensor2 (torch.Tensor): Second tensor of shape [1, 1024, 3].
        tensor3 (torch.Tensor): Third tensor of shape [N, 3].
        roll (float): The roll angle for 3D view initialization.
        pitch (float): The pitch angle for 3D view initialization.

    Returns:
        fig (plt.Figure): The matplotlib figure object containing the plot.
    """

    # Convert tensors to numpy arrays and squeeze the first dimension if needed
    points1 = tensor1.squeeze(0).cpu().numpy()
    points2 = tensor2.squeeze(0).cpu().numpy()
    points3 = tensor3.cpu().numpy()

    # Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    # ax.axis('scaled')  # Uncomment if you want equal scaling on all axes

    # Plot each point cloud with a different color
    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='r', marker='o', s=2, label='Point Cloud 1')
    ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='g', marker='^', s=2, label='Point Cloud 2')
    ax.scatter(points3[:, 0], points3[:, 1], points3[:, 2], c='b', marker='x', s=2, label='Point Cloud 3')

    # Set the roll and pitch for the viewpoint
    ax.view_init(roll, pitch)

    # Add a legend to distinguish between point clouds
    ax.legend(loc='upper right')

    # Return the figure
    return fig


def calc_confusion_matrix(confusion, filename="confusion_matrix.png"):
    """
    Calculate and visualize the mean confusion matrix from a list of tuples (TP, TN, FP, FN) and save it as a PNG file.

    Parameters:
    confusion (list of tuples): List of (TP, TN, FP, FN) values.
    filename (str): File name to save the image.
    """
    # Convert list of tuples to NumPy array for easy manipulation
    confusion_array = np.array(confusion)

    # Compute mean values
    mean_values = confusion_array.mean(axis=0)
    TP, TN, FP, FN = mean_values

    # Create a confusion matrix in table format
    matrix = np.array([[TP, FP], [FN, TN]])

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues", interpolation='nearest')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='black')

    # Set axis labels
    plt.xticks([0, 1], ["Predicted Positive", "Predicted Negative"])
    plt.yticks([0, 1], ["Actual Positive", "Actual Negative"])
    plt.title("Mean Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    # Save the figure instead of showing it
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def percentage_noise_inside_shape(points, labels, k=100, density_threshold=1.5):
    """
    Calculate the percentage of noise points inside the shape using KNN Density.

    Args:
        points (torch.Tensor): Tensor of shape [N, 3], representing the point cloud.
        labels (torch.Tensor): Tensor of shape [N, 1], where 1 = noise, 0 = clean.
        k (int): Number of nearest neighbors for density estimation.
        density_threshold (float): Threshold for defining the shape's density.

    Returns:
        float: Percentage of noise points inside the shape.
    """
    points_np = points.cpu().numpy()
    labels_np = labels.cpu().numpy().squeeze(-1)  # Ensure shape compatibility

    # Extract clean (GT) and noise points
    clean_points = points_np[labels_np == 0]
    noise_points = points_np[labels_np == 1]

    if len(clean_points) < k or len(noise_points) == 0:
        print("Not enough points for KNN calculation.")
        return 0.0

    # Fit KNN on clean points
    knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    knn.fit(clean_points)

    # Compute density for clean points
    distances, _ = knn.kneighbors(clean_points)
    clean_density = np.mean(distances, axis=1)  # Lower values = denser areas

    # Compute density for noise points
    noise_distances, _ = knn.kneighbors(noise_points)
    noise_density = np.mean(noise_distances, axis=1)

    # Define "inside shape" as noise points with density below threshold
    inside_mask = noise_density < density_threshold * np.median(clean_density)

    # Calculate percentage
    percentage_inside = np.sum(inside_mask) / len(noise_points)

    return percentage_inside


def percentage_noise_inside_shape_convex(point_cloud, noise_points):
    """
    Calculate the percentage of noise points inside the convex hull of the main point cloud shape.

    Args:
        point_cloud (torch.Tensor): Tensor of shape [N, 3], representing the clean point cloud.
        noise_points (torch.Tensor): Tensor of shape [M, 3], representing the noise points.

    Returns:
        float: Percentage of noise points inside the shape.
    """
    point_cloud_np = point_cloud.cpu().numpy()
    noise_points_np = noise_points.cpu().numpy()

    # If there are no noise points, return 0
    if noise_points_np.shape[0] == 0:
        return 0.0

    # If there are not enough points to form a convex hull
    if point_cloud_np.shape[0] < 4:
        print("Not enough points to form a convex hull.")
        return 0.0

    # Compute convex hull for the clean point cloud
    hull = ConvexHull(point_cloud_np)
    delaunay = Delaunay(hull.points)

    # Check which noise points are inside the convex hull
    inside_mask = delaunay.find_simplex(noise_points_np) >= 0

    # Calculate percentage of noise points inside the shape
    percentage_inside = np.sum(inside_mask) / len(noise_points_np)

    return percentage_inside

def mask_noise_inside_shape_convex(point_cloud, noise_points):
    """
    Calculate the percentage of noise points inside the convex hull of the main point cloud shape.

    Args:
        point_cloud (torch.Tensor): Tensor of shape [N, 3], representing the clean point cloud.
        noise_points (torch.Tensor): Tensor of shape [M, 3], representing the noise points.

    Returns:
        float: Percentage of noise points inside the shape.
    """
    point_cloud_np = point_cloud.cpu().numpy()
    noise_points_np = noise_points.cpu().numpy()

    # If there are no noise points, return 0
    if noise_points_np.shape[0] == 0:
        return 0.0

    # If there are not enough points to form a convex hull
    if point_cloud_np.shape[0] < 4:
        print("Not enough points to form a convex hull.")
        return 0.0

    # Compute convex hull for the clean point cloud
    hull = ConvexHull(point_cloud_np)
    delaunay = Delaunay(hull.points)

    # Check which noise points are inside the convex hull
    inside_mask = delaunay.find_simplex(noise_points_np) >= 0

    return inside_mask  # True means inside


def filter_neighborhood_points(neighborhood, center, cell_radius):
    """
    Filters individual neighborhood points based on their distance from the corresponding center.

    Args:
        neighborhood (torch.Tensor): Shape [1, 175, 32, 3], representing neighborhood points.
        center (torch.Tensor): Shape [1, 175, 3], representing center points.
        cell_radius (float): Radius threshold for valid points.

    Returns:
        torch.Tensor: Boolean mask of shape [1, 175, 32, 1], where True means the point is inside the radius.
    """
    # Remove batch dimension for easier computation
    neighborhood = neighborhood.squeeze(0)  # Shape: [175, 32, 3]
    center = center.squeeze(0)  # Shape: [175, 3]

    # Compute Euclidean distance of each neighborhood point to its center
    distances = torch.norm(neighborhood - center.unsqueeze(1), dim=-1)  # Shape: [175, 32]

    # Create mask for each individual point (True if inside cell_radius, False otherwise)
    mask = (distances <= cell_radius).unsqueeze(-1)  # Shape: [175, 32, 1]

    # Add batch dimension back
    mask = mask.unsqueeze(0)  # Shape: [1, 175, 32, 1]

    return mask  # True means keep!!


# def plot_classification_logits_heatmap(neighborhood_vis, classification_logits, path):
#     """
#     Plots neighborhood_vis points with colors based on classification_logits values.
#     """
#     # Ensure tensors are detached and converted to numpy
#     if isinstance(neighborhood_vis, torch.Tensor):
#         neighborhood_vis = neighborhood_vis.detach().cpu().numpy()
#     if isinstance(classification_logits, torch.Tensor):
#         classification_logits = classification_logits.detach().cpu().numpy()
#
#
#     # Create a scatter plot
#     plt.figure(figsize=(8, 6))
#     sc = plt.scatter(
#         neighborhood_vis[..., 0],
#         neighborhood_vis[..., 1],
#         c=classification_logits, cmap='viridis', alpha=0.7
#     )
#     plt.colorbar(sc, label='Classification Logits')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.title('Classification Logits Heatmap')
#
#     # Save the plot
#     plt.savefig(path, dpi=300)
#     plt.close()


def plot_classification_logits_heatmap(neighborhood_vis, classification_logits, path, roll=30, pitch=60):
    """
    Plots neighborhood_vis points in 3D with colors based on classification_logits values,
    and applies a custom camera orientation using roll and pitch.
    """
    # Ensure tensors are detached and converted to numpy
    if isinstance(neighborhood_vis, torch.Tensor):
        neighborhood_vis = neighborhood_vis.detach().cpu().numpy()
    if isinstance(classification_logits, torch.Tensor):
        classification_logits = classification_logits.detach().cpu().numpy()

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    # Set camera view
    ax.view_init(roll, pitch)

    # Bound axes equally
    max_val = np.max(neighborhood_vis)
    min_val = np.min(neighborhood_vis)
    ax.set_xbound(min_val, max_val)
    ax.set_ybound(min_val, max_val)
    ax.set_zbound(min_val, max_val)

    # Scatter plot with classification logits as color
    sc = ax.scatter(
        neighborhood_vis[..., 0],
        neighborhood_vis[..., 1],
        neighborhood_vis[..., 2],
        c=classification_logits,
        zdir='z',
        cmap='viridis',
        s=5,
        depthshade=False
    )

    # Add color bar
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label='Classification Logits')

    # Save the figure
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save the data to .npz
    base, _ = os.path.splitext(path)
    np.savez_compressed(base + "_data.npz", neighborhood_vis=neighborhood_vis, classification_logits=classification_logits)


def farthest_squared_distance(src, tgt, chunk_size=1024):
    """
    Computes the squared distance from each point in src to the farthest point in tgt,
    using chunked computation to avoid OOM errors.

    Args:
        src (torch.Tensor): Source point cloud of shape (B, N, 3).
        tgt (torch.Tensor): Target point cloud of shape (B, M, 3).
        chunk_size (int): Number of target points to process at a time.

    Returns:
        torch.Tensor: Squared distance to the farthest point for each source point (B, N).
    """
    B, N, _ = src.shape
    B, M, _ = tgt.shape
    farthest_distances = torch.zeros(B, N, device=src.device)  # To store max distances

    # Compute squared norms
    src_norm = torch.sum(src ** 2, dim=-1, keepdim=True)  # (B, N, 1)

    for i in range(0, M, chunk_size):
        tgt_chunk = tgt[:, i:i + chunk_size, :]  # Select chunk of target points
        tgt_norm = torch.sum(tgt_chunk ** 2, dim=-1, keepdim=True)  # (B, chunk_size, 1)

        # Compute distances for this chunk
        dists_chunk = src_norm - 2 * (src @ tgt_chunk.transpose(-2, -1)) + tgt_norm.transpose(-2,
                                                                                              -1)  # (B, N, chunk_size)
        dists_chunk = torch.clamp(dists_chunk, min=0)  # Ensure non-negative values

        # Update max distances
        farthest_distances = torch.maximum(farthest_distances, torch.max(dists_chunk, dim=2)[0])

    return farthest_distances.mean()


