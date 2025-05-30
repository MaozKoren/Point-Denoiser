import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc
# from pointnet2_ops import pointnet2_utils
from pointnet import pointnet2 as pointnet2_utils
from scipy.spatial import cKDTree

def gather_new(x, idx):
    B, C, N = x.size()
    _, M = idx.size()

    idx = idx.unsqueeze(1).expand(B, C, M)
    gathered_points = torch.gather(x, 2, idx)

    return gathered_points


def create_sphere_grid(spaces, dist=0.1, batch_size=1):
    """
    Creates a spherical grid of points with a fixed distance between them, evenly distributed
    on concentric spherical layers, and replicates the grid for the given number of batches.

    Args:
    - spaces (float): Distance between adjacent points in the grid.
    - dist (float): Minimum distance from the center of the unit sphere (default 0.1).
    - batch_size (int): Number of batches to generate the same spherical grid for (default 1).

    Returns:
    - centers (torch tensor): A tensor of shape (batch_size, num_points, 3) containing spherical grid points.
    """
    # Number of spherical layers
    num_layers = int((1 - dist) // spaces)

    points = []
    for layer in range(1, num_layers + 1):
        # Calculate radius of the current layer
        radius = layer * spaces

        # Calculate the number of points to place on the current spherical layer
        num_points_in_layer = int(4 * np.pi * radius ** 2 / spaces ** 2)  # Approximate number of points

        # Use Fibonacci Sphere to evenly distribute points on the surface of the sphere
        for i in range(num_points_in_layer):
            # Fibonacci sphere method to distribute points evenly
            phi = np.arccos(1 - 2 * (i + 0.5) / num_points_in_layer)  # Polar angle
            theta = np.pi * (3.0 - np.sqrt(5.0)) * (i + 0.5)  # Azimuthal angle

            # Convert spherical to Cartesian coordinates
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)

            points.append([x, y, z])

    # Convert list of points to tensor
    centers = torch.tensor(points, dtype=torch.float32)  # Shape: (num_points, 3)

    # Replicate the same grid for the specified batch size
    centers = centers.unsqueeze(0).repeat(batch_size, 1, 1).cuda()  # Shape: (batch_size, num_points, 3)

    return centers

def pad_point_clouds(filtered_xyz, max_points):
    """
    Pad point clouds with zeros to ensure all point clouds have the same number of points.

    Args:
        filtered_xyz (list or numpy.ndarray of varying shapes): List of point clouds with varying numbers of points.
        max_points (int): Maximum number of points to pad to.

    Returns:
        torch.Tensor: Padded point clouds of shape (batch_size, max_points, 3).
    """
    padded_point_clouds = []
    for point_cloud in filtered_xyz:
        # Ensure point_cloud is a NumPy array of proper type
        if isinstance(point_cloud, np.ndarray) and point_cloud.dtype == np.object_:
            point_cloud = np.array(point_cloud.tolist(), dtype=np.float32)

        # Convert to PyTorch tensor if needed
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

        # Create padding
        padding = torch.zeros((max_points - point_cloud.shape[0], point_cloud.shape[1]), dtype=torch.float32)

        # Pad along the first dimension
        padded_point_cloud = torch.cat([point_cloud, padding], dim=0)
        padded_point_clouds.append(padded_point_cloud)

    # Stack all padded point clouds into a batch tensor
    return torch.stack(padded_point_clouds, dim=0)

def statistical_outlier_removal(xyz, k=10, std_ratio=0.04):
    """
    Perform Statistical Outlier Removal (SOR) on a point cloud.

    Args:
        xyz (torch.Tensor): Input point cloud of shape (N, 3).
        k (int): Number of neighbors to consider for distance computation.
        std_ratio (float): Standard deviation multiplier for the distance threshold.

    Returns:
        filtered_xyz (torch.Tensor): Filtered point cloud of shape (M, 3).
    """
    batch_size, num_points, dim = xyz.shape
    filtered_batches = []
    xyz_np = xyz.cpu().numpy()  # Convert to numpy for k-d tree operations
    for i in range(batch_size):
        tree = cKDTree(xyz_np[i])
        distances, _ = tree.query(xyz_np[i], k=k + 1)  # k+1 because the nearest neighbor is the point itself
        mean_distances = distances[:, 1:].mean(axis=1)  # Exclude self-distance
        std_distance = mean_distances.std()
        threshold = mean_distances.mean() + std_ratio * std_distance
        # Filter points
        mask = mean_distances < threshold
        filtered_xyz = xyz_np[i][mask]
        filtered_batches.append(filtered_xyz)

    return np.array(filtered_batches, dtype=object)


def statistical_outlier_std(xyz, k=10, std_ratio=0.04):
    """
    Perform Statistical Outlier Removal (SOR) on a point cloud.

    Args:
        xyz (torch.Tensor): Input point cloud of shape (N, 3).
        k (int): Number of neighbors to consider for distance computation.
        std_ratio (float): Standard deviation multiplier for the distance threshold.

    Returns:
        filtered_xyz (torch.Tensor): Filtered point cloud of shape (M, 3).
    """
    batch_size, num_points, dim = xyz.shape
    mean_distance = []
    xyz_np = xyz.cpu().numpy()  # Convert to numpy for k-d tree operations
    for i in range(batch_size):
        tree = cKDTree(xyz_np[i])
        distances, _ = tree.query(xyz_np[i], k=k + 1)  # k+1 because the nearest neighbor is the point itself
        mean_distances = distances[:, 1:].mean(axis=1)  # Exclude self-distance
        # std_distance = mean_distances.std()
        # threshold = mean_distances.mean() + std_ratio * std_distance
        mean_distance.append(mean_distances.mean())
    return sum(mean_distance) / len(mean_distance)


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.farthest_point_sampling(data, number) 
#    fps_data = pointnet2_utils.gather(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    fps_data = gather_new(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous() ## B Gn 6
    return fps_data


# Hybrid FPS + Statistical Outlier Removal
def hybrid_sampling(xyz, num_samples, k=32, std_ratio=0.04*20):
    """
    Hybrid sampling combining Statistical Outlier Removal (SOR) and Farthest Point Sampling (FPS).

    Args:
        xyz (torch.Tensor): Input point cloud of shape (N, 3).
        num_samples (int): Number of points to sample.
        k (int): Number of neighbors for SOR.
        std_ratio (float): Standard deviation multiplier for the distance threshold in SOR.

    Returns:
        sampled_xyz (torch.Tensor): Sampled points of shape (num_samples, 3).
    """
    # Step 1: Remove outliers using SOR
    filtered_xyz = statistical_outlier_removal(xyz, k=k, std_ratio=std_ratio)

    # Step 2: Find the max number of points in any point cloud
    max_points = max([point_cloud.shape[0] for point_cloud in filtered_xyz])

    # Step 3: Pad point clouds to the same size and stack
    filtered_xyz = pad_point_clouds(filtered_xyz, max_points)
    # print(f"Filtered point cloud size after padding: {filtered_xyz.shape}")
    # Step 2: Apply FPS on the filtered point cloud
    sampled_xyz = fps(filtered_xyz, num_samples)
    return sampled_xyz

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_lambda_sche(opti, config):
    if config.get('decay_step') is not None:
        lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd)
    else:
        raise NotImplementedError()
    return scheduler

def build_lambda_bnsche(model, config):
    if config.get('decay_step') is not None:
        bnm_lmbd = lambda e: max(config.bn_momentum * config.bn_decay ** (e / config.decay_step), config.lowest_decay)
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    else:
        raise NotImplementedError()
    return bnm_scheduler
    
def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)



def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()

def get_ptcloud_img(ptcloud,roll,pitch):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
#    ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(roll,pitch)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    # ax.scatter(x, y, z, zdir='z', c=y, cmap='jet')
    ax.scatter(x, y, z, zdir='z', color='blue', s=5, depthshade=False)

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img

def get_ptcloud_img_denoise(ptcloud, roll, pitch):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    ax.view_init(roll, pitch)

    for cloud in ptcloud:
        x, z, y = cloud.transpose(1, 0)
        ax.scatter(x, y, z, zdir='z', c=y, cmap='jet')

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)  # Close the figure to prevent displaying it
    return img



def visualize_KITTI(path, data_list, titles = ['input','pred'], cmap=['bwr','autumn'], zdir='y', 
                         xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1) ):
    fig = plt.figure(figsize=(6*len(data_list),6))
    cmax = data_list[-1][:,0].max()

    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:,0] /cmax
        ax = fig.add_subplot(1, len(data_list) , i + 1, projection='3d')
        ax.view_init(30, -120)
        b = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir=zdir, c=color,vmin=-1,vmax=1 ,cmap = cmap[0],s=4,linewidth=0.05, edgecolors = 'black')
        ax.set_title(titles[i])

        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    if not os.path.exists(path):
        os.makedirs(path)

    pic_path = path + '.png'
    fig.savefig(pic_path)

    np.save(os.path.join(path, 'input.npy'), data_list[0].numpy())
    np.save(os.path.join(path, 'pred.npy'), data_list[1].numpy())
    plt.close(fig)


def random_dropping(pc, e):
    up_num = max(64, 768 // (e//50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1,1))[0,0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim = 1)
    return pc
    

def random_scale(partial, scale_range=[0.8, 1.2]):
    scale = torch.rand(1).cuda() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return partial * scale


def compute_local_density(xyz, k=10):
    """
    Compute local density for each point based on inverse distances to k-nearest neighbors.

    Args:
        xyz (torch.Tensor): Point cloud of shape (batch_size, num_points, 3).
        k (int): Number of nearest neighbors.

    Returns:
        local_density (torch.Tensor): Density score of shape (batch_size, num_points).
    """
    batch_size, num_points, _ = xyz.shape
    local_density = torch.zeros(batch_size, num_points, device=xyz.device)

    for i in range(batch_size):
        xyz_np = xyz[i].cpu().numpy()  # Convert to numpy
        tree = cKDTree(xyz_np)  # Build KD-tree
        distances, _ = tree.query(xyz_np, k=k+1)  # k+1 because first neighbor is the point itself
        distances = distances[:, 1:]  # Remove self-distance (0)

        inv_dist = 1.0 / (distances + 1e-6)  # Avoid division by zero
        local_density[i] = torch.tensor(inv_dist.sum(axis=1), device=xyz.device)

    return local_density


def compute_outlier_loss(xyz, k=10, lambda_outlier=1000000):
    """
    Compute an outlier loss term based on local density differences.

    Args:
        xyz (torch.Tensor): Point cloud of shape (batch_size, num_points, 3).
        k (int): Number of nearest neighbors.
        lambda_outlier (float): Weight for outlier loss.

    Returns:
        outlier_loss (torch.Tensor): Scalar loss penalizing outliers.
    """
    local_density = compute_local_density(xyz, k)

    batch_size, num_points = local_density.shape
    outlier_loss = torch.zeros(batch_size, device=xyz.device)

    for i in range(batch_size):
        xyz_np = xyz[i].cpu().numpy()
        tree = cKDTree(xyz_np)
        _, neighbors = tree.query(xyz_np, k=k+1)
        neighbors = neighbors[:, 1:]  # Remove self-index

        neighbor_density = local_density[i][torch.tensor(neighbors, device=xyz.device)]
        mean_neighbor_density = neighbor_density.mean(dim=1)

        # Outlier score: Penalize points with lower density than their neighbors
        outlier_score = F.relu(mean_neighbor_density - local_density[i])

        outlier_loss[i] = outlier_score.mean()

    return outlier_loss.mean() / lambda_outlier
