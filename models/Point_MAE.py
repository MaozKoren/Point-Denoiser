import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN # temporary disables to run on windows
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2 , ChamferDistanceL2_D1D2 # temporary disables to run on windows
import copy
from datasets.noise import Noise
from tools.helper_functions import farthest_squared_distance, count_repeated_values, save_latent_pic, get_intermediate_output, filter_neighborhood_points, filter_logits_with_voting
from utils.misc import statistical_outlier_std, compute_outlier_loss

def adjust_range(min_value, max_value, N):
    # Adjust the range to exclude the start and end points
    adjusted_min = min_value + (max_value - min_value) / (N + 1)
    adjusted_max = max_value - (max_value - min_value) / (N + 1)
    return adjusted_min, adjusted_max


def rank_centers_by_distance(centers: torch.Tensor, xyz: torch.Tensor, percent: float) -> torch.Tensor:
    # Calculate squared distances between each center and all points in xyz
    num_points = centers.shape[0]
    squared_distances = []

    for center in centers:
        # Compute squared distance for each center to all points in xyz
        distances = torch.sum((xyz - center) ** 2, dim=1)  # Shape: (N,)
        total_distance = torch.sum(distances).item()  # Sum of squared distances for this center
        squared_distances.append(total_distance)

    # Convert list of total squared distances to a tensor
    squared_distances = torch.tensor(squared_distances)

    # Sort the centers based on the sum of squared distances (ascending order)
    sorted_indices = torch.argsort(squared_distances)

    # Determine how many centers to keep based on the 'percent' value
    num_selected = int(num_points * percent)

    # Select the centers with the lowest sum of squared distances
    selected_indices = sorted_indices[:num_selected]
    selected_centers = centers[selected_indices]

    return selected_centers

def create_grid(xyz, num_points):
    '''
    xyz: torch tensor with shape (128, 1024, 3)
    '''
    reshaped_point_cloud = xyz.view(-1, 3)

    # num_points = round(num_points ** (1 / 3))
    num_points_grid = round(20)
    # Find the minimum and maximum coordinates along each axis
    min_coords = reshaped_point_cloud.min(dim=0)[0]
    max_coords = reshaped_point_cloud.max(dim=0)[0]

    # Adjust the range
    x_min, x_max = adjust_range(min_coords[0], max_coords[0], num_points_grid)
    y_min, y_max = adjust_range(min_coords[1], max_coords[1], num_points_grid)
    z_min, z_max = adjust_range(min_coords[2], max_coords[2], num_points_grid)

    # Generate points for each axis
    x_points = torch.linspace(x_min, x_max, num_points_grid)
    y_points = torch.linspace(y_min, y_max, num_points_grid)
    z_points = torch.linspace(z_min, z_max, num_points_grid)

    # Create a meshgrid of points
    X, Y, Z = torch.meshgrid(x_points, y_points, z_points)

    # Combine into a single tensor of 3D points
    centers = torch.stack([X, Y, Z], dim=-1).view(-1, 3).cuda()  # Shape: (num_points**3, 3)
    # Remove points that are not in the point cloud
    centers = rank_centers_by_distance(centers=centers, xyz=reshaped_point_cloud, percent=0.1)
    # print(f' centers shape: {centers.shape}')
    # fps to get num_points center points
    centers = misc.fps(centers.unsqueeze(0), num_points)
    # print(f' centers shape: {centers.shape}')
    centers = centers.expand(xyz.shape[0], -1, -1).cuda()   # Shape: (128, num_points**3, 3)  # Shape: (128, 64, 3)
    # print(f' centers shape: {centers.shape}')
    return centers


def extract_neighborhoods(xyz, centers, range):
    # Initialize an empty list to store neighborhoods
    neighborhoods = []

    # Iterate over each center
    for center in centers:
        # Calculate the distances from the center to all points
        distances = torch.norm(xyz - center, dim=-1)

        # Select points within the range to form the neighborhood
        neighborhood = xyz[distances < range]

        # Subtract the center coordinates from the neighborhood points
        neighborhood -= center

        # Append to the list of neighborhoods
        neighborhoods.append(neighborhood)

    # Stack the list of neighborhoods into a tensor
    neighborhoods = torch.stack(neighborhoods)

    return neighborhoods


# def changeOnePointToNoise(neighborhood):
#     batch_size, num_clusters, num_points, _ = neighborhood.shape
#
#     for b in range(batch_size):
#         for c in range(num_clusters):
#             # Select a random index within the cluster
#             random_idx = random.randint(0, num_points - 1)
#
#             # Change the selected point to (1, 1, 1)
#             neighborhood[b, c, random_idx] = torch.tensor([1, 1, 1], device=neighborhood.device)
#
#     return neighborhood

def changeOnePointToNoise(neighborhood, offset=0.15):
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

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, labels=None):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''

        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3
        # print(f' center shape in group": {center.shape}')
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        # print(f'xyz shape is: {xyz.shape}')
        # print(f'in Group, idx shape is {idx.size()}') # idx indicates what points of the input to add to each token
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx_groups = idx.clone()
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        if labels is not None:
            labels = labels.view(batch_size * num_points, -1)[idx, :]
            labels = labels.view(batch_size, self.num_group, self.group_size, 1).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        if labels is not None:
            return neighborhood, center, labels, idx_groups
        else:
            return neighborhood, center


class Group_SOR(nn.Module):  # SOR + FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, labels=None):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''

        batch_size, num_points, _ = xyz.shape
        # sor + fps the centers out
        # center = misc.fps(xyz, self.num_group) # B G 3
        center = misc.hybrid_sampling(xyz, self.num_group, self.group_size).cuda()
        # print(f' center shape in group": {center.shape}')
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        # print(f'xyz shape is: {xyz.shape}')
        # print(f'in Group, idx shape is {idx.size()}') # idx indicates what points of the input to add to each token
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx_groups = idx.clone()
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        if labels is not None:
            labels = labels.view(batch_size * num_points, -1)[idx, :]
            labels = labels.view(batch_size, self.num_group, self.group_size, 1).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        if labels is not None:
            return neighborhood, center, labels, idx_groups
        else:
            return neighborhood, center

class Grid(nn.Module): # GRID + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        # center = misc.fps(xyz, self.num_group)  # B G 3
        center = create_grid(xyz=xyz, num_points=self.num_group)
        # print(f' center shape in grid": {center.shape}')
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)
        # print(f'idx.size: {idx.size()}') # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Sphere(nn.Module):  # FPS + KNN
    def __init__(self, group_size):
        super().__init__()
        self.num_group = 64
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)
        # self.volume_search = volumeSearch(radius=self.sphere_radius) # find all points in a sphere_randius around each center point
    def forward(self, xyz, labels=None, inside_mask=None):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''

        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.create_sphere_grid(spaces=0.25, dist=0.25, batch_size=batch_size)
        # center_old = misc.fps(xyz, self.num_group) # B G 3
        self.num_group = center.shape[1]
        # print(f' center shape in group": {center.shape}')
        # volume search to get the neighborhood
        # _, idx = self.volume_search(xyz, center) # B G M
        # print(f' shape of center_old: {center_old.shape}')
        _, idx = self.knn(xyz, center)
        # print(f'xyz shape is: {xyz.shape}')
        # print(f'in Group, idx shape is {idx.size()}') # idx indicates what points of the input to add to each token
        assert idx.size(1) == self.num_group
        # assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx_groups = idx.clone()
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        if labels is not None:
            labels = labels.view(batch_size * num_points, -1)[idx, :]
            labels = labels.view(batch_size, self.num_group, self.group_size, 1).contiguous()
        if inside_mask is not None:
            inside_mask = inside_mask.view(batch_size * num_points, -1)[idx, :]
            inside_mask = inside_mask.view(batch_size, self.num_group, self.group_size, 1).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        if labels is not None and inside_mask is not None:
            return neighborhood, center, labels, idx_groups, inside_mask
        elif labels is not None:
            return neighborhood, center, labels, idx_groups
        else:
            return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DifferentialAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 6, bias=qkv_bias)  # Now dim * 6 to allow for dual q and k
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable parameter lambda to balance attention maps
        self.lambda_param = nn.Parameter(torch.zeros(1))  # Initialized to 0 for balanced

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 6, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # Split qkv into two sets for differential attention
        q1, k1, v = qkv[0], qkv[1], qkv[2]
        q2, k2 = qkv[3], qkv[4]

        # Compute two different attention maps
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale

        # Apply softmax normalization
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)

        # Compute the weighted differential attention using lambda
        attn_diff = (1 + self.lambda_param) * attn1 - self.lambda_param * attn2
        attn_diff = attn_diff.softmax(dim=-1)  # Re-normalize the differential attention map
        attn_diff = self.attn_drop(attn_diff)

        # Use differential attention to weigh values
        x = (attn_diff @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, diff=False):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if diff:
            self.attn = DifferentialAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., diff=False):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                diff=diff,
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, diff=False, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
            diff=diff,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug = False, mask = None):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)
        if mask is not None:
            bool_masked_pos = mask
        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


@MODELS.register_module()
class Point_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger ='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.group_sor_divider = Group_SOR(num_group=self.num_group, group_size=self.group_size)
        self.grid_divider = Grid(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        # elif loss_type =='cdl2weighted':
        #     self.loss_func = ChamferDistanceL2Weighted().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, pts, vis=False, **kwargs):
        # neighborhood, center = self.group_divider(pts)
        # neighborhood, center = self.grid_divider(pts)
        neighborhood, center = self.group_sor_divider(pts)

        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        if vis:  # visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1

    # def forward(self, pts, vis=False, denoise=False, noise_type=None, **kwargs):
    #     # if denoising, make sure to freeze encoder weights before fine-tuning loop (before forward call)
    #     neighborhood, center = self.group_divider(pts)
    #     # print(f'center shape {center.shape}')
    #     x_vis, mask = self.MAE_encoder(neighborhood, center)
    #     B, _, C = x_vis.shape  # B VIS C
    #
    #     pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
    #
    #     pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
    #
    #     _, N, _ = pos_emd_mask.shape
    #     mask_token = self.mask_token.expand(B, N, -1)
    #     # add noise to concat at next line
    #     if denoise:
    #         # noise = noise_type.createNoise(pts.min(), pts.max(), neighborhood.size(), num_noise_points=0)
    #         # noise = noise_type.generate_points_around_center(center, 1)
    #         batch_size, num_tokens, _ = center.shape
    #         noise = torch.ones(batch_size, num_tokens, 1, 3).cuda()
    #         x_vis_noise, _ = self.MAE_encoder(noise, center, mask=mask)
    #         x_vis = x_vis + x_vis_noise
    #         x_full = torch.cat([x_vis, mask_token], dim=1)
    #         # x_full = torch.cat([x_vis, x_vis_noise, mask_token], dim=1)
    #         # print(f' mask token shape: {mask_token.shape}')
    #         # print(f'x_vis shape: {x_vis.shape}')
    #         # print(f'x_full shape: {x_full.shape}')
    #     else:
    #         x_full = torch.cat([x_vis, mask_token], dim=1)
    #     pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
    #     # print(f' pos_full shape: {pos_full.shape}')
    #     x_rec = self.MAE_decoder(x_full, pos_full, N)
    #
    #     B, M, C = x_rec.shape
    #     rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
    #     torch.set_printoptions(threshold=10000)
    #     gt_points = neighborhood[mask].reshape(B * M, -1, 3)
    #     loss1 = self.loss_func(rebuild_points, gt_points)
    #     # print(f'gt : {find_largest_point(gt_points)}, rebuild: {find_largest_point(rebuild_points)}')
    #
    #     if vis:  # visualization
    #         vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
    #         full_vis = vis_points + center[~mask].unsqueeze(1)
    #         full_rebuild = rebuild_points + center[mask].unsqueeze(1)
    #         # full = torch.cat([full_vis, full_rebuild], dim=0) # temporarily disabled
    #         full = torch.cat([full_rebuild], dim=0)  # temporarily added
    #         # full_points = torch.cat([rebuild_points,vis_points], dim=0)
    #         full_center = torch.cat([center[mask], center[~mask]], dim=0)
    #         # full = full_points + full_center.unsqueeze(1)
    #         ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
    #         ret1 = full.reshape(-1, 3).unsqueeze(0)
    #         # return ret1, ret2
    #         if denoise:
    #             return ret1, ret2, full_center, noise
    #         else:
    #             return ret1, ret2, full_center
    #         # ret1 = reconstructed points (on the left picture)
    #         # ret2 = all original points (on the right picture)
    #         # full_center = all center points
    #     else:
    #         vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
    #         full_vis = vis_points + center[~mask].unsqueeze(1)
    #         full_rebuild = rebuild_points + center[mask].unsqueeze(1)
    #         # full = torch.cat([full_vis, full_rebuild], dim=0) # temporarily disabled
    #         full = torch.cat([full_rebuild], dim=0)  # temporarily added
    #         # full_points = torch.cat([rebuild_points,vis_points], dim=0)
    #         full_center = torch.cat([center[mask], center[~mask]], dim=0)
    #         # full = full_points + full_center.unsqueeze(1)
    #         gt = full_vis.reshape(-1, 3).unsqueeze(0)
    #         reconstruction = full.reshape(-1, 3).unsqueeze(0)
    #
    #         return loss1, gt, full_center, reconstruction
            # return loss1

    # def forward(self, pts, vis=False, denoise=False, **kwargs):
    #     # print(pts.shape)
    #     neighborhood, center = self.group_divider(pts)
    #     # neighborhood, center = self.grid_divider(pts)
    #     print(f'neighborhood shape: {neighborhood.shape}')
    #     if denoise:
    #         neighborhood_noise = neighborhood.clone()
    #         neighborhood_noise = changeOnePointToNoise(neighborhood_noise)
    #         x_vis, mask = self.MAE_encoder(neighborhood_noise, center)
    #     else:
    #         x_vis, mask = self.MAE_encoder(neighborhood, center)
    #     B, _, C = x_vis.shape  # B VIS C
    #
    #     print(f'x_vis.shape: {x_vis.shape}')
    #     print(f'mask.shape: {mask.shape}')
    #
    #     pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
    #
    #     pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
    #
    #     _, N, _ = pos_emd_mask.shape
    #     mask_token = self.mask_token.expand(B, N, -1)
    #     x_full = torch.cat([x_vis, mask_token], dim=1)
    #     pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
    #
    #     x_rec = self.MAE_decoder(x_full, pos_full, N)
    #
    #     B, M, C = x_rec.shape
    #     rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
    #
    #     gt_points = neighborhood[mask].reshape(B * M, -1, 3)
    #     loss1 = self.loss_func(rebuild_points, gt_points)
    #
    #     if vis:  # visualization
    #         if denoise:
    #             vis_points = neighborhood_noise[~mask].reshape(B * (self.num_group - M), -1, 3)
    #         else:
    #             vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
    #         full_vis = vis_points + center[~mask].unsqueeze(1)
    #         full_rebuild = rebuild_points + center[mask].unsqueeze(1)
    #         # full = torch.cat([full_vis, full_rebuild], dim=0)
    #         full = torch.cat([full_rebuild], dim=0)
    #         # full_points = torch.cat([rebuild_points,vis_points], dim=0)
    #         full_center = torch.cat([center[mask], center[~mask]], dim=0)
    #         # full = full_points + full_center.unsqueeze(1)
    #         ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
    #         ret1 = full.reshape(-1, 3).unsqueeze(0)
    #         # return ret1, ret2
    #         return ret1, ret2, full_center
    #     else:
    #         if denoise:
    #             vis_points = neighborhood_noise[~mask].reshape(B * (self.num_group - M), -1, 3)
    #         else:
    #             vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
    #         full_vis = vis_points + center[~mask].unsqueeze(1)
    #
    #         gt_clean = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
    #         full_gt_clean = gt_clean + center[~mask].unsqueeze(1)
    #         full_gt_clean = full_gt_clean.reshape(-1, 3).unsqueeze(0)
    #
    #         full_rebuild = rebuild_points + center[mask].unsqueeze(1)
    #         # full = torch.cat([full_vis, full_rebuild], dim=0) # temporarily disabled
    #         full = torch.cat([full_rebuild], dim=0)  # temporarily added
    #         # full_points = torch.cat([rebuild_points,vis_points], dim=0)
    #         # full_center = torch.cat([center[mask], center[~mask]], dim=0)
    #         # full = full_points + full_center.unsqueeze(1)
    #         gt_noise = full_vis.reshape(-1, 3).unsqueeze(0)
    #         reconstruction = full.reshape(-1, 3).unsqueeze(0)
    #
    #         return loss1, gt_noise, full_gt_clean, reconstruction

# denoiser classification model

@MODELS.register_module()
class Point_Denoiser(nn.Module):
    def __init__(self, config, encoder, load_weights=False):
        super().__init__()
        print_log(f'[Point_Denoiser] ', logger ='Point_Denoiser')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        # self.MAE_encoder = MaskTransformer(config)
        self.MAE_encoder = encoder
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate

        # print_log(f'[Point_Denoiser] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_Denoiser')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # Classification head: a simple linear layer that outputs logits for GT vs noise classification
        # self.classification_head = nn.Sequential(
        #     nn.Linear(self.trans_dim, 256),  # Input size is 384
        #     nn.ReLU(),
        #     nn.Linear(256, 256),  # Added an intermediate layer with 128 units
        #     nn.ReLU(),
        #     nn.Linear(256, 32 * 1),  # Output 32 values per point, reshaped to [32, 1] for binary classification
        # )
        self.classification_head = nn.Sequential(
                nn.Linear(self.trans_dim, 256),
                nn.BatchNorm1d(26),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(26),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.5),
                nn.Linear(256, 32 * 1)
            )

        # self._init_classification_weighs()
        # Define the loss function for classification (Binary Cross Entropy Loss)
        # pos_weight_value = torch.tensor(992 / 32)  # negative / positive
        self.loss_func = nn.BCEWithLogitsLoss()

        # Load weights to the classification head if a path is provided
        if load_weights is True:
            print('trying to load classification weights...')
            self.load_classification_head_weights('checkpoints/checkpoint_epoch_300.pth')
        else:
            self._init_classification_weighs()


    def load_classification_head_weights(self, weights_path):
        """
        Loads weights into the classification head.
        :param weights_path: path to the .pth file containing the saved weights.
        """
        try:
            checkpoint = torch.load(weights_path, map_location=torch.device('cuda'))
            # Filter only the classification head weights
            classification_checkpoint = {k: v for k, v in checkpoint['model_state_dict'].items() if 'classification_head' in k}
            # Assuming the checkpoint is a state_dict for classification head only
            self.classification_head.load_state_dict(classification_checkpoint, strict=False)
            print(f"Weights loaded successfully from {weights_path}")

        except Exception as e:
            print(f"Failed to load weights from {weights_path}: {e}")

    def _init_classification_weighs(self):
        # Loop through the layers and apply trunc_normal_ to the Linear layers
        for m in self.classification_head:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)  # Initialize weights
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # Initialize bias to 0

    def forward(self, pts, labels=None, vis=False, **kwargs):
        """
        Forward pass for classification:
        - pts: input point cloud (with GT and noise points)
        - labels: binary labels (0 for noise, 1 for GT)
        """
        # Divide the point cloud into neighborhoods and centers
        if vis:
            neighborhood, center, labels, idx = self.group_divider(pts, labels)
        elif labels is not None:
            neighborhood, center, labels, idx = self.group_divider(pts, labels)
        else:
            neighborhood, center = self.group_divider(pts)
        # print(f'neighborhood shape is: {neighborhood.shape}')
        # Pass through the encoder to get latent tokens
        x_vis, mask = self.MAE_encoder(neighborhood, center)

        N = 4  # The layer index you want to extract
        latent_space = get_intermediate_output(self.MAE_encoder, neighborhood, center, N)
        # print(f'latent_space shape: {latent_space.shape}')
        # Unsqueeze the mask to match the dimensions of labels
        expanded_mask = mask.unsqueeze(-1).unsqueeze(-1)  # Shape: [128, 64, 1, 1]
        # Classification head: apply it to the encoder's output (latent tokens)
        classification_logits = self.classification_head(x_vis)  # Output shape: (128, 26, 32)

        if labels is not None and vis is False:
            # Apply the mask to filter the labels
            labels = labels[~expanded_mask.expand_as(labels)].view(128, -1, 32, 1)
            # Unsqueeze the mask to match the dimensions of idx
            expanded_mask = mask.unsqueeze(-1)  # Shape: [128, 64, 1]
            # Apply the expanded mask to filter idx along the second dimension
            vis_idx = idx[~expanded_mask.expand_as(idx)].view(128, -1, 32)
            # print(f'filtered_idx shape is: {filtered_idx.shape}')  # Output shape will be [128, X, 32], with X varying per batch
            # save_latent_pic(x_vis, labels)
            # save_latent_pic(latent_space, labels)

        if vis:
            classification_logits = classification_logits.view(1, 26, 32, 1)  # Reshape to match labels' shape [128, 26, 32, 1]
            probabilities = torch.sigmoid(classification_logits) # Step 1: Apply sigmoid to get probabilities
            binary_output = (probabilities >= 0.5).float() # Step 2: Convert probabilities to 0 or 1 using threshold of 0.5
            neighborhood_vis = neighborhood[~mask]
            neighborhood_vis = neighborhood_vis + + center[~mask].unsqueeze(1) # add center points back
            # Calculate prediction loss
            # Unsqueeze the mask to match the dimensions of idx
            # Negate the mask and expand its last dimension to match idx's shape
            mask_expanded = (~mask).unsqueeze(-1).expand(-1, -1, idx.size(-1))

            # Apply the expanded mask to idx
            vis_labels = labels[mask_expanded].view(1, -1, idx.size(-1))

            num_noise_points = (vis_labels == 1).sum()
            print(f"Number of noise points (1 values) in vis_labels: {num_noise_points.item()}")
            num_clean_points = (vis_labels == 0).sum()
            print(f"Number of clean points (0 values) in vis_labels: {num_clean_points.item()}")
            pred_noise_points = (binary_output == 1).sum()
            print(f"Number of predicted noise points (1 values) in vis_labels: {pred_noise_points.item()}")

            correct = (vis_labels == binary_output.squeeze(-1).float()).float()  # check if predictions match labels
            accuracy = correct.mean()  # compute mean accuracy
            print(f'Accuracy: {accuracy}')

            filtered_points = neighborhood_vis[(binary_output.squeeze(0) == 0).squeeze(-1)]
            filtered_points = torch.unique(filtered_points, dim=0) # Filter out duplicate points
            # save_latent_pic(latent_space, labels)

            # Reshape vis_labels and binary_output to match for comparison
            vis_labels_flat = vis_labels.view(-1)  # Flatten vis_labels
            binary_output_flat = binary_output.squeeze(-1).view(-1)  # Flatten binary_output

            # Calculate confusion matrix components
            TP = ((binary_output_flat == 1) & (vis_labels_flat == 1)).sum().item()
            TN = ((binary_output_flat == 0) & (vis_labels_flat == 0)).sum().item()
            FP = ((binary_output_flat == 1) & (vis_labels_flat == 0)).sum().item()
            FN = ((binary_output_flat == 0) & (vis_labels_flat == 1)).sum().item()

            # Print confusion matrix
            print("Confusion Matrix:")
            print(f"True Positives (TP): {TP}")
            print(f"True Negatives (TN): {TN}")
            print(f"False Positives (FP): {FP}")
            print(f"False Negatives (FN): {FN}")

            return filtered_points, center, accuracy
        else:
            classification_logits = classification_logits.view(128, 26, 32, 1)  # Reshape to match labels' shape [128, 26, 32, 1]
            # Calculate Binary Cross Entropy loss for classification
            loss = self.loss_func(classification_logits.view(-1), labels.view(-1).float())  # Flatten logits and labels for loss computation
            # also calculate classification accuracy during training
            predictions = torch.sigmoid(classification_logits.view(-1))  # apply sigmoid to get probabilities
            predicted_labels = (predictions > 0.5).float()  # threshold at 0.5 to get binary labels
            correct = (predicted_labels == labels.view(-1).float()).float()  # check if predictions match labels
            accuracy = correct.mean()  # compute mean accuracy
            print(f'Loss: {loss.item()}, Accuracy: {accuracy.item()}')
            # Open the file in append mode
            with open('accuracy.txt', 'a') as file:
                file.write(f"{accuracy.item()}\n")  # Append the float value to the file
            return loss

# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.sphere_divider = Sphere(group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, vis=False):

        # neighborhood, center = self.group_divider(pts)
        neighborhood, center = self.sphere_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret

@MODELS.register_module()
class Point_Diff(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_Diff] ', logger='Point_Diff')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config, diff=True)
        # self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group

        print_log(f'[Point_Diff] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_Diff')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.classification_head = nn.Sequential(
            nn.Linear(self.trans_dim, 256),
            nn.BatchNorm1d(26),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(26),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(256, 32 * 1)
        )

        # loss
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, pts, labels=None, vis=False, **kwargs):
        # Divide the point cloud into neighborhoods and centers
        if vis:
            neighborhood, center, labels, idx = self.group_divider(pts, labels)
        elif labels is not None:
            neighborhood, center, labels, idx = self.group_divider(pts, labels)
        else:
            neighborhood, center = self.group_divider(pts)

        x_vis, mask = self.MAE_encoder(neighborhood, center)

        expanded_mask = mask.unsqueeze(-1).unsqueeze(-1)  # Shape: [128, 64, 1, 1]
        # Classification head: apply it to the encoder's output (latent tokens)
        classification_logits = self.classification_head(x_vis)  # Output shape: (128, 26, 32)


        if labels is not None and vis is False:
            # Apply the mask to filter the labels
            labels = labels[~expanded_mask.expand_as(labels)].view(128, -1, 32, 1)
            # Unsqueeze the mask to match the dimensions of idx
            expanded_mask = mask.unsqueeze(-1)  # Shape: [128, 64, 1]
            # Apply the expanded mask to filter idx along the second dimension
            vis_idx = idx[~expanded_mask.expand_as(idx)].view(128, -1, 32)
            # print(f'filtered_idx shape is: {filtered_idx.shape}')  # Output shape will be [128, X, 32], with X varying per batch
            # save_latent_pic(x_vis, labels)

        if vis:
            classification_logits = classification_logits.view(1, 26, 32, 1)  # Reshape to match labels' shape [128, 26, 32, 1]
            probabilities = torch.sigmoid(classification_logits) # Step 1: Apply sigmoid to get probabilities
            binary_output = (probabilities >= 0.5).float() # Step 2: Convert probabilities to 0 or 1 using threshold of 0.5
            neighborhood_vis = neighborhood[~mask]
            neighborhood_vis = neighborhood_vis + + center[~mask].unsqueeze(1) # add center points back
            mask_expanded = (~mask).unsqueeze(-1).expand(-1, -1, idx.size(-1))
            vis_labels = labels[mask_expanded].view(1, -1, idx.size(-1))

            num_noise_points = (vis_labels == 1).sum()
            print(f"Number of noise points (1 values) in vis_labels: {num_noise_points.item()}")
            num_clean_points = (vis_labels == 0).sum()
            print(f"Number of clean points (0 values) in vis_labels: {num_clean_points.item()}")
            pred_noise_points = (binary_output == 1).sum()
            print(f"Number of predicted noise points (1 values) in vis_labels: {pred_noise_points.item()}")

            correct = (vis_labels == binary_output.squeeze(-1).float()).float()  # check if predictions match labels
            accuracy = correct.mean()  # compute mean accuracy
            print(f'Accuracy: {accuracy}')

            print(f'neighborhood_vis shape before filtering: {neighborhood_vis.shape}')
            filtered_points = neighborhood_vis[(binary_output.squeeze(0) == 0).squeeze(-1)]

            print(f'neighborhood_vis shape after filtering: {filtered_points.shape}')
            filtered_points = torch.unique(filtered_points, dim=0) # Filter out duplicate points

            # Reshape vis_labels and binary_output to match for comparison
            vis_labels_flat = vis_labels.view(-1)  # Flatten vis_labels
            binary_output_flat = binary_output.squeeze(-1).view(-1)  # Flatten binary_output
            # Calculate confusion matrix components
            TP = ((binary_output_flat == 1) & (vis_labels_flat == 1)).sum().item()
            TN = ((binary_output_flat == 0) & (vis_labels_flat == 0)).sum().item()
            FP = ((binary_output_flat == 1) & (vis_labels_flat == 0)).sum().item()
            FN = ((binary_output_flat == 0) & (vis_labels_flat == 1)).sum().item()

            # Print confusion matrix
            print("Confusion Matrix:")
            print(f"True Positives (TP): {TP}")
            print(f"True Negatives (TN): {TN}")
            print(f"False Positives (FP): {FP}")
            print(f"False Negatives (FN): {FN}")

            return filtered_points, center
        else:
            classification_logits = classification_logits.view(128, 26, 32, 1)  # Reshape to match labels' shape [128, 26, 32, 1]
            # Calculate Binary Cross Entropy loss for classification
            loss = self.loss_func(classification_logits.view(-1), labels.view(-1).float())  # Flatten logits and labels for loss computation
            # also calculate classification accuracy during training
            predictions = torch.sigmoid(classification_logits.view(-1))  # apply sigmoid to get probabilities
            predicted_labels = (predictions > 0.5).float()  # threshold at 0.5 to get binary labels
            correct = (predicted_labels == labels.view(-1).float()).float()  # check if predictions match labels
            accuracy = correct.mean()  # compute mean accuracy
            print(f'Loss: {loss.item()}, Accuracy: {accuracy.item()}')
            # Open the file in append mode
            with open('accuracy.txt', 'a') as file:
                file.write(f"{accuracy.item()}\n")  # Append the float value to the file
            return loss



@MODELS.register_module()
class Point_ViT(nn.Module):
    def __init__(self, config, load_weights=False):
        super().__init__()
        print_log(f'[Point_ViT] ', logger ='Point_ViT')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        # self.MAE_encoder = encoder
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate

        # print_log(f'[Point_Denoiser] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_Denoiser')
        # self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.sphere_divider = Sphere(group_size=self.group_size)
        # Classification head: a simple linear layer that outputs logits for GT vs noise classification
        # self.classification_head = nn.Sequential(
        #     nn.Linear(self.trans_dim, 256),  # Input size is 384
        #     nn.ReLU(),
        #     nn.Linear(256, 256),  # Added an intermediate layer with 128 units
        #     nn.ReLU(),
        #     nn.Linear(256, 32 * 1),  # Output 32 values per point, reshaped to [32, 1] for binary classification
        # )
        self.classification_head = nn.Sequential(
                nn.Linear(self.trans_dim, 256),
                nn.BatchNorm1d(151),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(151),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.5),
                nn.Linear(256, 32 * 1)
            )

        # self._init_classification_weighs()
        # Define the loss function for classification (Binary Cross Entropy Loss)
        # pos_weight_value = torch.tensor(992 / 32)  # negative / positive
        self.loss_func = nn.BCEWithLogitsLoss()

        # Load weights to the classification head if a path is provided
        if load_weights is True:
            print('trying to load classification weights...')
            self.load_classification_head_weights('checkpoints/checkpoint_epoch_301.pth')
        else:
            self._init_classification_weighs()


    def load_classification_head_weights(self, weights_path):
        """
        Loads weights into the classification head.
        :param weights_path: path to the .pth file containing the saved weights.
        """
        try:
            checkpoint = torch.load(weights_path, map_location=torch.device('cuda'))
            # Filter only the classification head weights
            classification_checkpoint = {k: v for k, v in checkpoint['model_state_dict'].items() if 'classification_head' in k}
            # Assuming the checkpoint is a state_dict for classification head only
            self.classification_head.load_state_dict(classification_checkpoint, strict=False)
            print(f"Weights loaded successfully from {weights_path}")

        except Exception as e:
            print(f"Failed to load weights from {weights_path}: {e}")

    def _init_classification_weighs(self):
        # Loop through the layers and apply trunc_normal_ to the Linear layers
        for m in self.classification_head:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)  # Initialize weights
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # Initialize bias to 0

    def forward(self, pts, labels=None, vis=False, **kwargs):
        """
        Forward pass for classification:
        - pts: input point cloud (with GT and noise points)
        - labels: binary labels (0 for noise, 1 for GT)
        """
        # Divide the point cloud into neighborhoods and centers
        if vis:
            neighborhood, center, labels, idx = self.sphere_divider(pts, labels)
        elif labels is not None:
            neighborhood, center, labels, idx = self.sphere_divider(pts, labels) # for training
        else:
            neighborhood, center = self.sphere_divider(pts)
        # print(f'neighborhood shape is: {neighborhood.shape}')
        # Pass through the encoder to get latent tokens
        x_vis, mask = self.MAE_encoder(neighborhood, center)

        N = 4  # The layer index you want to extract
        # latent_space = get_intermediate_output(self.MAE_encoder, neighborhood, center, N)
        # print(f'latent_space shape: {latent_space.shape}')
        # Unsqueeze the mask to match the dimensions of labels
        expanded_mask = mask.unsqueeze(-1).unsqueeze(-1)  # Shape: [128, 64, 1, 1]
        # Classification head: apply it to the encoder's output (latent tokens)
        classification_logits = self.classification_head(x_vis)  # Output shape: (128, 26, 32)

        if labels is not None and vis is False:
            # Apply the mask to filter the labels
            labels = labels[~expanded_mask.expand_as(labels)].view(128, -1, 32, 1)
            # Unsqueeze the mask to match the dimensions of idx
            expanded_mask = mask.unsqueeze(-1)  # Shape: [128, 64, 1]
            # Apply the expanded mask to filter idx along the second dimension
            vis_idx = idx[~expanded_mask.expand_as(idx)].view(128, -1, 32)
            # print(f'filtered_idx shape is: {filtered_idx.shape}')  # Output shape will be [128, X, 32], with X varying per batch
            # save_latent_pic(x_vis, labels)
            # save_latent_pic(latent_space, labels)

        if vis:
            classification_logits = classification_logits.view(1, 151, 32, 1)  # Reshape to match labels' shape [128, 26, 32, 1]
            probabilities = torch.sigmoid(classification_logits) # Step 1: Apply sigmoid to get probabilities
            binary_output = (probabilities >= 0.5).float() # Step 2: Convert probabilities to 0 or 1 using threshold of 0.5
            neighborhood_vis = neighborhood[~mask]
            neighborhood_vis = neighborhood_vis + + center[~mask].unsqueeze(1) # add center points back
            # Calculate prediction loss
            # Unsqueeze the mask to match the dimensions of idx
            # Negate the mask and expand its last dimension to match idx's shape
            mask_expanded = (~mask).unsqueeze(-1).expand(-1, -1, idx.size(-1))

            # Apply the expanded mask to idx
            vis_labels = labels[mask_expanded].view(1, -1, idx.size(-1))

            num_noise_points = (vis_labels == 1).sum()
            print(f"Number of noise points (1 values) in vis_labels: {num_noise_points.item()}")
            num_clean_points = (vis_labels == 0).sum()
            print(f"Number of clean points (0 values) in vis_labels: {num_clean_points.item()}")
            pred_noise_points = (binary_output == 1).sum()
            print(f"Number of predicted noise points (1 values) in vis_labels: {pred_noise_points.item()}")

            correct = (vis_labels == binary_output.squeeze(-1).float()).float()  # check if predictions match labels
            accuracy = correct.mean()  # compute mean accuracy
            print(f'Accuracy: {accuracy}')

            filtered_points = neighborhood_vis[(binary_output.squeeze(0) == 0).squeeze(-1)]
            filtered_points = torch.unique(filtered_points, dim=0) # Filter out duplicate points
            # save_latent_pic(latent_space, labels)

            # Reshape vis_labels and binary_output to match for comparison
            vis_labels_flat = vis_labels.view(-1)  # Flatten vis_labels
            binary_output_flat = binary_output.squeeze(-1).view(-1)  # Flatten binary_output

            # Calculate confusion matrix components
            TP = ((binary_output_flat == 1) & (vis_labels_flat == 1)).sum().item()
            TN = ((binary_output_flat == 0) & (vis_labels_flat == 0)).sum().item()
            FP = ((binary_output_flat == 1) & (vis_labels_flat == 0)).sum().item()
            FN = ((binary_output_flat == 0) & (vis_labels_flat == 1)).sum().item()

            # Print confusion matrix
            print("Confusion Matrix:")
            print(f"True Positives (TP): {TP}")
            print(f"True Negatives (TN): {TN}")
            print(f"False Positives (FP): {FP}")
            print(f"False Negatives (FN): {FN}")

            return filtered_points, center, accuracy
        else:
            classification_logits = classification_logits.view(128, 151, 32, 1)  # Reshape to match labels' shape [128, 26, 32, 1]
            # Calculate Binary Cross Entropy loss for classification
            loss = self.loss_func(classification_logits.view(-1), labels.view(-1).float())  # Flatten logits and labels for loss computation
            # also calculate classification accuracy during training
            predictions = torch.sigmoid(classification_logits.view(-1))  # apply sigmoid to get probabilities
            predicted_labels = (predictions > 0.5).float()  # threshold at 0.5 to get binary labels
            correct = (predicted_labels == labels.view(-1).float()).float()  # check if predictions match labels
            accuracy = correct.mean()  # compute mean accuracy
            print(f'Loss: {loss.item()}, Accuracy: {accuracy.item()}')
            # Open the file in append mode
            with open('accuracy.txt', 'a') as file:
                file.write(f"{accuracy.item()}\n")  # Append the float value to the file
            return loss

@MODELS.register_module()
class Point_MAE_ViT(nn.Module): # This class is for pre-training the Sphere Divider visual transformer
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE_ViT] ', logger ='Point_MAE_ViT')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE_ViT] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE_ViT')
        # self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.sphere_divider = Sphere(group_size=self.group_size)
        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, pts, labels=None, vis=False, **kwargs):
        """
        Forward pass for classification:
        - pts: input point cloud (with GT and noise points)
        - labels: binary labels (0 for noise, 1 for GT)
        """
        # Divide the point cloud into neighborhoods and centers
        if vis:
            neighborhood, center, labels, idx = self.sphere_divider(pts, labels)
        elif labels is not None:
            neighborhood, center, labels, idx = self.sphere_divider(pts, labels) # for training
        else:
            neighborhood, center = self.sphere_divider(pts)
        # print(f'center shape: {center.shape}')
        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B,_,C = x_vis.shape # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B*M,-1,3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        if vis: #visualization
            vis_points = neighborhood[~mask].reshape(B * (175 - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0) # gt_noise
            ret1 = full.reshape(-1, 3).unsqueeze(0) # reconstruction
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            # print(gt_points.shape)
            # print(rebuild_points.shape)
            return loss1
            # return loss1, gt_points.view(-1, 3), rebuild_points.view(-1, 3)


class DenoisingLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.05, lambda_dist=5.0):
        """
        Loss function combining:
        - BCE loss for classification
        - Chamfer Distance for regularization
        - Exponential distance penalty for far-away noise
        """
        super(DenoisingLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cdl2_loss = ChamferDistanceL2().cuda()
        self.alpha = alpha
        self.beta = beta
        self.lambda_dist = lambda_dist

    def forward(self, predictions, labels, pred_clean_points, shape_points):
        """
        Args:
            predictions: Tensor of shape [B, N, 1] (logits for classification).
            labels: Tensor of shape [B, N, 1] (ground truth, 0 for shape, 1 for noise).
            pred_clean_points: Tensor of shape [B, N, 3] (predicted clean points).
            shape_points: Tensor of shape [B, M, 3] (clean shape points).
        Returns:
            Total loss (BCE + Chamfer Distance + Distance Penalty)
        """

        assert shape_points.shape[1] > 0, "Shape points are empty!"
        assert pred_clean_points.shape[1] > 0, "Noise points are empty!"
        if torch.isnan(pred_clean_points).any() or torch.isinf(pred_clean_points).any():
            print("NaN or Inf detected in noise_points")
        if torch.isnan(shape_points).any() or torch.isinf(shape_points).any():
            print("NaN or Inf detected in shape_points")
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            print("NaN or Inf detected in labels")


        # BCE Loss
        bce = self.bce_loss(predictions, labels)

        # Chamfer Distance Loss (returns per-point distances)
        chamferDist = self.cdl2_loss(pred_clean_points, shape_points)
        # print(f'dist1: {dist1}')
        # meanFarthestDist = farthest_squared_distance(src=shape_points, tgt=pred_clean_points)
        # maenDist = statistical_outlier_std(xyz=pred_clean_points)
        outlier_loss = compute_outlier_loss(xyz=pred_clean_points)
        print(f'outlier_loss: {outlier_loss}')
        chamfer_loss = torch.mean(chamferDist * labels.squeeze(-1))
        print(f'chamfer_loss: {chamfer_loss}')
        # dist1 = dist1.mean()  # Now shape is [128, 70, 32]
        # Apply only to noise points (where label=1)

        # Exponential Distance Penalty (ensure exp(x) does not explode to inf)
        # exp_penalty = torch.mean(labels.squeeze(-1) * torch.exp(torch.clamp(self.lambda_dist * meanFarthestDist, max=10)))
        # Total Loss
        # total_loss = bce + self.alpha * chamfer_loss + self.beta * exp_penalty
        # print(f'exp_penalty: {exp_penalty * self.beta}')
        # print(f'dist1: {chamferDist}')
        # print(f'farthestDist: {meanFarthestDist}')
        total_loss = bce + outlier_loss #  + self.beta * (1/stdDist)
        return total_loss


@MODELS.register_module()
class Point_Denoiser_Full(nn.Module):
    def __init__(self, config, load_weights=False):
        super().__init__()
        print_log(f'[Point_Denoiser_Full] ', logger ='Point_Denoiser_Full')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_Denoiser_Full] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_Denoiser_Full')
        # self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.sphere_divider = Sphere(group_size=self.group_size)

        self.classification_head = nn.Sequential(
                nn.Linear(self.trans_dim, 256),
                # nn.BatchNorm1d(70),
                # nn.ReLU(inplace=True),
                # # nn.Dropout(0.05),
                # nn.Linear(256, 256),
                # nn.BatchNorm1d(70),
                # nn.ReLU(inplace=True),
                # nn.Dropout(0.05),
                nn.Linear(256, self.group_size * 1)
            )

        # Define the loss function for classification (Binary Cross Entropy Loss)
        # Number of non-noise and noise points
        num_non_noise = 1024
        num_noise = 700

        # Compute pos_weight: it should be (num_non_noise / num_noise)
        # pos_weight = torch.tensor(num_non_noise / num_noise)
        self.loss_func = nn.BCEWithLogitsLoss()
        # self.loss_func = DenoisingLoss()

        # Load weights to the classification head if a path is provided
        if load_weights is True:
            print('trying to load classification weights...')
            self.load_classification_head_weights('checkpoints/checkpoint_epoch_300.pth')
        else:
            print('initializing classification weights')
            self._init_classification_weighs()


    def load_classification_head_weights(self, weights_path):
        """
        Loads weights into the classification head.
        :param weights_path: path to the .pth file containing the saved weights.
        """
        try:
            checkpoint = torch.load(weights_path, map_location=torch.device('cuda'))
            # Filter only the classification head weights
            classification_checkpoint = {k: v for k, v in checkpoint['model_state_dict'].items() if 'classification_head' in k}
            # Assuming the checkpoint is a state_dict for classification head only
            self.classification_head.load_state_dict(classification_checkpoint, strict=False)
            print(f"Weights loaded successfully from {weights_path}")

        except Exception as e:
            print(f"Failed to load weights from {weights_path}: {e}")

    def _init_classification_weighs(self):
        # Loop through the layers and apply trunc_normal_ to the Linear layers
        for m in self.classification_head:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)  # Initialize weights
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # Initialize bias to 0

    def forward(self, pts, labels=None, inside_mask=None, vis=False, **kwargs):
        """
        Forward pass for classification:
        - pts: input point cloud (with GT and noise points)
        - labels: binary labels (0 for noise, 1 for GT)
        """
        # Divide the point cloud into neighborhoods and centers
        # plot_neighborhood_lables(pts, labels, filename='before_sphere_divider.png')
        # if vis:
        #     neighborhood, center, labels, idx = self.sphere_divider(xyz=pts, labels=labels)
        if vis and labels is not None and inside_mask is not None:
            neighborhood, center, labels, idx, inside_mask = self.sphere_divider(xyz=pts, labels=labels, inside_mask=inside_mask)  #  inside_mask True means inside
        elif vis and labels is not None and inside_mask is None:
            neighborhood, center, labels, idx = self.sphere_divider(xyz=pts, labels=labels)
        elif vis is False and labels is not None:
            neighborhood, center, labels, idx = self.sphere_divider(xyz=pts, labels=labels)
        else:
            neighborhood, center = self.sphere_divider(pts)

        # Pass through the encoder to get latent tokens
        x_vis, mask = self.MAE_encoder(neighborhood, center)


        # plot_neighborhood_lables(neighborhood + center.unsqueeze(2), labels, filename='after_sphere_divider.png')
        N = 4  # The layer index you want to extract
        # latent_space = get_intermediate_output(self.MAE_encoder, neighborhood, center, N)
        # Unsqueeze the mask to match the dimensions of labels
        expanded_mask = mask.unsqueeze(-1).unsqueeze(-1)  # Shape: [128, 64, 1, 1]
        # Add positional embeddings the vis embeddings before passing to classification head

        B, _, C = x_vis.shape  # B VIS C
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        # x_vis = x_vis + pos_emd_vis
        # Classification head: apply it to the encoder's output (latent tokens)
        classification_logits = self.classification_head(x_vis)  # Output shape: (128, 26, 32)

        if labels is not None and vis is False:
            # Apply the mask to filter the labels
            labels = labels[~expanded_mask.expand_as(labels)].view(128, -1, self.group_size, 1)
            expanded_mask = mask.unsqueeze(-1)  # Unsqueeze the mask to match the dimensions of idx Shape: [128, 64, 1]

        if vis:
            # neighborhood_mask = filter_neighborhood_points(neighborhood=neighborhood, center=center, cell_radius=0.25 * 0.708) #  * 0.708 # output is torch.Size([1, 175, 1]), # True means keep!!
            classification_logits = classification_logits.view(1, 70, self.group_size, 1)  # Reshape to match labels' shape [128, 26, 32, 1]
            # print(f'classification_logits probabilities and binary_output shape: {classification_logits.shape}')
            probabilities = torch.sigmoid(classification_logits) # Step 1: Apply sigmoid to get probabilities
            binary_output = (probabilities >= 0.5).float() # Step 2: Convert probabilities to 0 or 1 using threshold of 0.5, binary_output == 1 → Predicted as Noise

            # filtered_mask = neighborhood_mask[~mask].squeeze(-1).unsqueeze(0).unsqueeze(-1)  # Shape: [1, 70, 32, 1]
            # binary_output = ((binary_output == 1) | (~filtered_mask)).float()

            neighborhood_vis = neighborhood[~mask]
            neighborhood_vis = neighborhood_vis + center[~mask].unsqueeze(1) # add center points back
            # Calculate prediction loss
            # Unsqueeze the mask to match the dimensions of idx
            # Negate the mask and expand its last dimension to match idx's shape
            mask_expanded = (~mask).unsqueeze(-1).expand(-1, -1, idx.size(-1))
            vis_labels = labels[mask_expanded].view(1, -1, idx.size(-1))  # Apply the expanded mask to idx

            num_noise_points = (torch.unique(vis_labels, dim=0) == 1).sum()
            print(f"Number of noise points (1 values) in vis_labels: {num_noise_points.item()}")
            num_clean_points = (torch.unique(vis_labels, dim=0) == 0).sum()
            print(f"Number of clean points (0 values) in vis_labels: {num_clean_points.item()}")
            pred_noise_points = (binary_output == 1).sum()
            # print(f"Number of predicted noise points (1 values) in vis_labels: {pred_noise_points.item()}")
            # ________________________________________________________
            # calculate prediction accuracy of non-inside noise points
            if inside_mask is not None:
                vis_inside_mask = inside_mask[~mask].unsqueeze(0)
                outside_mask = ~vis_inside_mask.bool()  # Shape: [1, 70, 32, 1]
                # Apply the mask to filter binary_output and vis_labels
                binary_output_out = binary_output[outside_mask]  # Shape: [N_outside]
                vis_labels_out = vis_labels[outside_mask.squeeze(-1)]  # Shape: [N_outside]
                # Calculate accuracy for points outside the shape
                correct = (vis_labels_out == binary_output_out.squeeze(-1).float()).float()
                accuracy_outside = correct.mean()
                print(f'Accuracy (outside points only): {accuracy_outside.item()}')
            # ________________________________________________________

            correct = (vis_labels == binary_output.squeeze(-1).float()).float()  # check if predictions match labels
            accuracy = correct.mean()  # compute mean accuracy
            print(f'Accuracy: {accuracy}')

            filtered_points = neighborhood_vis[(binary_output.squeeze(0) == 0).squeeze(-1)] # predicted GT points
            filtered_points = torch.unique(filtered_points, dim=0)  # Filter out duplicate points

            # save_latent_pic(latent_space, labels)

            # Reshape vis_labels and binary_output to match for comparison
            vis_labels_flat = vis_labels.view(-1)  # Flatten vis_labels
            binary_output_flat = binary_output.squeeze(-1).view(-1)  # Flatten binary_output

            # Calculate confusion matrix components
            TP = ((binary_output_flat == 1) & (vis_labels_flat == 1)).sum().item()
            TN = ((binary_output_flat == 0) & (vis_labels_flat == 0)).sum().item()
            FP = ((binary_output_flat == 1) & (vis_labels_flat == 0)).sum().item()
            FN = ((binary_output_flat == 0) & (vis_labels_flat == 1)).sum().item()

            # Print confusion matrix
            print("Confusion Matrix:")
            print(f"True Positives (TP): {TP}")
            print(f"True Negatives (TN): {TN}")
            print(f"False Positives (FP): {FP}")
            print(f"False Negatives (FN): {FN}")
            cdL2 = ChamferDistanceL2().cuda()
            neighborhood_vis = neighborhood_vis.cuda()
            filtered_points = filtered_points.cuda()

            # Calculate prediction Chamfer Distance
            filtered_points = neighborhood_vis[(binary_output.squeeze(0) == 0).squeeze(-1)]  # predicted GT points
            filtered_points = torch.unique(filtered_points, dim=0)  # Filter out duplicate points
            filtered_GT = neighborhood_vis[(vis_labels.squeeze(0) == 0).squeeze(-1)]  # actual GT points
            filtered_GT = torch.unique(filtered_GT, dim=0)  # Filter out duplicate points
            CDL2 = cdL2(
                filtered_GT.view(-1, 3).unsqueeze(0),  # actual GT points
                filtered_points.unsqueeze(0))  # predicted GT points
            print(f'Chamfer Distance L2: {CDL2}')

            cleaned_coords = filter_logits_with_voting(neighborhood_vis, classification_logits)

            if inside_mask is not None:
                # return filtered_points, center, accuracy, accuracy_outside, TP, TN, FP, FN, CDL2, neighborhood_vis, classification_logits, vis_labels
                return cleaned_coords, center, accuracy, accuracy_outside, TP, TN, FP, FN, CDL2, neighborhood_vis, classification_logits, vis_labels
            else:
                # return filtered_points, center, accuracy, TP, TN, FP, FN, CDL2, neighborhood_vis, classification_logits, vis_labels
                return cleaned_coords, center, accuracy, TP, TN, FP, FN, CDL2, neighborhood_vis, classification_logits, vis_labels
        else:
            classification_logits = classification_logits.view(128, 70, self.group_size, 1)  # Reshape to match labels' shape [128, 26, 32, 1]
            # classification_logits = classification_logits.view(128, 70, self.group_size, 1)  # Reshape to match labels' shape [128, 26, 32, 1]
            probabilities = torch.sigmoid(classification_logits)  # Step 1: Apply sigmoid to get probabilities
            binary_output = (probabilities >= 0.5).float()  # Step 2: Convert probabilities to 0 or 1 using threshold of 0.5
            neighborhood_vis = neighborhood[~mask]
            neighborhood_vis = neighborhood_vis + center[~mask].unsqueeze(1)  # add center points back
            # get the first point cloud here
            neighborhood_first = neighborhood[0]  # Shape: [175, 32, 3]
            labels_first = labels[0]  # Shape: [70, 32, 1]
            binary_output_first = binary_output[0]  # Shape: [70, 32, 1]
            mask_first = mask[0]
            center_first = center[0]
            # Create a mask where binary_output == 0 (clean points)
            mask_plot = (binary_output_first == 0).squeeze(-1)  # predictions mask (vis only)
            # Flatten binary_output and labels to match neighborhood_vis
            neighborhood_first_vis = neighborhood_first[~mask_first]
            neighborhood_first_vis = neighborhood_first_vis + center_first[~mask_first].unsqueeze(1)  # add center points back
            # Apply mask to filter points and labels
            filtered_points = neighborhood_first_vis[mask_plot]  # predicted gt points
            filtered_labels = labels_first[mask_plot]  # true label for each point at filtered_points
            # Calculate prediction loss
            # cdl2 = ChamferDistanceL2()
            # print(f'Chamfer Distance L2 is: {cdl2(neighborhood_vis, filtered_points)}')
            #__________________________________________________________________________________________________________________________________
            # gt_points = neighborhood_vis[labels[~mask] == 0]
            labels_flat = labels.view(-1, 32, 1)  # Shape: [8960, 32, 1]
            mask_labels = (labels_flat == 0).squeeze(-1)  # Remove last dimension, shape: [8960, 32]
            GT_points = neighborhood_vis[mask_labels]  # Shape: [num_GT_points, 3]

            # print(f'classification_logits shape: {classification_logits.shape}')
            # print(f'neighborhood_vis shape: {neighborhood_vis.shape}')
            # print(f'binary_output shape: {binary_output.shape}')
            print(f'neighborhood_first_vis shape: {neighborhood_first_vis.shape}')
            print(f'mask_first.sum(): {mask_first.sum()}')

            binary_mask = (binary_output == 0).squeeze(-1).reshape(8960, 32)  # Shape: (8960, 32)
            pred_clean_points = neighborhood_vis[binary_mask]  # Select only where mask is True
            # print(f'pred_clean_points shape: {pred_clean_points.shape}')
            # print(f'GT_points shape: {GT_points.shape}')
            # print(f'labels shape: {labels.shape}')

            # loss = self.loss_func(classification_logits,
            #                       labels,
            #                       torch.unique(pred_clean_points, dim=0).unsqueeze(0),
            #                       torch.unique(GT_points, dim=0).unsqueeze(0))
            loss = self.loss_func(classification_logits.view(-1), labels.view(-1).float()) # Flatten logits and labels for loss computation
            # also calculate classification accuracy during training
            predictions = torch.sigmoid(classification_logits.view(-1))  # apply sigmoid to get probabilities
            predicted_labels = (predictions > 0.5).float()  # threshold at 0.5 to get binary labels
            correct = (predicted_labels == labels.view(-1).float()).float()  # check if predictions match labels
            accuracy = correct.mean()  # compute mean accuracy
            print(f'Loss: {loss.item()}, Accuracy: {accuracy.item()}')
            # plot_neighborhood_lables(filtered_points, filtered_labels, filename='predictions.png')
            return loss, accuracy, filtered_points, filtered_labels, neighborhood_first_vis.view(-1)
