import numpy as np
import math
import torch
import random


class Noise:
    def __init__(self, type='Gaussian', intensity=0.00001, mean=0, std=0.1):

        self.intensity = intensity # percent of points in original point cloud
        self.type = type
        self.mean = mean
        self.std = std
    def load(self):
        pass
    # def addNoise(self, points):
    #     '''
    #     input:
    #     1. points: list of ndarray of size (8192, 6)
    #     returns:
    #     1. points including noise points, normals of noise points will be set to zero
    #     (normals will not be used for training/prediction in this work)
    #     '''
    #     print(f'Adding random noise to {len(points)} point clouds...')
    #     N_noise = math.floor(self.intensity * points[0].shape[0])
    #     xyz = []
    #     for item in points:
    #         # Generate random indices for adding noise
    #         indices = np.random.choice(item[:, 0:3].shape[0], N_noise, replace=False)
    #
    #         # Generate Gaussian noise
    #         noise = np.random.normal(self.mean, self.std, size=(N_noise, 3))
    #
    #         # Add zeros at location of normals to noise
    #         zeros_columns = np.zeros(noise.shape)
    #
    #         # Concatenate the original array and the zeros column along the second axis
    #         noise_with_zeros = np.concatenate((noise, zeros_columns), axis=1)
    #
    #         # Add noise to selected points
    #         for idx, noise_elem in zip(indices, noise_with_zeros):
    #             item = np.insert(item, idx, [noise_elem], axis=0)
    #         xyz.append(item)
    #     return xyz

    def addNoise(self, tensor, num_noise_points=0, mirror_plane='Z'):
        """
        Add random noise to a torch.FloatTensor tensor.

        Args:
        - tensor (torch.FloatTensor): The input tensor of size [batch_size, num_points, num_features].
        - num_noise_points (int): The number of random noise points to add.
        - mirror_plane (string): name of mirror plane ('X', 'Y', 'Z')

        Returns:
        - torch.FloatTensor: The tensor with noise added, of size [batch_size, num_points + num_noise_points, num_features].
        """
        if self.type == 'NORMAL':
            batch_size, num_points, num_features = tensor.size()

            # Compute the range of values in the tensor
            min_val = tensor.min()
            max_val = tensor.max()

            # Expand tensor to accommodate noise
            tensor_with_noise = torch.zeros(batch_size, num_points + num_noise_points, num_features)
            tensor_with_noise[:, :num_points, :] = tensor

            # Add noise to random positions
            for i in range(batch_size):
                for j in range(num_noise_points):
                    tensor_with_noise[i, num_points + j, :] = torch.rand(num_features).cuda() * (max_val - min_val) + min_val

            return tensor_with_noise
        elif self.type == 'MIRROR':
            mirrored_points = tensor.clone()
            if mirror_plane == 'X':
                # Mirror along the x-axis (negate x coordinates)
                mirrored_points[:, :, 0] = -mirrored_points[:, :, 0]
            elif mirror_plane == 'Y':
                # Mirror along the y-axis (negate y coordinates)
                mirrored_points[:, :, 1] = -mirrored_points[:, :, 1]
            elif mirror_plane == 'Z':
                # Mirror along the z-axis (negate z coordinates)
                mirrored_points[:, :, 2] = -mirrored_points[:, :, 2]
            return mirrored_points

    def createNoise(self, min, max, shape, num_noise_points=0, mirror_plane='Z'):
        """
        Creates a tensor with random noise.

        Parameters:
        - min_val (float): The minimum value of the noise.
        - max_val (float): The maximum value of the noise.
        - shape (tuple): The shape of the tensor.
        - num_noise_points (int, optional): Number of elements in the tensor that will have noise. Default is 0.
        - mirror_plane (str, optional): The plane to mirror noise points across. Default is 'Z'.

        Returns:
        - torch.Tensor: A tensor with random noise values in `num_noise_points` positions.
        """

        if self.type == 'NORMAL':

            # Compute the range of values in the tensor
            min_val = min
            max_val = max

            # Create a tensor of zeros
            noise_tensor = torch.zeros(shape).cuda()

            if num_noise_points > 0:
                # Compute total number of elements in the tensor
                total_elements = noise_tensor.numel()

                # Ensure num_noise_points does not exceed the total number of elements
                # num_noise_points = min(num_noise_points, total_elements)

                # Generate random indices for noise points
                indices = torch.randperm(total_elements)[:num_noise_points]

                # Generate random noise values for the selected indices
                noise_values = (max_val - min_val) * torch.rand(num_noise_points).cuda() + min_val

                # Flatten the tensor, set noise values at the selected indices, and reshape
                flat_noise_tensor = noise_tensor.view(-1)
                flat_noise_tensor[indices] = noise_values
                noise_tensor = flat_noise_tensor.view(shape)

            return noise_tensor

        # elif self.type == 'MIRROR':
        #     mirrored_points = tensor.clone()
        #     if mirror_plane == 'X':
        #         # Mirror along the x-axis (negate x coordinates)
        #         mirrored_points[:, :, 0] = -mirrored_points[:, :, 0]
        #     elif mirror_plane == 'Y':
        #         # Mirror along the y-axis (negate y coordinates)
        #         mirrored_points[:, :, 1] = -mirrored_points[:, :, 1]
        #     elif mirror_plane == 'Z':
        #         # Mirror along the z-axis (negate z coordinates)
        #         mirrored_points[:, :, 2] = -mirrored_points[:, :, 2]
        #     return mirrored_points

    def generate_points_around_center(self, center, N):
        """
        Generates a tensor of shape [batch_size, num_tokens, N, 3] where each of the N points
        is normally distributed around the corresponding center points.

        Args:
        - center (torch.Tensor): A tensor of shape [batch_size, num_points, 3] representing the center points.
        - N (int): The number of points to generate around each center point.

        Returns:
        - torch.Tensor: A tensor of shape [batch_size, num_points, N, 3] with normally distributed points around each center point.
        """
        # Get the batch size and number of points from the center tensor
        batch_size, num_tokens, _ = center.shape
        max_val = torch.max(center).cpu()
        # Generate noise with values between (-1, 1)
        noise = 2 * torch.rand(batch_size, num_tokens, N, 3) - 1  # Uniform distribution in range (-1, 1)
        noise = noise / 10
        noise = noise + max_val*0.5
        noise = noise.cuda()

        # Expand center to match the dimensions for broadcasting
        center_expanded = center.unsqueeze(2).expand(-1, -1, N, -1)
        # print(noise)
        # Create points normally distributed around the center points
        points = center_expanded + noise

        return points

    def addNoiseOnes(self, tensor, N=1):
        batch_size, num_points, _ = tensor.shape

        # Generate N points with coordinates (1, 1, 1)
        new_points = torch.ones((batch_size, N, 3))

        # Iterate over each batch and insert new points at random indices
        for b in range(batch_size):
            indices = random.sample(range(num_points), N)
            for i in range(N):
                idx = indices[i]
                tensor[b, idx] = new_points[b, i]

        return tensor

    def changeOnePointToNoise(self, neighborhood, offset=0.15, train=True):
        if train:
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
        else:
            batch_size, num_points, num_coords = neighborhood.shape
            num_replacements = 64

            # Calculate the maximum x, y, z values across all points
            max_values = neighborhood.max(dim=1, keepdim=True).values  # Shape: (batch_size, 1, 3)

            # Generate random offsets slightly larger than the max values
            random_offsets = torch.rand(batch_size, num_replacements, num_coords, device=neighborhood.device) * offset

            # Create the noise points by adding the random offsets to the max values
            noise_points = max_values + random_offsets  # Shape: (batch_size, num_replacements, 3)

            # Generate random indices for each batch to replace 64 points
            random_indices = torch.randint(0, num_points, (batch_size, num_replacements), device=neighborhood.device)

            # Iterate through each batch to replace 64 points
            for b in range(batch_size):
                for idx in random_indices[b]:
                    neighborhood[b, idx, :] = noise_points[b, random_indices[b].tolist().index(idx), :]

        return neighborhood
