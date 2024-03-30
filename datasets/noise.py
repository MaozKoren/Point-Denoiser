import numpy as np
import math
import torch
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

    def addNoise(self, tensor, num_noise_points=0):
        """
        Add random noise to a torch.FloatTensor tensor.

        Args:
        - tensor (torch.FloatTensor): The input tensor of size [batch_size, num_points, num_features].
        - num_noise_points (int): The number of random noise points to add.

        Returns:
        - torch.FloatTensor: The tensor with random noise added, of size [batch_size, num_points + num_noise_points, num_features].
        """
        batch_size, num_points, num_features = tensor.size()

        # Generate random indices to add noise
        random_indices = torch.randint(0, num_points, size=(batch_size, num_noise_points))

        # Expand tensor to accommodate noise
        tensor_with_noise = torch.zeros(batch_size, num_points + num_noise_points, num_features)
        tensor_with_noise[:, :num_points, :] = tensor

        # Add noise to random positions
        for i in range(batch_size):
            for j in range(num_noise_points):
                tensor_with_noise[i, num_points + j, :] = torch.randn(num_features)

        return tensor_with_noise
