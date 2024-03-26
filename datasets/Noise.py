import numpy

class AddNoise:
    def __init__(self, intensity=1, type='Gaussian'):

        self.intensity = intensity
        self.type = type
    def load(self):
        pass
    def addNoise(self, points, labels):
        '''
        input:
        1. points: list of ndarray of size (8192, 6)
        2. labels: list of ndarray of size (1,) indicating label of each point in dataset
        returns:
        1. points including noise points, normals of noise points will be set to zero
        (normals will not be used for training/prediction in this work)
        2. corresponding labels
        '''
        xyz = []
        normals = []
        for item in points:
            xyz.append(item[:, 0:3])
            normals.append(item[:, 3:6])

        return points, labels