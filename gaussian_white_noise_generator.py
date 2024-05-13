import numpy as np


class GaussianWhiteNoiseGenerator:
    """
    A class to generate Gaussian noise samples.
    """

    def __init__(self, seed=None):
        """
        Initializes the GaussianNoiseGenerator object with the given seed.

        Args:
            seed (int, optional): The seed for the random number generator. If None, the current system time is used.
        """
        self.rng = np.random.default_rng(seed)

    def generate_noise(self, size):
        """
        Generates Gaussian noise samples with the specified size.

        Args:
            size (int): The number of noise samples to generate.

        Returns:
            numpy.ndarray: Array of Gaussian noise samples.
        """
        return self.rng.normal(0, 1, size)
