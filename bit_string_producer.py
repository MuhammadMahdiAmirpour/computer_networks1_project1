import numpy as np


class BitStringProducer:
    """
    A class to represent a bit string producer.
    """

    def __init__(self, V, T):
        """
        Initializes the BitStringProducer object with the given voltage level V and period T.

        Args:
            V (float): The voltage level representing 1 and -V representing 0.
            T (float): The period of the alternating bit string.
        """
        self.V = V
        self.T = T

    def __call__(self, x):
        """
        Evaluates the alternating bit string for the given input values x.

        Args:
            x (numpy.ndarray): Input values for which the alternating bit string is evaluated.

        Returns:
            numpy.ndarray: The alternating bit string values corresponding to the input values x.
        """
        return np.where(np.sin(2 * np.pi * x / self.T) >= 0, self.V, -self.V)
