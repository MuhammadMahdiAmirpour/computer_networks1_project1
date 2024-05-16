from bit_string_producer import BitStringProducer
import numpy as np


class Transmitter:
    """
    A class to represent a transmitter that generates and transmits a bit string signal.
    """

    def __init__(self, V, T):
        """
        Initializes the Transmitter object with the given voltage level V and period T.

        Args:
            V (float): The voltage level representing 1 and -V representing 0.
            T (float): The period of the alternating bit string.
        """
        self.bit_string_producer = BitStringProducer(V, T)

    def transmit(self, x_range=(0, 10), num_points=1000):
        """
        Generates and transmits the bit string signal.

        Args:
            x_range (tuple, optional): The range of x values to transmit. Default is (0, 10).
            num_points (int, optional): The number of points to transmit. Default is 1000.

        Returns:
            numpy.ndarray: The transmitted bit string signal.
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        transmitted_signal = self.bit_string_producer(x)
        return transmitted_signal
