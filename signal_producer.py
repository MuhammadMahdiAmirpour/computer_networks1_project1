import numpy as np
import matplotlib.pyplot as plt


class SignalProducer:
    """
    A class to represent a signal producer.
    """

    def __init__(self, V, T):
        """
        Initializes the SignalProducer object with the given voltage level V and period T.

        Args:
            V (float): The voltage level representing 1 and -V representing 0.
            T (float): The period of the alternating bit string.
        """
        self.V = V
        self.T = T

    def generate_perfect_signal(self, x):
        """
        Evaluates the alternating bit string for the given input values x.

        Args:
            x (numpy.ndarray): Input values for which the alternating bit string is evaluated.

        Returns:
            numpy.ndarray: The alternating bit string values corresponding to the input values x.
        """
        return self.generate_limited_frequency_signal(freq_num=1000, x=x)

    def generate_limited_frequency_signal(self, freq_num: int, x):
        return self.V * (4 / np.pi) * sum([(1/(2 * i + 1)) * np.sin(2 * np.pi * x * (2 * i + 1) / self.T) for i in range(freq_num)])


if __name__ == '__main__':
    x_range = np.linspace(0, 1e-5, 1000)
    signal_producer = SignalProducer(5, 2/1e5)
    limited_frequency_signal = signal_producer.generate_limited_frequency_signal(2, x_range)
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_range, limited_frequency_signal, label="Filtered Signal")
    plt.show()
