import numpy as np


class Receiver:
    def __init__(self, V, T, x):
        """
        Initializes the Receiver object with the given voltage level V and period T.

        Args:
            V (float): The voltage level representing 1 and -V representing 0.
            T (float): The period of the transmitted signal.
        """
        self.V = V
        self.T = T
        self.x = x

    def get_points_per_period(self):
        """
        Calculates the number of points per period for a given transmitted signal and period T.

        Returns:
            int: The number of points per period.
        """
        n = len(self.x)  # Length of the transmitted signal
        x_range = self.x[-1] - self.x[0]  # Range of the transmitted signal
        points_per_period = n * self.T / x_range  # Calculate the number of points per period

        return points_per_period

    def receive_and_sample(self, transmitted_signal, num_points):
        """
        Receives the signal, samples it at the middle of each bit interval, and reconstructs the original transmitted signal.
        Also adds Gaussian white noise to the transmitted signal to obtain the noisy signal.

        Args:
            transmitted_signal (numpy.ndarray): The transmitted signal.
            num_points (int): The number of points to sample in each bit interval.

        Returns:
            tuple: A tuple containing the received signal and the noisy signal.
        """
        # Calculate the number of intervals based on the length of the signal and the period T
        num_intervals = int((self.x[-1] - self.x[0]) / self.T)

        received_signal = np.zeros(len(transmitted_signal))
        noisy_signal = transmitted_signal + np.random.normal(0, 1, len(transmitted_signal))

        # Calculate the number of points per bit interval
        points_per_interval = self.get_points_per_period()

        for i in range(num_intervals):
            # Get the range for the current bit interval
            interval_start = int(i * points_per_interval)
            interval_end = int((i + 1) * points_per_interval)

            # Sample the noisy signal at the middle of the bit interval
            sample_index = int(interval_start + points_per_interval // 2)
            y_k = noisy_signal[sample_index]

            # Reconstruct the original transmitted signal value for the current bit interval
            if y_k > 0:
                received_signal[interval_start:interval_end] = self.V
            else:
                received_signal[interval_start:interval_end] = -self.V

        return received_signal, noisy_signal
