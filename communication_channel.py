from gaussian_white_noise_generator import GaussianWhiteNoiseGenerator


class CommunicationChannel:
    """
    A class to represent a communication channel that applies Gaussian white noise to the transmitted signal.
    """

    def __init__(self, seed=None):
        """
        Initializes the CommunicationChannel object with the given seed for the noise generator.

        Args:
            seed (int, optional): The seed for the random number generator used for adding noise. If None, the current system time is used.
        """
        self.noise_generator = GaussianWhiteNoiseGenerator(seed=seed)

    def transmit(self, signal):
        """
        Applies Gaussian white noise to the transmitted signal and returns the noisy signal.

        Args:
            signal (numpy.ndarray): The transmitted signal to which noise is added.

        Returns:
            numpy.ndarray: The noisy signal with added Gaussian white noise.
        """
        noisy_signal = signal + self.noise_generator.generate_noise(len(signal))
        return noisy_signal
