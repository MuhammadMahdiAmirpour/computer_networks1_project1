from signal_producer import SignalProducer


class Transmitter:
    """
    A class to represent a transmitter that generates and transmits a bit string signal.
    """

    def __init__(self, V, T, x):
        """
        Initializes the Transmitter object with the given voltage level V and period T.

        Args:
            V (float): The voltage level representing 1 and -V representing 0.
            T (float): The period of the alternating bit string.
        """
        self.signal_producer = SignalProducer(V, T)
        self.x = x
        self.signal = None

    def generate_signal(self):
        """
        Generates and transmits the bit string signal.

        Returns:
            numpy.ndarray: The transmitted bit string signal.
        """
        transmitted_signal = self.signal_producer.generate_perfect_signal(self.x)
        self.signal = transmitted_signal

    def generate_signal_with_limited_bandwidth(self, first_n_frequency_elements):
        transmitted_signal = self.signal_producer.generate_limited_frequency_signal(first_n_frequency_elements, self.x)
        self.signal = transmitted_signal

    def generate_and_transmit(self):
        self.generate_signal()
        return self.get_saved_transmitted_signal()

    def generate_and_transmit_with_limited_bandwidth(self, first_n_frequency_elements):
        self.generate_signal_with_limited_bandwidth(first_n_frequency_elements)
        return self.get_saved_transmitted_signal()

    def get_saved_transmitted_signal(self):
        return self.signal
