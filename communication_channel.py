import numpy as np
from scipy.fft import fft, ifft
from receiver import Receiver
from transmitter import Transmitter


class CommunicationChannel:
    def __init__(self, transmitter: Transmitter, receiver: Receiver):
        self.transmitter = transmitter
        self.receiver = receiver

    def transmit_signal(self):
        transmitted_signal = self.transmitter.generate_and_transmit()
        noisy_signal = transmitted_signal + np.random.normal(0, 1, len(transmitted_signal))
        self.receiver.receive_signal(noisy_signal)

    def transmit_with_limited_bandwidth_with_using_fft(self, first_n_frequency_elements: int):
        transmitted_signal = self.transmitter.generate_and_transmit()
        fft_transmitted = fft(transmitted_signal)
        # Sort the FFT coefficients by their absolute values in descending order
        sorted_indices = np.argsort(np.abs(fft_transmitted))[::-1]
        # Select the first k frequency components
        top_k_indices = sorted_indices[:first_n_frequency_elements]
        top_k_fft_signal = np.zeros_like(fft_transmitted)
        top_k_fft_signal[top_k_indices] = fft_transmitted[top_k_indices]
        # Reconstruct the signal from the top k frequency components
        reconstructed_signal = ifft(top_k_fft_signal).real
        noisy_signal = reconstructed_signal + np.random.normal(0, 1, len(transmitted_signal))
        self.receiver.receive_signal(noisy_signal)
        return reconstructed_signal

    def transmit_with_limited_bandwidth_without_using_fft(self, first_n_frequency_elements: int):
        transmitted_signal = self.transmitter.generate_and_transmit_with_limited_bandwidth(first_n_frequency_elements)
        noisy_signal = transmitted_signal + np.random.normal(0, 1, len(transmitted_signal))
        self.receiver.receive_signal(noisy_signal)
        return transmitted_signal
