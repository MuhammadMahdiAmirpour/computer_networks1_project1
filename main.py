import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from communication_channel import CommunicationChannel
from receiver import Receiver
from transmitter import Transmitter


def init_run_normal(V, T, signal_length=1000, start=0, end=1e-5):
    # Create instances of Transmitter ,Receiver and CommunicationChannel
    transmitter = Transmitter(V, T, np.linspace(start, end, signal_length))
    receiver = Receiver(V, T, np.linspace(start, end, signal_length))
    communication_channel = CommunicationChannel(transmitter, receiver)
    communication_channel.transmit_signal()
    return transmitter, receiver, communication_channel


def init_run_limited_bandwidth_fft(V: float, T: float, first_frequency_component: int, signal_length: int = 1000,
                                   start: float = 0, end: float = 1e-5):
    transmitter = Transmitter(V, T, np.linspace(start, end, signal_length))
    receiver = Receiver(V, T, np.linspace(start, end, signal_length))
    communication_channel = CommunicationChannel(transmitter, receiver)
    limited_bandwidth_signal = communication_channel.transmit_with_limited_bandwidth_with_using_fft(
        first_frequency_component)
    return transmitter, receiver, communication_channel, limited_bandwidth_signal


def init_run_limited_bandwidth_no_fft(V, T, first_frequency_component, signal_length=1000, start: float = 0,
                                      end: float = 1e-5):
    transmitter = Transmitter(V, T, np.linspace(start, end, signal_length))
    receiver = Receiver(V, T, np.linspace(start, end, signal_length))
    communication_channel = CommunicationChannel(transmitter, receiver)
    limited_bandwidth_signal = communication_channel.transmit_with_limited_bandwidth_without_using_fft(
        first_frequency_component)
    return transmitter, receiver, communication_channel, limited_bandwidth_signal


def generate_transmit_receive_signal(V, T, signal_length=1000, start=0, end=1e-5):
    transmitter, receiver, communication_channel = init_run_normal(V, T, signal_length=signal_length, start=start,
                                                                   end=end)
    received_signal, noisy_signal = receiver.process_signal()

    # Calculate the noise signal
    noise_signal = noisy_signal - received_signal

    transmitted_signal = transmitter.get_saved_transmitted_signal()

    return transmitted_signal, received_signal, noise_signal, noisy_signal


def plot_end_to_end_simulation(transmitted_signal, received_signal, noise_signal, noisy_signal, signal_length, start,
                               end):
    # Plot each signal in separate axes
    x_transmitted = np.linspace(start, end, signal_length)
    num_bits = len(received_signal)
    x_received = np.linspace(start, end, num_bits * 2)  # Create x_received with double the length
    x_noise = np.linspace(start, end, len(noise_signal))

    _, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(x_transmitted, transmitted_signal)
    axs[0].set_title('Transmitted Signal')
    axs[0].grid(True)

    axs[1].plot(x_transmitted, noisy_signal)
    axs[1].set_title('Noisy Signal')
    axs[1].grid(True)

    axs[2].step(x_received, np.repeat(received_signal, 2), where='mid')
    axs[2].set_title('Received Signal')
    axs[2].grid(True)

    axs[3].plot(x_noise, noise_signal)
    axs[3].set_title('Noise Signal')
    axs[3].grid(True)

    # Combine the separate axes into a single figure with one axis
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_transmitted, transmitted_signal, label='Transmitted Signal')
    ax.plot(x_transmitted, noisy_signal, label='Noisy Signal')
    ax.step(x_received, np.repeat(received_signal, 2), where='mid', label='Received Signal')
    ax.plot(x_noise, noise_signal, label='Noise Signal')
    ax.set_title('Transmitted, Noisy, Received, and Noise Signals')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()
    plt.show()


def get_final_results(receiver, filtered_signal, signal_length=1000, start: float = 0, end: float = 1e-5):
    # _, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(np.linspace(start, end, signal_length), filtered_signal, label="Filtered Signal")
    # plt.show()

    # Add noise and receive
    received_signal, signal = receiver.process_signal()

    points_per_period = receiver.get_points_per_period()
    error_probability = 1 - norm.cdf(signal[int(points_per_period // 4)])
    return received_signal, signal, error_probability, receiver.signal


def simulate_with_limited_bandwidth_with_fft(V: float, T: float,
                                             first_frequency_components: int, signal_length: int = 1000,
                                             start: float = 0, end: float = 1e-5) -> object:
    """
    Simulates communication with limited bandwidth.

    Args:
    V (float): Voltage level.
    T (float): Period.
    first_frequency_components (int): number which specifies how many frequency components we want to keep,
        for example we want to keep the first k frequency components signal_length
    signal_length (int): Length of the signal start (fload): start of the range end (fload): end of the range

    Returns:
        tuple: A tuple containing the original signal, noisy signal, received signal,
            and the probability of error.
    """

    transmitter, receiver, communication_channel, filtered_signal = (
        init_run_limited_bandwidth_fft(V, T, first_frequency_components, signal_length=1000, start=start, end=end))

    received_signal, signal, error_probability, noisy_signal = get_final_results(receiver, filtered_signal,
                                                                                 signal_length, start, end)
    noise_signal = signal - received_signal
    return signal, noise_signal, received_signal, error_probability, noisy_signal


def simulate_with_limited_bandwidth_without_fft(V, T, first_frequency_components, signal_length=1000,
                                                start: float = 0, end: float = 1e-5):
    transmitter, receiver, communication_channel, filtered_signal = (
        init_run_limited_bandwidth_no_fft(V, T, first_frequency_components, signal_length=signal_length,
                                          start=start, end=end))

    received_signal, signal, error_probability, noisy_signal = get_final_results(receiver, filtered_signal,
                                                                                 signal_length, start, end)
    noise_signal = signal - received_signal
    return signal, noise_signal, received_signal, error_probability, noisy_signal


def plot_with_using_fft(V, T, signal_length, start, end):
    # Simulate for different cases and store results
    signal, noise_signal, received_signal, error_probability, noisy_signal = (
        simulate_with_limited_bandwidth_with_fft(V, T, 4, signal_length=signal_length,
                                                 start=start,
                                                 end=end))
    plot_end_to_end_simulation(signal, noise_signal, received_signal, noisy_signal, signal_length, start, end)
    signal, noise_signal, received_signal, error_probability, noisy_signal = simulate_with_limited_bandwidth_with_fft(V,
                                                                                                                      T,
                                                                                                                      3,
                                                                                                                      signal_length=signal_length,
                                                                                                                      start=start,
                                                                                                                      end=end)
    plot_end_to_end_simulation(signal, noise_signal, received_signal, noisy_signal, signal_length, start, end)
    signal, noise_signal, received_signal, error_probability, noisy_signal = simulate_with_limited_bandwidth_with_fft(V,
                                                                                                                      T,
                                                                                                                      2,
                                                                                                                      signal_length=signal_length,
                                                                                                                      start=start,
                                                                                                                      end=end)
    plot_end_to_end_simulation(signal, noise_signal, received_signal, noisy_signal, signal_length, start, end)


def plot_without_using_fft(V, T, signal_length, start, end):
    # Simulate for different cases and store results
    signal, noise_signal, received_signal, error_probability, noisy_signal = simulate_with_limited_bandwidth_without_fft(
        V, T, 4, signal_length=signal_length, start=start,
        end=end)
    plot_end_to_end_simulation(signal, noise_signal, received_signal, noisy_signal, signal_length, start, end)
    signal, noise_signal, received_signal, error_probability, noisy_signal = simulate_with_limited_bandwidth_without_fft(
        V, T, 3, signal_length=signal_length, start=start,
        end=end)
    plot_end_to_end_simulation(signal, noise_signal, received_signal, noisy_signal, signal_length, start, end)
    signal, noise_signal, received_signal, error_probability, noisy_signal = simulate_with_limited_bandwidth_without_fft(
        V, T, 2, signal_length=signal_length, start=start,
        end=end)
    plot_end_to_end_simulation(signal, noise_signal, received_signal, noisy_signal, signal_length, start, end)


def plot_error_for_v(V):
    # Calculate Q(V) for each value
    Q_V = 1 - norm.cdf(V)
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(V, Q_V, '-o', label="Q(V)")  # Combine line style and marker style
    plt.xlabel("V")
    plt.ylabel("Q(V)")
    plt.title("Q(V) for V from 0.1 to 2.1 (step size 0.1)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_error_for_v_vector_with_limited_frequency(V: list, T: float,
                                                   first_frequency_components: int, signal_length: int = 1000,
                                                   start: float = 0, end: float = 1e-5):
    with_fft = {
        v: simulate_with_limited_bandwidth_with_fft(v, T, first_frequency_components, signal_length, start, end)[3]
        for v in V
    }
    without_fft = {
        v: simulate_with_limited_bandwidth_without_fft(v, T, first_frequency_components, signal_length, start, end)[3]
        for v in V
    }
    plt.figure(figsize=(8, 6))
    plt.plot(list(with_fft.keys()), list(with_fft.values()), '-o', label="Q(V) with fft")
    plt.plot(list(without_fft.keys()), list(without_fft.values()), '-o', label="Q(V) without fft")
    plt.xlabel("V")
    plt.ylabel("Q(V)")
    plt.title("Q(V) for V from 0.1 to 2.1 (step size 0.1)")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    # Global variables
    V = 5  # Voltage level
    bit_rate = 1e6  # Bit rate in bits per second
    T = 2 / bit_rate  # Calculate the period T from the bit rate
    signal_length = 1000
    start = 0
    end = 1e-5
    transmitted_signal, received_signal, noise_signal, noisy_signal = (
        generate_transmit_receive_signal(V, T, signal_length=signal_length, start=start, end=end))
    plot_end_to_end_simulation(transmitted_signal, received_signal, noise_signal, noisy_signal, signal_length, start,
                               end)
    # Define the range of V values
    V_for_err = np.arange(.1, 2.11, .1)
    plot_error_for_v(V_for_err)
    plot_with_using_fft(V, T, signal_length=signal_length, start=start, end=end)
    plot_without_using_fft(V, T, signal_length=signal_length, start=start, end=end)
    plot_error_for_v_vector_with_limited_frequency(V_for_err, T, 4, signal_length=signal_length, start=start, end=end)
    plot_error_for_v_vector_with_limited_frequency(V_for_err, T, 3, signal_length=signal_length, start=start, end=end)
    plot_error_for_v_vector_with_limited_frequency(V_for_err, T, 2, signal_length=signal_length, start=start, end=end)


if __name__ == "__main__":
    main()
