import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from communication_channel import CommunicationChannel
from receiver import Receiver
from transmitter import Transmitter


def init_run_normal(voltage, interval, signal_length, start, end):
    # Create instances of Transmitter ,Receiver and CommunicationChannel
    transmitter = Transmitter(voltage, interval, np.linspace(start, end, signal_length))
    receiver = Receiver(voltage, interval, np.linspace(start, end, signal_length))
    communication_channel = CommunicationChannel(transmitter, receiver)
    communication_channel.transmit_signal()
    return transmitter, receiver, communication_channel


def init_run_limited_bandwidth_fft(voltage: float, interval: float, first_frequency_component: int, signal_length: int,
                                   start: float, end: float):
    transmitter = Transmitter(voltage, interval, np.linspace(start, end, signal_length))
    receiver = Receiver(voltage, interval, np.linspace(start, end, signal_length))
    communication_channel = CommunicationChannel(transmitter, receiver)
    limited_bandwidth_signal = communication_channel.transmit_with_limited_bandwidth_with_using_fft(
        first_frequency_component)
    return transmitter, receiver, communication_channel, limited_bandwidth_signal


def init_run_limited_bandwidth_no_fft(voltage, interval, first_frequency_component, signal_length, start: float,
                                      end: float):
    transmitter = Transmitter(voltage, interval, np.linspace(start, end, signal_length))
    receiver = Receiver(voltage, interval, np.linspace(start, end, signal_length))
    communication_channel = CommunicationChannel(transmitter, receiver)
    limited_bandwidth_signal = communication_channel.transmit_with_limited_bandwidth_without_using_fft(
        first_frequency_component)
    return transmitter, receiver, communication_channel, limited_bandwidth_signal


def generate_transmit_receive_signal(voltage, interval, signal_length, start, end):
    transmitter, receiver, communication_channel = (
        init_run_normal(voltage, interval, signal_length=signal_length, start=start, end=end))
    received_signal, noisy_signal = receiver.process_signal()

    # Calculate the noise signal
    noise_signal = noisy_signal - received_signal

    transmitted_signal = transmitter.get_saved_transmitted_signal()

    return transmitted_signal, received_signal, noise_signal, noisy_signal


def plot_end_to_end_simulation(transmitted_signal, received_signal, noise_signal,
                               noisy_signal, signal_length, start, end):
    # Plot each signal in separate axes
    x = np.linspace(start, end, signal_length)

    _, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(x, transmitted_signal)
    axs[0].set_title('Transmitted Signal')
    axs[0].grid(True)

    axs[1].plot(x, noisy_signal)
    axs[1].set_title('Noisy Signal')
    axs[1].grid(True)

    axs[2].step(x, received_signal)
    axs[2].set_title('Received Signal')
    axs[2].grid(True)

    axs[3].plot(x, noise_signal)
    axs[3].set_title('Noise Signal')
    axs[3].grid(True)

    # Combine the separate axes into a single figure with one axis
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, transmitted_signal, label='Transmitted Signal')
    ax.plot(x, noisy_signal, label='Noisy Signal')
    ax.step(x, received_signal, label='Received Signal')
    ax.plot(x, noise_signal, label='Noise Signal')
    ax.set_title('Transmitted, Noisy, Received, and Noise Signals')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()
    plt.show()


def get_final_results(receiver, filtered_signal):
    received_signal, _ = receiver.process_signal()

    points_per_period = receiver.get_points_per_period()
    error_probability = 1 - norm.cdf(filtered_signal[int(points_per_period // 4)])
    return received_signal, filtered_signal, error_probability, receiver.signal


def simulate_with_limited_bandwidth_with_fft(voltage: float, interval: float,
                                             first_frequency_components: int, signal_length: int,
                                             start: float = 0, end: float = 1e-5):
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
        init_run_limited_bandwidth_fft(voltage, interval, first_frequency_components, signal_length=signal_length, start=start, end=end))

    received_signal, signal, error_probability, noisy_signal = get_final_results(receiver, filtered_signal)
    noise_signal = noisy_signal - filtered_signal
    return signal, noise_signal, received_signal, error_probability, noisy_signal


def simulate_with_limited_bandwidth_without_fft(voltage, interval, first_frequency_components, signal_length=1000,
                                                start: float = 0, end: float = 1e-5):
    transmitter, receiver, communication_channel, filtered_signal = (
        init_run_limited_bandwidth_no_fft(voltage, interval, first_frequency_components,
                                          signal_length=signal_length, start=start, end=end))

    received_signal, signal, error_probability, noisy_signal = get_final_results(receiver, filtered_signal)
    noise_signal = noisy_signal - filtered_signal
    return signal, noise_signal, received_signal, error_probability, noisy_signal


def plot_with_using_fft(voltage, interval, signal_length, start, end):
    # Simulate for different cases and store results
    signal, noise_signal, received_signal, error_probability, noisy_signal = (
        simulate_with_limited_bandwidth_with_fft(voltage, interval, 4,
                                                 signal_length=signal_length, start=start, end=end))
    plot_end_to_end_simulation(signal, received_signal, noise_signal, noisy_signal, signal_length, start, end)
    signal, noise_signal, received_signal, error_probability, noisy_signal = (
        simulate_with_limited_bandwidth_with_fft(voltage, interval, 3,
                                                 signal_length=signal_length, start=start, end=end))
    plot_end_to_end_simulation(signal, received_signal, noise_signal, noisy_signal, signal_length, start, end)
    signal, noise_signal, received_signal, error_probability, noisy_signal = (
        simulate_with_limited_bandwidth_with_fft(voltage, interval, 2,
                                                 signal_length=signal_length, start=start, end=end))
    plot_end_to_end_simulation(signal, received_signal, noise_signal, noisy_signal, signal_length, start, end)


def plot_without_using_fft(voltage, interval, signal_length, start, end):
    # Simulate for different cases and store results
    signal, noise_signal, received_signal, error_probability, noisy_signal = simulate_with_limited_bandwidth_without_fft(
        voltage, interval, 4, signal_length=signal_length, start=start,
        end=end)
    plot_end_to_end_simulation(signal, received_signal, noise_signal, noisy_signal, signal_length, start, end)
    signal, noise_signal, received_signal, error_probability, noisy_signal = simulate_with_limited_bandwidth_without_fft(
        voltage, interval, 3, signal_length=signal_length, start=start,
        end=end)
    plot_end_to_end_simulation(signal, received_signal, noise_signal, noisy_signal, signal_length, start, end)
    signal, noise_signal, received_signal, error_probability, noisy_signal = simulate_with_limited_bandwidth_without_fft(
        voltage, interval, 2, signal_length=signal_length, start=start,
        end=end)
    plot_end_to_end_simulation(signal, received_signal, noise_signal, noisy_signal, signal_length, start, end)


def plot_error_for_v(voltage_vector):
    # Calculate Q(V) for each value
    q_function_result = 1 - norm.cdf(voltage_vector)
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(voltage_vector, q_function_result, '-o', label="Q(V)")  # Combine line style and marker style
    plt.xlabel("V")
    plt.ylabel("Q(V)")
    plt.title("Q(V) for V from 0.1 to 2.1 (step size 0.1)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_error_for_v_vector_with_limited_frequency(voltage_vector: np.ndarray, interval: float,
                                                   first_frequency_components: int, signal_length: int,
                                                   start: float, end: float):
    with_fft = {
        v: simulate_with_limited_bandwidth_with_fft(v, interval, first_frequency_components,
                                                    signal_length, start, end)[3]
        for v in voltage_vector
    }
    without_fft = {
        v: simulate_with_limited_bandwidth_without_fft(v, interval, first_frequency_components,
                                                       signal_length, start, end)[3]
        for v in voltage_vector
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
    voltage = 5  # Voltage level
    bit_rate = 1e6  # Bit rate in bits per second
    interval = 2 / bit_rate  # Calculate the period interval from the bit rate
    signal_length = 1000
    start = 0
    end = 1e-5
    transmitted_signal, received_signal, noise_signal, noisy_signal = (
        generate_transmit_receive_signal(voltage, interval, signal_length=signal_length, start=start, end=end))
    plot_end_to_end_simulation(transmitted_signal, received_signal, noise_signal,
                               noisy_signal, signal_length, start, end)
    # Define the range of voltage values
    voltage_vector = np.arange(.1, 2.11, .1)
    plot_error_for_v(voltage_vector)
    plot_with_using_fft(voltage, interval, signal_length=signal_length, start=start, end=end)
    plot_without_using_fft(voltage, interval, signal_length=signal_length, start=start, end=end)
    plot_error_for_v_vector_with_limited_frequency(voltage_vector, interval, 4,
                                                   signal_length=signal_length, start=start, end=end)
    plot_error_for_v_vector_with_limited_frequency(voltage_vector, interval, 3,
                                                   signal_length=signal_length, start=start, end=end)
    plot_error_for_v_vector_with_limited_frequency(voltage_vector, interval, 2,
                                                   signal_length=signal_length, start=start, end=end)


if __name__ == "__main__":
    main()
