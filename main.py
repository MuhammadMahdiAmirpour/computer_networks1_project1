from scipy.stats import norm
from transmitter import Transmitter
from receiver import Receiver
import numpy as np
import matplotlib.pyplot as plt

def simulate(V, bit_rate, T, signal_length=1000, start=0, end=1e-5):
    # Create instances of Transmitter and Receiver
    transmitter = Transmitter(V, T)
    receiver = Receiver(V, T, np.linspace(start, end, signal_length))

    # Transmit the bit string signal
    transmitted_signal = transmitter.transmit(x_range=(start, end), num_points=signal_length)

    # Receive the noisy signal and sample it
    received_signal, noisy_signal = receiver.receive_and_sample(transmitted_signal, num_points=signal_length)

    # Calculate the noise signal
    noise_signal = noisy_signal - transmitted_signal

    return transmitted_signal, received_signal, noise_signal, noisy_signal

def plot_end_to_end_simulation(transmitted_signal, received_signal, noise_signal, noisy_signal, signal_length, start, end):
   # Plot each signal in separate axes
    x_transmitted = np.linspace(start, end, signal_length)
    num_bits = len(received_signal)
    x_received = np.linspace(start, end, num_bits * 2)  # Create x_received with double the length
    x_noise = np.linspace(start, end, len(noise_signal))

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

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
    fig, ax = plt.subplots(figsize=(10, 6))
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

def plot_error_for_v():
    # Define the range of V values
    V = np.arange(0.1, 2.1, 0.1)
    # Calculate Q(V) for each value
    Q_V = 1 - norm.cdf(V)
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(V, Q_V, '-o', label="Q(V)")  # Combine line style and marker style
    plt.xlabel("V")
    plt.ylabel("Q(V)")
    plt.title("Q(V) for V from 0.1 to 2 (step size 0.1)")
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
    transmitted_signal, received_signal, noise_signal, noisy_signal = simulate(V, bit_rate, T, signal_length, start, end)
    plot_end_to_end_simulation(transmitted_signal, received_signal, noise_signal, noisy_signal, signal_length, start, end)
    plot_error_for_v()

if __name__ == "__main__":
    main()

