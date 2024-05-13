from transmitter import Transmitter
from receiver import Receiver
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Global variables
    V = 4.5  # Voltage level
    bit_rate = 1e6  # Bit rate in bits per second
    T = 2 / bit_rate  # Calculate the period T from the bit rate

    # Create instances of Transmitter and Receiver
    transmitter = Transmitter(V, T)
    receiver = Receiver(V, T, np.linspace(0, 10, 1000))

    # Transmit the bit string signal
    transmitted_signal = transmitter.transmit(x_range=(0, 10), num_points=1000)

    # Receive the noisy signal and sample it
    received_signal, noisy_signal = receiver.receive_and_sample(transmitted_signal, num_points=1000)

    # Calculate the noise signal
    noise_signal = noisy_signal - transmitted_signal

    # Plot each signal in separate axes
    x_transmitted = np.linspace(0, 10, 1000)
    num_bits = len(received_signal)
    x_received = np.linspace(0, 10, num_bits * 2)  # Create x_received with double the length
    x_noise = np.linspace(0, 10, len(noise_signal))

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


if __name__ == "__main__":
    main()
