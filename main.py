from scipy.stats import norm
from scipy.fft import fft, ifft  # For frequency filtering
from scipy.signal import butter, lfilter
from transmitter import Transmitter
from receiver import Receiver
import numpy as np
import matplotlib.pyplot as plt

def simulate(V, T, signal_length=1000, start=0, end=1e-5):
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

def simulate_with_limited_bandwidth_with_fft(V, T, case, signal_length=1000, start=0, end=1e-5):
    """
    Simulates communication with limited bandwidth.

    Args:
        V (float): Voltage level.
        T (float): Period.
        num_points (int): Number of points in the signal.
        case (int): Case number (1, 2, or 3) for bandwidth limitation.

    Returns:
        tuple: A tuple containing the original signal, noisy signal, received signal,
            and the probability of error.
    """

    # Generate the transmitted signal
    transmitter = Transmitter(V, T)
    receiver = Receiver(V, T, np.linspace(start, end, signal_length))

    # Transmit the bit string signal
    transmitted_signal = transmitter.transmit(x_range=(start, end), num_points=signal_length)

    # Receive the noisy signal and sample it
    received_signal, noisy_signal = receiver.receive_and_sample(transmitted_signal)

    # Apply frequency filtering based on the case
    fft_transmitted = fft(transmitted_signal)  # Fast Fourier Transform
    if case == 1:
        fft_transmitted[4:] = 0  # Keep only the first 4 frequency elements
    elif case == 2:
        fft_transmitted[3:] = 0  # Keep only the first 3 frequency elements
    elif case == 3:
        fft_transmitted[2:] = 0  # Keep only the first 2 frequency elements
    else:
        raise ValueError("Invalid case number")
    filtered_signal = ifft(fft_transmitted).real  # Inverse FFT, get real part

    # Add noise and receive
    noisy_signal = filtered_signal + np.random.normal(0, 1, signal_length)
    received_signal, _ = receiver.receive_and_sample(noisy_signal)

    # Calculate error probability
    num_errors = np.count_nonzero(transmitted_signal != received_signal)
    error_probability = num_errors / signal_length

    return transmitted_signal, noisy_signal, received_signal, error_probability


def simulate_with_limited_bandwidth_without_fft(V, T, cutoff_freq, signal_length=1000, start=0, end=1e-5):
  """
  Simulates communication with limited bandwidth using a high-order Butterworth filter.

  Args:
      V (float): Voltage level.
      T (float): Period.
      cutoff_freq (float): Cutoff frequency for the limited bandwidth channel.
      signal_length (int, optional): Number of points in the signal. Defaults to 1000.
      start (float, optional): Starting x-axis value. Defaults to 0.
      end (float, optional): Ending x-axis value. Defaults to 1e-5.

  Returns:
      tuple: A tuple containing the original signal, noisy signal, received signal,
          and the probability of error.
  """

  # Generate the transmitted signal
  transmitter = Transmitter(V, T)
  receiver = Receiver(V, T, np.linspace(start, end, signal_length))
  transmitted_signal = transmitter.transmit(x_range=(start, end), num_points=signal_length)

  # Apply high-order Butterworth filter for smoother roll-off
  nyquist_freq = 0.5 / T  # Nyquist frequency
  fs = 1 / (end - start)  # Sampling frequency
  order = 15  # Increased order for smoother transition (adjust as needed)
  Wn = cutoff_freq / nyquist_freq  # Normalized cutoff frequency
  b, a = butter(order, Wn, btype='low')  # Design Butterworth filter
  filtered_signal = lfilter(b, a, transmitted_signal)

  # Add noise and receive
  noisy_signal = filtered_signal + np.random.normal(0, 1, signal_length)
  received_signal, _ = receiver.receive_and_sample(noisy_signal)

  # Calculate error probability
  num_errors = np.count_nonzero(transmitted_signal != received_signal)
  error_probability = num_errors / signal_length

  return transmitted_signal, noisy_signal, received_signal, error_probability

def plot_with_using_fft(V, T, signal_length, start, end):
    # Simulate for different cases and store results
    case1_results = simulate_with_limited_bandwidth_with_fft(V, T, 1, signal_length=signal_length, start=start, end=end)
    case2_results = simulate_with_limited_bandwidth_with_fft(V, T, 2, signal_length=signal_length, start=start, end=end)
    case3_results = simulate_with_limited_bandwidth_with_fft(V, T, 3, signal_length=signal_length, start=start, end=end)

    # Calculate error probabilities
    case1_error, case2_error, case3_error = (
        case1_results[3], case2_results[3], case3_results[3]
    )

    # Plot error probabilities
    plt.bar(["Case 1", "Case 2", "Case 3"], [case1_error, case2_error, case3_error])
    plt.xlabel("Case")
    plt.ylabel("Error Probability")
    plt.title("Error Probability for Different Bandwidth Limitations")
    plt.grid(True)
    plt.show()

def plot_without_using_fft(V, T, signal_length, start, end):
    # Simulate for different cases and store results
    case1_results = simulate_with_limited_bandwidth_without_fft(V, T, 1, signal_length=signal_length, start=start, end=end)
    case2_results = simulate_with_limited_bandwidth_without_fft(V, T, 2, signal_length=signal_length, start=start, end=end)
    case3_results = simulate_with_limited_bandwidth_without_fft(V, T, 3, signal_length=signal_length, start=start, end=end)

    # Calculate error probabilities
    case1_error, case2_error, case3_error = (
        case1_results[3], case2_results[3], case3_results[3]
    )

    # Plot error probabilities
    plt.bar(["Case 1", "Case 2", "Case 3"], [case1_error, case2_error, case3_error])
    plt.xlabel("Case")
    plt.ylabel("Error Probability")
    plt.title("Error Probability for Different Bandwidth Limitations")
    plt.grid(True)
    plt.show()

def main():
    V = 5  # Voltage level
    bit_rate = 1e6  # Bit rate in bits per second
    T = 2 / bit_rate  # Calculate the period T from the bit rate
    signal_length = 1000
    start = 0
    end = 1e-5
    plot_with_using_fft(V, T, signal_length=signal_length, start=start, end=end)
    plot_without_using_fft(V, T, signal_length=signal_length, start=start, end=end)


# def main():
#     # Global variables
#     V = 5  # Voltage level
#     bit_rate = 1e6  # Bit rate in bits per second
#     T = 2 / bit_rate  # Calculate the period T from the bit rate
#     signal_length = 1000
#     start = 0
#     end = 1e-5
#     transmitted_signal, received_signal, noise_signal, noisy_signal = simulate(V, bit_rate, T, signal_length, start, end)
#     plot_end_to_end_simulation(transmitted_signal, received_signal, noise_signal, noisy_signal, signal_length, start, end)
#     plot_error_for_v()

if __name__ == "__main__":
    main()

