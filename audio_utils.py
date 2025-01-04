import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import simpleaudio as sa
from numpy.ma.core import ones_like, zeros_like
import scipy
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter

import utils
from utils import save_result, load_result, load_labels



def load_audio(file_path):
    """
    Load an audio file into a NumPy array.

    Parameters:
        file_path (str): Path to the audio file.

    Returns:
        tuple: audio data as numpy array, sample rate
    """
    print("loading...")
    data, sample_rate = sf.read(file_path)
    print("loaded")
    return data, sample_rate


def play_audio(data, sample_rate):
    """
    Play audio data using simpleaudio.

    Parameters:
        data (numpy array): Audio data.
        sample_rate (int): Sampling rate.
    """
    # Ensure data is in the correct format
    audio = (data * 32767).astype(np.int16) if data.dtype != np.int16 else data

    # Create a playback object
    play_obj = sa.play_buffer(audio, 1 if len(audio.shape) == 1 else audio.shape[1], 2, sample_rate)
    play_obj.wait_done()

def plot_audio(ax, data, sample_rate, start_time = 0, y_label = "Amplituda"):
    """
    Plot the audio waveform using Matplotlib.

    Parameters:
        data (numpy array): Audio data.
        sample_rate (int): Sampling rate.
        ax (matplotlib.axes.Axes): Axis to plot on.
    """
    time = np.linspace(0, len(data) / sample_rate, num=len(data)) + start_time
    ax.plot(time, data)
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("Čas (s)")
    ax.set_ylabel(y_label)
    ax.grid(True)

def plot_spectrogram(ax, data, sample_rate, start_time=0):
    """
    Plot the spectrogram of the audio data.

    Parameters:
        data (numpy array): Audio data.
        sample_rate (int): Sampling rate.
        ax (matplotlib.axes.Axes): Axis to plot on.
        x_shift (float): Shift the x-axis by a specified amount in seconds.
    """
    time = np.linspace(0, len(data) / sample_rate, num=len(data))
    # Pxx, freqs, bins, im = ax.specgram(data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
    Pxx, freqs, bins, im = ax.specgram(data, Fs=sample_rate, NFFT=256, noverlap=128, cmap='viridis')
    ax.set_xlabel("Čas (s)")
    ax.set_ylabel("Frekvence (Hz)")
    ax.set_xlim(time[0], time[-1])

def plot_timestamps(ax, ts1, ts2, scatter=False, length = 1, label=True):
    if label:
        label1 = "manuálně anotovaný náraz přední nápravy"
        label2 = "manuálně anotovaný náraz zadní nápravy"
    else:
        label1 = None
        label2 = None
    if ts1 is not None:
        if scatter:
            ax.scatter(ts1, 10000*ones_like(ts1), c="m", label=label1)
        else:
            ax.vlines(ts1, -length/2, length/2, colors=["m"], label=label1)
    if ts2 is not None:
        if scatter:
            ax.scatter(ts2, 10000*ones_like(ts2), c="r", label=label2)
        else:
            ax.vlines(ts2, -length/2, length/2, colors=["r"], label=label2)
    pass
def plot_peaks(ax, signal, peaks, sample_rate, start_time=0, label=""):
    ax.plot((peaks/sample_rate) + start_time, signal[peaks], "x", label=label)
    pass

def equalizer(data, sample_rate, freq_bands, gains):
    """
    Apply an equalizer effect to the audio data.

    Parameters:
        data (numpy array): Input audio data.
        sample_rate (int): Sampling rate of the audio.
        freq_bands (list): List of cutoff frequencies for bands.
        gains (list): Gain factors for each frequency band.

    Returns:
        numpy array: Equalized audio data.
    """
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        return lfilter(b, a, data)

    output = np.zeros_like(data)
    for i in range(len(freq_bands) - 1):
        filtered = bandpass_filter(data, freq_bands[i], freq_bands[i + 1], sample_rate)
        output += gains[i] * filtered

    return output

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def highpass_filter(data, cutoff, sample_rate, order=5, show=False):
    """
    Apply a high-pass filter to the audio data.

    Parameters:
        data (numpy array): Input audio data.
        cutoff (float): Cutoff frequency for the high-pass filter.
        sample_rate (int): Sampling rate of the audio.
        order (int): Order of the filter.

    Returns:
        numpy array: Filtered audio data.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high')
    if show:
        # Frequency response
        w, h = scipy.signal.freqz(b, a, worN=8000)

        # Convert from rad/sample to Hz
        freqs = w * sample_rate / (2 * np.pi)
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.semilogx(freqs, 20 * np.log10(abs(h)))  # Magnitude in dB
        plt.title('Bodeho plot horní propusti')
        plt.xlabel('Frekvence (Hz)')
        plt.ylabel('Zesílení (dB)')
        plt.grid(which='both', linestyle='--', color='gray')

        # Plot phase response
        plt.subplot(2, 1, 2)
        plt.semilogx(freqs, np.angle(h, deg=True))  # Phase in degrees
        plt.xlabel('Frekvence (Hz)')
        plt.ylabel('Fáze (degrees)')
        plt.grid(which='both', linestyle='--', color='gray')

        plt.tight_layout()
        plt.show()
    return lfilter(b, a, data)


def average_energy_shifting_window(data, window_size):
    # Ensure the input is a 1D numpy array
    arr = data

    # Check if the window size is valid
    if window_size <= 0 or window_size > len(arr):
        raise ValueError("Window size must be positive and less than or equal to the length of the array.")

    # List to store the average energy values
    avg_energy = []

    # Loop through the array with the shifting window
    for i in range(len(arr) - window_size + 1):
        # Get the current window
        window = arr[i:i + window_size]

        # Calculate the energy (sum of squares)
        energy = np.sum(window ** 2)

        # Append the average energy to the list
        avg_energy.append(energy / window_size)

    return np.array(avg_energy)

def numerical_derivative(data, sample_rate):
    dt = 1/sample_rate
    return (data[1:] - data[:-1])/dt


def compute_signal_envelope(signal):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope

def get_peak_pairs(peaks, peak_values, min_speed, max_speed, sample_rate, treshold = 5, dist = 8):
    """
    :param peaks: timestamps of peaks [sample]
    :param min_speed: in [km/h]
    :param max_speed:  in [km/h]
    :param sample_rate: [sample/s]
    :param dist: [m]
    :param treshold [km/h]
    """
    min_dt = dist/(max_speed*1000/(60*60))  # [s]
    max_dt = dist/(min_speed*1000/(60*60))  # [s]
    D = np.abs((peaks[:, np.newaxis] - peaks).T/sample_rate)  # [s]
    mask = (D > min_dt) & (D < max_dt)
    mask = mask & (np.tri(*mask.shape).T == 1)
    # no_ambiguity_intervals =
    valid_intervals = D[mask]
    valid_speeds = (dist/valid_intervals)*(60*60)/1000  # [km/h]
    valid_speeds_indices = np.array(np.where(mask==True))
    valid_peak_heights = peak_values[valid_speeds_indices][0, :]*0.7 > peak_values[valid_speeds_indices][1, :]
    # only peak pairs that start with a high peak and end with a low peak are considered valid
    valid_speeds = valid_speeds[valid_peak_heights]
    valid_speeds_indices = valid_speeds_indices[:, valid_peak_heights]
    # ransac
    best_match = None
    most_matches = -1
    for i in range(len(valid_speeds)):
        matches = np.sum(np.abs(valid_speeds - valid_speeds[i]) < treshold)
        if most_matches < matches:
            most_matches = matches
            best_match = valid_speeds[i]
    inliers = np.abs(valid_speeds - best_match) < treshold
    pairs = valid_speeds_indices[:, inliers].T
    velocities = valid_speeds[inliers]
    return pairs, velocities

def get_spectogram(signal, sample_rate, blur = 1, normalize = True, show=False):
    nperseg = 128
    noverlap = nperseg//2
    mode = 'magnitude'
    f, t, Sxx = scipy.signal.spectrogram(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, mode=mode)
    Sxx = gaussian_filter(Sxx, sigma=blur)
    if normalize:
        frobenius_norm = np.linalg.norm(Sxx, 'fro')
        Sxx /= frobenius_norm
    if show:
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)]')
        plt.title('Spectrogram')
        plt.colorbar(label='Intensity')
        plt.show()
    return f, t, Sxx

def get_spectogram_similarity(signal_long, signal_short, nperseg = 8, show=False):
    iter_num = (signal_long.shape[1] - signal_short.shape[1])//nperseg
    ret = np.zeros((iter_num))
    for i in range((signal_long.shape[1] - signal_short.shape[1])//nperseg):
        window = signal_long[:, i*nperseg:i*nperseg+signal_short.shape[1]]
        frobenious_norm = np.linalg.norm(window - signal_short, 'fro')
        ret[i] = frobenious_norm
    return ret


def main():
    """
    Main function to load, play, and plot audio data.

    Parameters:
        file_path (str): Path to the audio file.
    """
    data, sample_rate = load_audio("video.mp3")

    # meant for faster loading in exchange for disk space
    # save_result("audio_data.p", (data, sample_rate))
    # (data, sample_rate) = load_result("audio_data.p")

    ts1, ts2 = load_labels("Labels2.txt")
    first_ts_index = 15
    snippet = data[int((ts1[first_ts_index]-1)*sample_rate):int((ts1[first_ts_index]+5)*sample_rate), 1]
    snippet_start_time = ts1[first_ts_index]-1  # [s]
    window_size = 50
    energy = zeros_like(snippet)
    energy = energy[window_size//2:-(window_size//2 - 1)] + average_energy_shifting_window(snippet, window_size)
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    plot_audio(axs[0, 0], snippet, sample_rate, snippet_start_time)
    plot_timestamps(axs[0, 0], ts1, ts2, length=0.4)
    axs[0, 0].set_title("Nefiltrovaný zvuk")
    axs[0, 0].legend()
    plot_spectrogram(axs[1, 0], snippet, sample_rate, snippet_start_time)
    plot_timestamps(axs[1, 0], ts1-snippet_start_time, ts2-snippet_start_time, scatter=True)
    axs[1, 0].set_title("Spektogram nefiltrovaného zvuku")
    plot_audio(axs[2, 0], energy, sample_rate, snippet_start_time, y_label="Okamžitá energie")
    plot_timestamps(axs[2, 0], ts1, ts2, length=0.02)
    axs[2, 0].set_title("Průběh energie nefiltrovaného zvuku")

    filtered_snippet = highpass_filter(snippet, 12000, sample_rate, show=False)
    pass
    energy = zeros_like(filtered_snippet)
    energy = energy[window_size//2:-(window_size//2 - 1)] + average_energy_shifting_window(filtered_snippet, window_size)
    plot_audio(axs[0, 1], filtered_snippet, sample_rate, snippet_start_time)
    axs[0, 1].set_title("Filtrovaný zvuk")
    # axs[0, 1].legend()
    plot_timestamps(axs[0, 1], ts1, ts2, length=0.02)
    plot_spectrogram(axs[1, 1], filtered_snippet, sample_rate, snippet_start_time)
    plot_timestamps(axs[1, 1], ts1-snippet_start_time, ts2-snippet_start_time, scatter=True)
    axs[1, 1].set_title("Spektogram filtrovaného zvuku")
    plot_audio(axs[2, 1], energy, sample_rate, snippet_start_time, y_label="Okamžitá energie")
    plot_timestamps(axs[2, 1], ts1, ts2, length=0.00006, label=False)
    axs[2, 1].set_title("Průběh energie filtrovaného zvuku")
    peaks, _ = find_peaks(energy, distance=4410, prominence=3e-5)
    peak_pairs, velocities = get_peak_pairs(peaks, energy[peaks], 30, 80, sample_rate, treshold=3)
    plot_peaks(axs[2, 1], energy, peaks, sample_rate, snippet_start_time, label="Detekované energetické špičky")
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()
    pass

if __name__ == "__main__":
    main()

