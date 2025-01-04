import utils
import numpy as np
import matplotlib.pyplot as plt
import audio_utils

def plot_position(ax, position, timestamps):
    ax.plot(timestamps, position)
    # ax.set_title("position")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.grid(True)

def plot_velocity(ax, velocity, timestamps, label=""):
    ax.plot(timestamps, velocity, label=label)
    # ax.set_title("position")
    ax.set_xlabel("Čas (s)")
    ax.set_ylabel("Rychlost (km/h)")
    ax.grid(True)

def sound_ts_to_velocity(ts1, ts2, dist = 8):
    dt = (ts2-ts1)
    v = dist/dt  # [m/s]
    return v

def main():
    fps = 25
    frames = utils.load_frame_indexes_from_csv("annotated_frames.csv")
    (data, sample_rate) = audio_utils.load_audio("video.mp3")
    ts_vid = frames/fps
    ts1, ts2 = utils.load_labels("Labels2.txt")
    snippet_start = ts1[0] - 5  # [s]
    snippet = data[int((ts1[0] - 5) * sample_rate):int((ts1[-1] + 5) * sample_rate), 1]

    filtered_snippet = audio_utils.highpass_filter(snippet, 12000, sample_rate, order=5)
    window_size = 50
    energy = np.zeros_like(snippet)
    energy = energy[window_size//2:-(window_size//2 - 1)] + audio_utils.average_energy_shifting_window(filtered_snippet, window_size)
    peaks, _ = audio_utils.find_peaks(energy, distance=4410, prominence=3e-5)
    peak_pairs, velocities = audio_utils.get_peak_pairs(peaks, energy[peaks], 30, 80, sample_rate, treshold=5)
    velocity_detected = velocities
    frames_detected = peaks[peak_pairs[:, 0]].T

    position = np.cumsum(100*np.ones_like(ts_vid))-100  # [m]
    velocity1 = (position[1:] - position[:-1])/(ts_vid[1:] - ts_vid[:-1])  # [m/s]
    velocity2 = (position[2:] - position[:-2])/(ts_vid[2:] - ts_vid[:-2])  # [m/s]
    velocity3 = sound_ts_to_velocity(ts1, ts2)  # [m/s]
    velocity1 = velocity1*(60*60)/1000
    velocity2 = velocity2*(60*60)/1000
    velocity3 = velocity3*(60*60)/1000  # [km/h]
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    plot_velocity(axs, velocity1, frames[:-1], label="Hektometrovníky, dopředná diference")
    plot_velocity(axs, velocity2, frames[1:-1], label="Hektometrovníky, centrální diference")
    plot_velocity(axs, velocity3, fps*(ts1+ts2)/2, label="Nárazy náprav, manuální anotace")
    plot_velocity(axs, velocity_detected, (frames_detected/sample_rate + snippet_start)*fps, label="Nárazy náprav, získané algoritmem")
    axs.legend()
    plt.show()
    pass


if __name__ == "__main__":
    main()