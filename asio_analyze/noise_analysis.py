import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis, skew
from scipy.signal import savgol_filter
import os

"""
These functions are used to analyze the data collected from ASIO. They operate on in-memory
DataFrames with shape (6, N) — one row per ASIO channel in canonical order
[SXR1, SXR2, SXR3, SXR4, HXR, EUV] — produced by `latex_report._prepare_analysis_frame`.
Plot functions also take an `output_dir` for where to drop their PNGs.
"""

CHANNEL_NAMES = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']


def _channel_stats(df):
    """Compute [name, mean, rms, std, skew, kurtosis] for each row of `df`."""
    stats = []
    for i, name in enumerate(CHANNEL_NAMES):
        data = df.iloc[i].values
        mean = np.mean(data)
        rms = np.sqrt(np.mean(data ** 2))
        std = np.std(data)
        skewedness = skew(data, bias=True)
        kurt = kurtosis(data, bias=True, fisher=True)
        stats.append([name, mean, rms, std, skewedness, kurt])
    return stats


def channel_voltage_stats(df):
    """Basic per-channel stats for a (6, N) voltage frame.

    Returns: list of [name, mean, rms, std, skew, kurtosis].
    """
    return _channel_stats(df)


def channel_voltage_stats_detrended(df):
    """Basic per-channel stats for a cleaned/detrended (6, N) voltage frame."""
    return _channel_stats(df)


def plot_signals_voltages(df, output_dir):
    """Plot raw voltage time series for each channel into `output_dir`."""
    x = np.arange(df.shape[1]) * 0.01
    cmap = plt.get_cmap('Dark2')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    axes = axes.ravel()

    for i, name in enumerate(CHANNEL_NAMES):
        color = cmap(i / len(CHANNEL_NAMES))
        axes[i].scatter(x, df.iloc[i], color=color, label=name, s=0.1)
        axes[i].tick_params(axis='x', labelrotation=45)
        axes[i].set_ylabel('Voltage')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_title(f'{name}')
        axes[i].grid(True)

    fig.suptitle('ASIO Raw Voltage Values', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(output_dir, 'Raw_Voltages.png')
    plt.savefig(filename)
    plt.close()
    return filename


def plot_fft(df, output_dir):
    """Plot FFTs (one subplot per channel) into `output_dir`. Data is not mean-centered."""
    n_samples = df.shape[1]
    sampling_rate = 100
    cmap = plt.get_cmap('Dark2')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    axes = axes.ravel()

    f = np.fft.fftshift(np.fft.fftfreq(n_samples, d=1 / sampling_rate))
    positive_freqs = f >= 0

    for i, name in enumerate(CHANNEL_NAMES):
        data = df.iloc[i].values
        data = data - np.mean(data)
        freq = np.fft.fft(data)
        freq_shifted = np.fft.fftshift(freq)
        amplitude = np.abs(freq_shifted)

        color = cmap(i / len(CHANNEL_NAMES))
        axes[i].plot(f[positive_freqs], amplitude[positive_freqs], color=color)
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f'{name}')
        axes[i].grid(True)

    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle('FFTs', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(output_dir, 'ASIO_FFT.png')
    plt.savefig(filename)
    plt.close()
    return filename


def apply_savgol_filter(df, window_length=51, polyorder=3, num_passes=2):
    """Iteratively Savitsky-Golay-filter each channel of `df` and subtract the trend.

    Returns (savgol_df, trend_df), both shape (6, N).
    """
    detrended_signals = []
    trends = []
    for i in range(len(CHANNEL_NAMES)):
        data = df.iloc[i].values
        trend_input = data.copy()
        for _ in range(num_passes):
            trend = savgol_filter(trend_input, window_length=window_length, polyorder=polyorder)
            trend_input = trend
        detrended_signals.append(data - trend)
        trends.append(trend)

    return pd.DataFrame(detrended_signals), pd.DataFrame(trends)


def remove_room_emi(df):
    """Notch ±40 Hz EMI from each channel of `df` via FFT/IFFT, after Savitsky-Golay detrend.

    Returns (cleaned_df, rms_results) where rms_results is a list of
    (channel_name, rms_cleaned, rms_original) tuples on mean-centered signals.
    """
    detrended_df, _ = apply_savgol_filter(df, window_length=51, polyorder=3, num_passes=2)

    n_channels = detrended_df.shape[0]
    n_samples = detrended_df.shape[1]
    sampling_rate = 100
    f = np.fft.fftshift(np.fft.fftfreq(n_samples, d=1 / sampling_rate))

    cleaned_signals = []
    rms_results = []

    for i in range(n_channels):
        data = detrended_df.iloc[i].values
        data_detrended = data - np.mean(data)

        freq = np.fft.fft(data_detrended)
        freq_shifted = np.fft.fftshift(freq)

        RoomEMI_pos = [40.1 - 0.2, 40.1 + 0.2]
        RoomEMI_neg = [-40.1 - 0.2, -40.1 + 0.2]
        RemoveFFT_pos = np.where((f >= RoomEMI_pos[0]) & (f <= RoomEMI_pos[1]))[0]
        RemoveFFT_neg = np.where((f >= RoomEMI_neg[0]) & (f <= RoomEMI_neg[1]))[0]

        freq_shifted[RemoveFFT_pos] = 0
        freq_shifted[RemoveFFT_neg] = 0

        cleaned = np.fft.ifft(np.fft.ifftshift(freq_shifted))
        cleaned = np.real(cleaned)

        rms_cleaned = np.sqrt(np.mean(cleaned ** 2))
        rms_original = np.sqrt(np.mean(data_detrended ** 2))

        cleaned_signals.append(cleaned)
        rms_results.append((CHANNEL_NAMES[i], rms_cleaned, rms_original))

    cleaned_df = pd.DataFrame(cleaned_signals)
    return cleaned_df, rms_results


def plot_cleaned_signals_time_series(cleaned_df, output_dir):
    """Plot EMI-filtered, detrended voltage time series into `output_dir`."""
    n_channels = cleaned_df.shape[0]
    n_samples = cleaned_df.shape[1]
    sampling_rate = 100
    time = np.linspace(0, n_samples / sampling_rate, n_samples, endpoint=False)
    cmap = plt.get_cmap('Dark2')

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    axes = axes.ravel()

    for i in range(n_channels):
        ax = axes[i]
        color = cmap(i / len(CHANNEL_NAMES))
        ax.scatter(time, cleaned_df.iloc[i].values, color=color, s=0.1)
        ax.set_title(f'{CHANNEL_NAMES[i]}')
        ax.set_ylabel('Voltage(V)')
        ax.set_xlabel('Time (s)')
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid(True)

    fig.suptitle('EMI-Filtered and Detrended Voltage Values')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(output_dir, "Room_EMI_Removed_Voltages.png")
    plt.savefig(filename)
    plt.close()
    return filename


def plot_cleaned_histogram(df, output_dir):
    """Plot per-channel noise histograms with Gaussian best-fit into `output_dir`.

    `df` is a (6, N) frame of cleaned, mean-centered signals.
    """
    cmap = plt.get_cmap('Dark2')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    axes = axes.ravel()

    for i, name in enumerate(CHANNEL_NAMES):
        ax = axes[i]
        data = df.iloc[i].values
        sigma = np.std(data)
        color = cmap(i / len(CHANNEL_NAMES))

        num_bins = 45
        data_min, data_max = data.min(), data.max()
        bin_width = (data_max - data_min) / num_bins
        centers = np.linspace(data_min + bin_width / 2, data_max - bin_width / 2, num_bins)
        edges = np.concatenate(([centers[0] - bin_width / 2], centers + bin_width / 2))
        shift = bin_width * 0.03
        plotted_bins = edges + shift
        ax.hist(data, bins=plotted_bins, color=color, density=False, alpha=0.6,
                edgecolor='black', linewidth=0.4)
        ax.tick_params(axis='x', labelrotation=45)

        xmin, xmax = ax.get_xlim()
        N = len(data)
        x = np.linspace(xmin, xmax, 1000)
        p = norm.pdf(x, loc=np.mean(data), scale=sigma)
        p_scaled = p * N * bin_width
        ax.plot(x, p_scaled, 'r', linewidth=2, label='Gaussian Fit')

        ax.set_title(f'{name}')
        ax.set_xlabel('(V)')
        ax.set_ylabel('Counts')
        ax.grid(True)

    fig.suptitle('ASIO Noise Histograms w/EMI-filtering and Detrending', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(output_dir, 'Room_EMI_Removed_Noise_Histograms.png')
    plt.savefig(filename)
    plt.close()
    return filename
