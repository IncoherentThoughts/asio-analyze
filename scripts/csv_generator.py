import pandas as pd
import numpy as np

"""
This series of functions is used to generate test csvs for the ASIO_Noise_Analysis.py. The csvs are formatted the same as what is expected output from the GSE for ASIO
"""
def create_gaussian(file_name, mu = 0, sigma = 1, n_points = 10000):
    """
    This function creates a pure Gaussian signal with given parameters for each channel
        Arguments:
            file_name (string): the name of the csv file to be produced
            mu (float): the mean of the Gaussian signal
            sigma (float): the standard deviation of the Gaussian signal
            n_points (int): the number of points of the Gaussian signal

    """
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    df = pd.DataFrame([np.random.normal(mu, sigma, n_points) for _ in range(len(channel_names))], index = channel_names)
    df.to_csv(file_name, header = None, index = True)

#create_gaussian('Gaussian.csv')

def create_gaussian_trended(file_name, mu = 0, sigma = 1, n_points = 10000):
    """
    This function creates a signal with a trend and a Gaussian noise distribution with given parameters for each channel.
        Arguments:
            file_name (string): the name of the csv file to be produced
            mu (float): the mean of the Gaussian signal
            sigma (float): the standard deviation of the Gaussian signal
            n_points (int): the number of points of the Gaussian signal
    """
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    x = np.arange(n_points)
    half_point = n_points // 2
    pyramid = np.concatenate([np.linspace(0, 1, half_point), np.linspace(1, 0, n_points - half_point)])

    SXR1 = np.random.normal(mu, sigma, n_points) + (0.0002 * x)      #positive linear trend
    SXR2 = np.random.normal(mu, sigma, n_points) - (0.0002 * x)      #negative linear trend
    SXR3 = np.random.normal(mu, sigma, n_points) + 2 * pyramid       #pyramid shaped trend, combination of positive and negative
    SXR4 = np.random.normal(mu, sigma, n_points) + 1e-7 * (x - n_points / 2) ** 2   #quadratic polynomial
    HXR = np.random.normal(mu, sigma, n_points) + 0.02 * np.exp(x / (n_points / 5)) #exponential
    EUV = np.random.normal(mu, sigma, n_points)                                     #pure Gaussian noise

    data = [SXR1, SXR2, SXR3, SXR4, HXR, EUV]
    df = pd.DataFrame(data, index = channel_names)
    df.to_csv(file_name, header = None, index = True)

#create_gaussian_trended('Gaussian_trended.csv')

def create_sinusoidal_signal(file_name, phase = 0, duration = 100, sampling_rate = 100):
    """
    This function creates a sinusoidal signal with given parameters for each channel
        Arguments:
            file_name (string): the name of the csv file to be produced
            phase (int): the phase of the sinusoidal signal
            duration (int): the duration of the sinusoidal signal
            sampling_rate (int): the sampling rate of the sinusoidal signal
    """
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    time_array = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    n = len(time_array)
    noise = np.random.normal(0, 1, n)

    SXR1 = 1 * (np.sin(np.pi * 10 * time_array + phase))
    SXR2 = 3 * (np.sin(2 * np.pi * 20 * time_array + phase) + 1)
    SXR3 = 4 * (np.sin(2 * np.pi * 30 * time_array + phase) + 1)
    SXR4 = 5 * (np.sin(2 * np.pi * 40 * time_array + phase) + 1)
    HXR = 6 * (np.sin(2 * np.pi * 25 * time_array + phase) + 1)
    EUV = 7 * (np.sin(2 * np.pi * 45 * time_array + phase) + 1)

    df = pd.DataFrame(
        [SXR1, SXR2 + noise, SXR3 + noise, SXR4 + noise, HXR + noise, EUV + noise], index = channel_names)

    df.to_csv(file_name, header = None, index = True)

create_sinusoidal_signal('sin_wave_sampler.csv')


def create_sin_in_gaussian(file_name, mu = 0, sigma = 3, phase = 0, duration = 100, sampling_rate = 100):
    """
    This function creates a sinusoidal signal buried in Gaussian noise for each channel.
    The noise level (sigma) is set high to "bury" the signal.
        Arguments:
            file_name (string): the name of the csv file to be produced
            mu (float): the mean of the Gaussian noise
            sigma (float): the standard deviation of the Gaussian noise (noise level)
            phase (int): the phase of the sinusoidal signal
            duration (int): the duration of the signal
            sampling_rate (int): the sampling rate of the signal
    """
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    time_array = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    n = len(time_array)

    # Base sine waves (amplitudes kept small relative to sigma=3)
    SXR1_sin = 0.5 * np.sin(np.pi * 10 * time_array + phase)
    SXR2_sin = 1.0 * np.sin(2 * np.pi * 20 * time_array + phase)
    SXR3_sin = 1.5 * np.sin(2 * np.pi * 30 * time_array + phase)
    SXR4_sin = 2.0 * np.sin(2 * np.pi * 40 * time_array + phase)
    HXR_sin = 2.5 * np.sin(2 * np.pi * 25 * time_array + phase)
    EUV_sin = 3.0 * np.sin(2 * np.pi * 45 * time_array + phase)

    # Generate Gaussian noise for all channels
    noise = np.random.normal(mu, sigma, n)

    # Add noise to each sine wave
    SXR1 = SXR1_sin + noise
    SXR2 = SXR2_sin + noise
    SXR3 = SXR3_sin + noise
    SXR4 = SXR4_sin + noise
    HXR = HXR_sin + noise
    EUV = EUV_sin + noise

    data = [SXR1, SXR2, SXR3, SXR4, HXR, EUV]
    df = pd.DataFrame(data, index = channel_names)
    df.to_csv(file_name, header = None, index = True)

create_sin_in_gaussian('sin_in_gaussian.csv')

def create_sin_in_gaussian_trended(file_name, mu = 0, sigma = 3, phase = 0, duration = 100, sampling_rate = 100):
    """
    This function creates a sinusoidal signal buried in Gaussian noise with a trend
    applied to the noise component for each channel.
        Arguments:
            file_name (string): the name of the csv file to be produced
            mu (float): the mean of the Gaussian noise
            sigma (float): the standard deviation of the Gaussian noise (noise level)
            phase (int): the phase of the sinusoidal signal
            duration (int): the duration of the signal
            sampling_rate (int): the sampling rate of the signal
    """
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    time_array = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    n = len(time_array)
    x = np.arange(n) # Index array for trends

    # Base sine waves (Amplitudes kept small relative to sigma=3)
    SXR1_sin = 0.5 * np.sin(np.pi * 10 * time_array + phase)
    SXR2_sin = 1.0 * np.sin(2 * np.pi * 20 * time_array + phase)
    SXR3_sin = 1.5 * np.sin(2 * np.pi * 30 * time_array + phase)
    SXR4_sin = 2.0 * np.sin(2 * np.pi * 40 * time_array + phase)
    HXR_sin = 2.5 * np.sin(2 * np.pi * 25 * time_array + phase)
    EUV_sin = 3.0 * np.sin(2 * np.pi * 45 * time_array + phase)

    # Generate BASE Gaussian noise
    base_noise = np.random.normal(mu, sigma, n)

    # --- Apply different trends to the noise before adding the sine wave ---
    half_point = n // 2
    pyramid = np.concatenate([np.linspace(0, 1, half_point), np.linspace(1, 0, n - half_point)])

    # Trended Noise = Base Noise + Trend
    SXR1_trend = base_noise + (0.002 * x)                   # Positive linear trend
    SXR2_trend = base_noise - (0.002 * x)                   # Negative linear trend
    SXR3_trend = base_noise + 10 * pyramid                  # Pyramid trend
    SXR4_trend = base_noise + 1e-5 * (x - n / 2) ** 2        # Quadratic trend
    HXR_trend = base_noise + 0.01 * np.exp(x / (n / 5))      # Exponential trend
    EUV_trend = base_noise                                   # Pure Gaussian noise (no trend)

    # Final Signal = Sine Wave + Trended Noise
    SXR1 = SXR1_sin + SXR1_trend
    SXR2 = SXR2_sin + SXR2_trend
    SXR3 = SXR3_sin + SXR3_trend
    SXR4 = SXR4_sin + SXR4_trend
    HXR = HXR_sin + HXR_trend
    EUV = EUV_sin + EUV_trend

    data = [SXR1, SXR2, SXR3, SXR4, HXR, EUV]
    df = pd.DataFrame(data, index = channel_names)
    df.to_csv(file_name, header = None, index = True)

create_sin_in_gaussian_trended('sin_in_gaussian_trended.csv')

