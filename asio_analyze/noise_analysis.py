import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis, skew
from scipy.signal import savgol_filter
import os

"""
These functions are used to analyze the data collected from ASIO, they operate on csvs or dataframes that are formatted like what we
expect the GSE software to output. Most of these functions output a png of a graph with six subplots, one for each of ASIO's channels. 
"""
def channel_voltage_stats(file_path):
    """
        Reads a CSV file with 6 rows, each row is a time series of voltage values from each ASIO channel.
        Generates basic statistics for each channel.

        Arguments:
            file_path (str): Path to the CSV file
        Returns:
            a list of lists for each ASIO channel: [mean signal, RMS, standard deviation, min, max, skewness, kurtosis]
    """
    df = pd.read_csv(file_path, header=None)                       #reads in csv converts to dataframe
    df = df.drop(df.columns[0], axis=1)                            #drops first column with channel labels
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV'] #list of channel names

    #uncomment below to print stats in terminal of IDE
    #print(f"{'Channel':<6} {'Mean':>8} {'RMS':>8} {'StdDev':>10} {'Min':>10} {'Max':>8} {'Skew':>10} {'Kurtosis':>15}") #prints title column
    #print('-' * 90)

    stats = [] #empty list to hold stats

    for i, name in enumerate(channel_names):
        data = df.iloc[i].values                          #gets values for each row, which represents a channel
        mean = np.mean(data)                              #calculates mean
        rms = np.sqrt(np.mean(data**2))                   #calculates RMS of full signal
        std = np.std(data)                                #calculates standard devaition
        skewedness = skew(data, bias = True)              #calculates skew of dataset, should be 0
        kurt = kurtosis(data, bias = True, fisher = True) #calculates fisher kurtosis, should be 0
        stats.append([name, mean, rms, std,skewedness, kurt])  #adds to stat list

        #uncomment below to print stats in terminal of IDE
        #print(f"{name:<6} {mean:10.10f} {rms:10.10f} {std:10.10f} {min:10.10f} {max:10.10f} {skewedness:10.4f}, {kurt:10.4f}") #formats and prints stats

    return stats

#test call
#stats = channel_voltage_stats('Gaussian_trended.csv')
#print(stats)


def channel_voltage_stats_detrended(df):
    """
        Takes a dataframe where each row is a time series of voltage values from each ASIO channel. Intended to be used after 40hz spike has been removed
        and data has been detrended by applying the apply_savgol_filter function. Generates basic statistics for each channel.

        Arguments:
            df (pd.DataFrame): Dataframe of cleaned signals.
        Returns:
            a list of lists for each ASIO channel: [mean signal, RMS, standard deviation, min, max, skewness, kurtosis]
    """

    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    stats_detrended = [] #empty list to hold stats

    for i, name in enumerate(channel_names):
        data = df.iloc[i].values                          #gets values for each row, which represents a channel
        mean = np.mean(data)                              #calculates mean
        rms = np.sqrt(np.mean(data**2))                   #calculates RMS of full signal
        std = np.std(data)                                #calculates standard devaition
        skewedness = skew(data, bias = True)              #calculates skew of dataset, should be 0
        kurt = kurtosis(data, bias = True, fisher = True) #calculates fisher kurtosis, should be 0
        stats_detrended.append([name, mean, rms, std, skewedness, kurt])  #adds to stats_detrended list

    return stats_detrended

#test call
#stats_detrended = channel_voltage_stats('Gaussian_trended.csv')
#print(stats_detrended)

def Std_and_RMS_current_comparison(file_path, df):
    """
        Takes a csv and a dataframe where each row is a time series of voltage values from each ASIO channel calculates the current

        Arguments:
            file_path (str): Path to the CSV file
            df (pd.DataFrame): Dataframe of cleaned signals.
        Returns:
            a list of lists for each ASIO channel: [Channel, mean(fA), RMS(fA), StdDev(fA)]]
    """
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    df_csv = pd.read_csv(file_path, header=None)  # reads in csv converts to dataframe
    df_csv = df_csv.drop(df_csv.columns[0], axis=1)  # drops first column with channel labels

    SXR_transimpedance = 112e6               #56e6
    HXR_transimpedance = 9e6                 #4.5e6
    EUV_transimpedance = 7.8e6               #3.9e6

    df.iloc[0:4] = df.iloc[0:4] / SXR_transimpedance
    df_csv.iloc[0:4] = df_csv.iloc[0:4] / SXR_transimpedance

    df.iloc[4] = df.iloc[4] / HXR_transimpedance
    df_csv.iloc[4] = df_csv.iloc[4] / HXR_transimpedance

    df.iloc[5] = df.iloc[5] / EUV_transimpedance
    df_csv.iloc[5] = df_csv.iloc[5] / EUV_transimpedance

    stats_from_csv = []
    stats_from_df = []

    for i, name in enumerate(channel_names):
        current_csv = df_csv.iloc[i].values
        current_df = df.iloc[i].values

        #csv_stats
        current_csv_mean = np.mean(current_csv) * 1e15          #convert to fA
        current_csv_sd = np.std(current_csv) * 1e15
        current_csv_rms = np.sqrt(np.mean(current_csv**2)) * 1e15

        #df stats after 40 Hz removal and detrending
        current_df_mean = np.mean(current_df) * 1e15            #convert to fA
        current_df_sd = np.std(current_df) * 1e15
        current_df_rms = np.sqrt(np.mean(current_df ** 2)) * 1e15

        stats_from_csv.append([name, current_csv_mean, current_csv_rms, current_csv_sd])
        stats_from_df.append([name, current_df_mean, current_df_rms, current_df_sd])

    return stats_from_csv, stats_from_df

def plot_signals_voltages(file_path):
    """
    Plots the raw voltage values obtained from ASIO. It reads a CSV file with 6 rows, each row is a time series of voltage values from each ASIO channel.
    Plots them all on a single graphic with six subplots.

    Arguments:
        file_path (str): Path to the CSV file
    Returns:
        filename (str): Path to the png of the graph
    """

    df = pd.read_csv(file_path, header=None)                      #read CSV converts to dataframe
    df = df.drop(df.columns[0], axis=1)                           #drops first column with channel labels

    x = np.arange(df.shape[1]) * 0.01                             #gets number of columns, this is the time series for x axis, multiplies by 0.01 to convert from centiseconds to s
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    cmap = plt.get_cmap('Dark2')                                  #creates color map
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))  #creates subplots: 6 rows, 1 column
    axes = axes.ravel()

    for i, (name, ax) in enumerate(zip(channel_names, axes)):
        color = cmap(i / len(channel_names))                               #assigns unique color
        axes[i].scatter(x, df.iloc[i], color=color, label=name, s=0.1)     #creates scatter plots for each row in csv
        axes[i].tick_params(axis='x', labelrotation=45)
        axes[i].set_ylabel('Voltage')                                      #sets Y axis label
        axes[i].set_xlabel('Time (s)')                                     #sets X axis label
        axes[i].set_title(f'{name}')                                       #creates title for subplot
        axes[i].grid(True)

    fig.suptitle('ASIO Raw Voltage Values', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    base_dir = os.path.dirname(file_path)                                  #gets directory of input csvs
    filename = os.path.join(base_dir, 'Raw_Voltages.png')                  #combines directory and png name
    plt.savefig(filename)
    plt.close()
    return filename

#test call
#plot_signals_voltages('Gaussian_trended.csv')

def plot_fft(file_path):
    """
      Reads a CSV file with 6 rows, each row is a time series of voltage values from each ASIO channel.
      Generates ffts for each channel. Data has not been mean centered for this FFT, meaning a large spike should
      appear at 0.

        Arguments:
            file_path (str): Path to the CSV file
        Returns:
             filename (str): Path to the png of the FFT graphs with six subplots, one for each channel
    """
    df = pd.read_csv(file_path, header=None)                          #reads in csv converts to dataframe
    df = df.drop(df.columns[0], axis=1)                               #drops first column
    n_samples = df.shape[1]                                           #gets number of columns this is the number of points in fft
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']    #labels for ASIO channels
    sampling_rate = 100                                               #ASIO measuring cadence
    cmap = plt.get_cmap('Dark2')                                      #creates color map
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))      #creates subplots: 6 rows, 1 column
    axes = axes.ravel()                                               #allows for loops to work correctly on subplots

    f = np.fft.fftshift(np.fft.fftfreq(n_samples, d=1 /sampling_rate)) #creates frequency axis for fft, shifts so 0 frequencies is at center of fft
    positive_freqs = f >= 0                                            #limits to positive frequencies

    for i, (name, ax) in enumerate(zip(channel_names, axes)):
        data = df.iloc[i].values                #gets values for each channel
        data = data - np.mean(data)             #mean centers data (removes DC offset)
        freq = np.fft.fft(data)                 #computes fft
        freq_shifted = np.fft.fftshift(freq)    #shifts zero freq to center
        amplitude = np.abs(freq_shifted)        #abs value gives magnitude of freq component present in signal

        #plots fft
        color = cmap(i / len(channel_names))
        axes[i].plot(f[positive_freqs], amplitude[positive_freqs], color=color)
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f'{name}')
        axes[i].grid(True)

    axes[-1].set_xlabel('Frequency (Hz)')                  #labels last x axis in graph
    fig.suptitle('FFTs', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])                 #formatting which leaves room for title
    base_dir = os.path.dirname(file_path)                  #gets directory of input csvs
    filename = os.path.join(base_dir, 'ASIO_FFT.png')      #create png file
    plt.savefig(filename)
    plt.close()
    return filename

#test call
#plot_fft('sin_wave_sampler.csv')

def apply_savgol_filter(file_path, window_length=51, polyorder=3, num_passes = 2):

    """
    This function applies a Savitsky-Golay filter to each of ASIO's channel to identify trends in the data, which
    are then subtracted from the signal. This is applied after the 40 Hz signal is removed from the signal.

       Arguments:
            file_path (str): Path to the CSV file.
            window_length (int): Length of the Savitsky-Golay filter.
            polyorder (int): Polynomial order of the Savitsky-Golay filter.
        Returns:
            savgol_df (pd.Dataframe): df of detrended data with 40 Hz signal removed.
            trend_df (pd.Dataframe): df of the calculated trend. 
    """
    df = pd.read_csv(file_path, header=None)
    df_raw = df.drop(df.columns[0], axis=1) # Renamed to df_raw for clarity

    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']

    detrended_signals = []
    trends = [] # List to store the calculated trends 
    for i in range (len(channel_names)):
        data = df_raw.iloc[i].values
        trend_input = data.copy() # Use a temporary variable for iterative filtering 
        
        # Applying the filter iteratively for multiple passes  
        for j in range(num_passes):
            trend = savgol_filter(trend_input, window_length=window_length, polyorder=polyorder)
            trend_input = trend # Update the input for the next pass with the current trend

        detrended_data = data - trend
        detrended_signals.append(detrended_data)
        trends.append(trend) # Store the final trend after all passes 

    savgol_df = pd.DataFrame(detrended_signals)
    trend_df = pd.DataFrame(trends) # Create a DataFrame for the trends 
    
    return savgol_df, trend_df # Return both detrended data and trend 

#test call
#savgol_df = apply_savgol_filter('Gaussian_trended.csv', window_length=51, polyorder=3, num_passes = 2)
#print(savgol_df)

def remove_room_emi(file_path):
    """
    Removes 40 Hz EMI from all six ASIO channel mean centered time series using FFT, reconstructs time series data using an IFFT. Returns
    a dataframe of voltages for each channel with 40 Hz signal removed. Additionally compares before and after RMS values.

    Arguments:
        file_path (str): Path to the CSV file.

    Returns:
        cleaned_df: pd.DataFrame, cleaned time-domain signals
        rms_results: list of tuples (channel_name, rms_cleaned, rms_original)
    """
    df, _ = apply_savgol_filter(file_path, window_length=51, polyorder=3, num_passes=2)

    #df = pd.read_csv(file_path, header=None)  #reads csv, converts to df
    #df = df.drop(df.columns[0], axis=1)       #drops first column

    n_channels = df.shape[0]                  #gets number of rows
    n_samples = df.shape[1]                   #gets number of columns, this is the number of points in fft
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']       #labels channels
    sampling_rate = 100                                                  #ASIO sampling rate = 100 Hz
    f = np.fft.fftshift(np.fft.fftfreq(n_samples, d=1 / sampling_rate))  #creates frequency axis for fft, shifts so 0 frequencies is at center of fft

    cleaned_signals = []
    rms_results = []

    for i in range(n_channels):
        data = df.iloc[i].values               #gets values for each channel
        data_detrended = data - np.mean(data)  #mean centers the data

        freq = np.fft.fft(data_detrended)      #computes fft
        freq_shifted = np.fft.fftshift(freq)   #shifts zero freq to center


        #remove room EMI around ±40 Hz
        RoomEMI_pos = [40.1 - 0.2, 40.1 + 0.2]  #identifies 40hz spike
        RoomEMI_neg = [-40.1 - 0.2, -40.1 + 0.2]
        RemoveFFT_pos = np.where((f >= RoomEMI_pos[0]) & (f <= RoomEMI_pos[1]))[0] #removes frequencies between indices of tuple above
        RemoveFFT_neg = np.where((f >= RoomEMI_neg[0]) & (f <= RoomEMI_neg[1]))[0]

        freq_shifted[RemoveFFT_pos] = 0         #replaces with 0
        freq_shifted[RemoveFFT_neg] = 0

        cleaned = np.fft.ifft(np.fft.ifftshift(freq_shifted))          #shifts back from 0 center and computes inverse FFT
        cleaned = np.real(cleaned)                                         #removes imaginary component

        #Calculate RMS of original and cleaned signal, both are mean subtracted values
        rms_cleaned = np.sqrt(np.mean(cleaned ** 2))
        rms_original = np.sqrt(np.mean(data_detrended ** 2))

        cleaned_signals.append(cleaned)                                    #adds mean subtracted cleaned signals to a list
        rms_results.append((channel_names[i], rms_cleaned, rms_original))  #adds RMS before and after values for each channel to a list

    cleaned_df = pd.DataFrame(cleaned_signals)                             #converts list of cleaned signals to a dataframe
    return cleaned_df, rms_results

#test calls
#cleaned_df, rms_results = remove_room_emi('Gaussian_trended.csv')
#print(cleaned_df)

#for channel, rms_cleaned, rms_original in rms_results:
    #print(f"{channel} - RMS (Cleaned): {rms_cleaned:.3e}, RMS (Original): {rms_original:.3e}")

def plot_cleaned_signals_time_series(cleaned_df, file_path):
    """
    Uses dataframe of voltages after 40hz has been removed to recreate time series plot, adds mean of original signal back to dataset.

    Arguments:
        cleaned_df (pd.DataFrame): Dataframe of cleaned signals.
        file_path (str): Path to the CSV file.

    Returns:
        filename (str): Path to the png of the FFT graphs with six subplots, one for each channel
    """

    n_channels = cleaned_df.shape[0]                                                       #number of rows in df
    n_samples = cleaned_df.shape[1]                                                        #number of columns in df
    sampling_rate = 100                                                                    #ASIO sampling rate
    time = np.linspace(0, n_samples / sampling_rate, n_samples, endpoint=False)            #creates x axis
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    cmap = plt.get_cmap('Dark2')

    fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (10,10))                      #creates subplots
    axes = axes.ravel()


    for i in range(n_channels):
        ax = axes[i]
        color = cmap(i / len(channel_names))
        #original_mean = np.mean(original_df.iloc[i].values)                                #adds original mean back to signal for comparison with raw voltage values
        ax.scatter(time, cleaned_df.iloc[i].values, color=color, s = 0.1)                   #creates scatter plot
        ax.set_title(f'{channel_names[i]}')
        ax.set_ylabel('Voltage(V)')
        ax.set_xlabel('Time (s)')
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid(True)

    fig.suptitle('EMI-Filtered and Detrended Voltage Values')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    base_dir = os.path.dirname(file_path)                                                  #gets directory of input csvs
    filename = os.path.join(base_dir, "Room_EMI_Removed_Voltages.png")
    plt.savefig(filename)
    plt.close()
    return(filename)

#test call
#plot_cleaned_signals_time_series(cleaned_df)

def plot_cleaned_fft(cleaned_df, file_path):
    """
    Uses dataframe of voltages after 40hz has been removed to create an FFT, similar to FFT function above but operates
    on cleaned signals which have already been mean centered.

    Arguments:
        cleaned_df (pd.DataFrame): DataFrame with 6 rows, each a time series of voltage values for an ASIO channel.
        file_path (str): Path to the original CSV file.

    Returns:
        filename (str): Path to the png of the FFT graphs with six subplots, one for each channel
    """
    n_samples = cleaned_df.shape[1]
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    sampling_rate = 100
    cmap = plt.get_cmap('Dark2')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    axes = axes.ravel()

    f = np.fft.fftshift(np.fft.fftfreq(n_samples, d=1 / sampling_rate))
    positive_freqs = f >= 0

    for i, (name, ax) in enumerate(zip(channel_names, axes)):                #note: not explicitly mean centering because cleaned dataset already is mean centered
        data = cleaned_df.iloc[i].values
        freq = np.fft.fft(data)
        freq_shifted = np.fft.fftshift(freq)
        amplitude = np.abs(freq_shifted)

        #plots FFTs
        color = cmap(i / len(channel_names))
        ax.plot(f[positive_freqs], amplitude[positive_freqs], color=color)    #only plots positive frequencies
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{name}')
        ax.grid(True)

    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle('EMI-filtered and Detrended FFTs', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    base_dir = os.path.dirname(file_path)
    filename = os.path.join(base_dir, 'Cleaned_ASIO_FFT.png')
    plt.savefig(filename)
    plt.close()
    return filename

#test call
#plot_cleaned_fft(cleaned_df, "sin_wave_sampler.csv")

def plot_raw_noise_histograms(file_path):
    """
    Reads a CSV file with 6 rows, each row is a time series of voltage values from each ASIO channel.
    generates noise histogram for each channel.

        Arguments:
            file_path (str): Path to the CSV file.
        Returns:
           filename (str): Path to the png of histogram graphs with six subplots, one for each channel, includes Gaussian best fit over plotted on graphs.
    """
    df = pd.read_csv(file_path, header = None)
    df = df.drop(df.columns[0], axis=1)
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    cmap = plt.get_cmap('Dark2')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))
    axes = axes.ravel()

    for i, (name, ax) in enumerate(zip(channel_names, axes)):
        data = df.iloc[i].values
        noise = data                                       #mean centers data (removes DC offset)
        sigma = np.std(noise)
        color = cmap(i / len(channel_names))

        #plot histogram
        num_bins = 40
        data_min, data_max = noise.min(), noise.max()
        bin_width = (data_max - data_min) / num_bins
        centers = np.linspace(data_min + bin_width / 2, data_max - bin_width / 2, num_bins) #calculates the center of each bin
        edges = np.concatenate(([centers[0] - bin_width / 2], centers + bin_width / 2))     #creates an array of bin edges
        shift = bin_width * 0.03     #calculates a small shift
        plotted_bins = edges + shift #shifts all bins by 4 percent of the width
        ax.hist(noise, bins=plotted_bins, color=color, density = False, alpha = 0.6, edgecolor = 'black', linewidth=0.4)
        ax.tick_params(axis='x', labelrotation=45)

        #plot Gaussian fit
        N = len(data)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 10000)
        p = norm.pdf(x, loc=np.mean(noise), scale=sigma)
        p_scaled = p * N * bin_width
        ax.plot(x, p_scaled, 'r', linewidth=2)

        ax.set_title(f'{name}')
        ax.set_xlabel('(V)')
        ax.set_ylabel('Counts')
        ax.grid(True)

    fig.suptitle('ASIO Noise Histograms', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    base_dir = os.path.dirname(file_path)  #gets directory of input csvs
    filename = os.path.join(base_dir, 'Raw_Noise_Histograms.png')
    plt.savefig(filename)
    plt.close()
    return(filename)

#test call
#plot_raw_noise_histograms('Gaussian_trended.csv')


def plot_cleaned_histogram(df, file_path):
    """
    Reads a df with 6 rows, each row is a time series of voltage values from each ASIO channel.
    Generates a noise histogram for each channel. Similar to histogram plotting function above but operates on cleaned signals which have already been mean centered.

    Arguments:
        df: pd.DataFrame of cleaned time-series data (6 rows, N columns)
        file_path(str): Path to the CSV file.

    Returns:
        filename (str): Path to the png of histogram graphs with six subplots, one for each channel, includes Gaussian best fit over plotted on graphs.
    """
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    cmap = plt.get_cmap('Dark2')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    axes = axes.ravel()

    for i, (name, ax) in enumerate(zip(channel_names, axes)):
        data = df.iloc[i].values
        #noise = data - np.mean(data)  # Remove DC offset
        sigma = np.std(data)
        color = cmap(i / len(channel_names))

        # plot histogram
        num_bins = 45
        data_min, data_max = data.min(), data.max()
        bin_width = (data_max - data_min) / num_bins
        centers = np.linspace(data_min + bin_width / 2, data_max - bin_width / 2, num_bins)  # calculates the center of each bin
        edges = np.concatenate(([centers[0] - bin_width / 2], centers + bin_width / 2))  # creates an array of bin edges
        shift = bin_width * 0.03  # calculates a small shift
        plotted_bins = edges + shift  # shifts all bins by 4 percent of the width
        ax.hist(data, bins=plotted_bins, color=color, density=False, alpha=0.6, edgecolor='black', linewidth=0.4)
        ax.tick_params(axis='x', labelrotation=45)

        #plot Gaussian fit (mean should be ~ 0 since noise is mean-centered)
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

    base_dir = os.path.dirname(file_path)  # gets directory of input csvs
    filename = os.path.join(base_dir, 'Room_EMI_Removed_Noise_Histograms.png')
    plt.savefig(filename)
    plt.close()
    return(filename)

#test call
#plot_cleaned_histogram(cleaned_df, 'Fe55Testing_20250730_2/08_20250730_SXR2_Lights.csv')

def plot_raw_voltage_histograms(file_path):
    """
    Reads a CSV file with 6 rows, each row is a time series of voltage values from each ASIO channel.
    generates noise histogram for each channel.

        Arguments:
            file_path (str): Path to the CSV file.
        Returns:
            filename (str): Path to the png of histogram graphs with six subplots, one for each channel, includes Gaussian best fit over plotted on graphs.
    """
    df = pd.read_csv(file_path, header = None)
    df = df.drop(df.columns[0], axis=1)
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    cmap = plt.get_cmap('Dark2')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))
    axes = axes.ravel()

    for i, (name, ax) in enumerate(zip(channel_names, axes)):
        data = df.iloc[i].values
        sigma = np.std(data)
        color = cmap(i / len(channel_names))

        num_bins = 40
        data_min, data_max = data.min(), data.max()
        bin_width = (data_max - data_min) / num_bins
        centers = np.linspace(data_min + bin_width / 2, data_max - bin_width / 2, num_bins)  #calculates the center of each bin
        edges = np.concatenate(([centers[0] - bin_width / 2], centers + bin_width / 2))  #creates an array of bin edges
        shift = bin_width * 0.03  # calculates a small shift
        plotted_bins = edges + shift  # shifts all bins by 4 percent of the width
        ax.hist(data, bins=plotted_bins, color=color, density=False, alpha=0.6, edgecolor='black', linewidth=0.4)
        ax.tick_params(axis='x', labelrotation=45)

        #plot Gaussian fit
        N = len(data)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 1000)
        p = norm.pdf(x, loc=np.mean(data), scale=sigma)
        p_scaled = p * N * bin_width
        ax.plot(x, p_scaled, 'r', linewidth=2)

        ax.set_title(f'{name}')
        ax.set_xlabel('(V)')
        ax.set_ylabel('Counts')
        ax.grid(True)

    fig.suptitle('ASIO Raw Data Histograms', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    base_dir = os.path.dirname(file_path)  # gets directory of input csvs
    filename = os.path.join(base_dir, 'Raw_Voltages_Histogram.png')
    plt.savefig(filename)
    plt.close()
    return(filename)

def plot_raw_data_and_trend(file_path, window_length=51, polyorder=3, num_passes=2):
    """
    Plots the raw voltage values and the calculated Savitzky-Golay trend for each ASIO channel.

    Arguments:
        file_path (str): Path to the CSV file.
        window_length (int): Window length for the Savitzky-Golay filter.
        polyorder (int): Polynomial order for the Savitzky-Golay filter.
        num_passes (int): Number of passes for the Savitzky-Golay filter.
    Returns:
        filename (str): Path to the png of the graph.
    """

    df_raw = pd.read_csv(file_path, header=None)
    df_raw = df_raw.drop(df_raw.columns[0], axis=1)

    # Call the modified detrending function to get the trend
    _, df_trend = apply_savgol_filter(file_path, window_length, polyorder, num_passes)

    x = np.arange(df_raw.shape[1]) * 0.01
    channel_names = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
    cmap = plt.get_cmap('Dark2')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    axes = axes.ravel()

    for i, (name, ax) in enumerate(zip(channel_names, axes)):
        color = cmap(i / len(channel_names))

        # Plot Raw Data
        axes[i].scatter(x, df_raw.iloc[i], color=color, label='Raw Data', s=0.1)

        # Plot Trend
        axes[i].plot(x, df_trend.iloc[i], color='red', linewidth=2, label='SG Trend')

        axes[i].tick_params(axis='x', labelrotation=45)
        axes[i].set_ylabel('Voltage (V)')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_title(f'{name} - Raw Data and SG Trend')
        axes[i].legend(loc='best')
        axes[i].grid(True)

    fig.suptitle('ASIO Raw Voltage Values with Savitzky-Golay Trend Overlay', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    base_dir = os.path.dirname(file_path)
    filename = os.path.join(base_dir, 'Raw_Voltages_and_Trend.png')
    plt.savefig(filename)
    plt.close()
    return filename

#test call
#plot_raw_voltage_histograms('Gaussian_trended.csv')


