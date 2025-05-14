import explorepy
from explorepy.stream_processor import TOPICS
import numpy as np
import time


def parse_eeg_buffer(eeg_buffer, sampling_rate=250, window_size=1.0, selected_channels=None):
    """
    Converts raw EEG buffer into a structured NumPy array for feature extraction.

    Parameters:
        eeg_buffer (list): List of EEG data arrays from the callback.
        sampling_rate (int): Sampling rate in Hz.
        window_size (float): Duration of the window in seconds.
        selected_channels (list): Indices of channels to include (e.g., [1, 2, 3]).

    Returns:
        np.ndarray: EEG data array of shape (channels, samples) or None if insufficient data.
    """
    n_samples = int(window_size * sampling_rate)
    if len(eeg_buffer) < n_samples:
        return None

    # Concatenate and transpose to shape (channels, samples)
    eeg_data = np.concatenate(eeg_buffer[-n_samples:], axis=0).T

    if selected_channels is not None:
        eeg_data = eeg_data[selected_channels, :]

    return eeg_data


def main():

    headset_name = 'Explore_84FE'  #change the name if necessary
    sample_rate = 250
    channel_dict = {
        0: 'TP9', 1: 'Fz', 2: 'C3', 3: 'Cz', 4: 'C4', 5: 'P3', 6: 'P4', 7: 'Fz', 8: 'TP10'
    }
    explorer = explorepy.Explore()
    explorer.connect(device_name=headset_name)
    explorer.set_sampling_rate(sample_rate)
    global eeg_list
    eeg_list = []

    def eeg_callback(packet):
        global eeg_list
        t_vector, exg_data = packet.get_data()
        eeg_list.append(exg_data)

    explorer.stream_processor.subscribe(callback=eeg_callback, topic=TOPICS.raw_ExG)
    time.sleep(5)

    try:
        while True:
            eeg_data =parse_eeg_buffer(eeg_buffer=eeg_list, sampling_rate=sample_rate,window_size=1.0, selected_channels=[2,3,4])
            time.sleep(1)
            print(eeg_data)
            #Start Pipeline
            
    except KeyboardInterrupt:
        explorer.stop_recording()
        explorer.disconnect()
        print('Stopped.')
   


    
main()