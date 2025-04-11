import explorepy
import time
from explorepy.stream_processor import TOPICS
import numpy as np
import mne 
def main():
    #Headset Setup
    
    headset_name = 'Explore_849D' 
    sample_rate = 250
    channel_dict = {
            0: 'TP9',
            1: 'Fz', 2: 'C3', 3: 'Cz', 4: 'C4', 5: 'P3', 6: 'P4', 7: 'Fz', 8: 'TP10'
        }
    explorer = explorepy.Explore()
    explorer.connect(device_name=headset_name)
    explorer.set_sampling_rate(sample_rate)

    global eeg_list
    eeg_list = []

    def eeg_callback(packet):
        global time_data
        t_vector , exg_data = packet.get_data()
        eeg_list.append(exg_data)
    
    explorer.stream_processor.subscribe(callback=eeg_callback, topic=TOPICS.raw_ExG)

    print('5 sec')
    time.sleep(10)

    #as np
    global eeg_raw
    eeg_raw = np.concatenate(eeg_list, axis= -1)
    print(eeg_raw)

    #as mne
    info = mne.create_info(ch_names=['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8'], sfreq=250, ch_types='eeg')

    global mne_data
    mne_data = mne.io.RawArray(eeg_raw, info)
    




    
main()
