import pandas as pd
import numpy as np
import mne

# --- Load EEG and marker data ---
eeg_path = r'data/expe1/v1/002_002_2025-03-24-17h47.37.040_ExG.csv'
marker_path = r'data/expe1/v1/002_002_2025-03-24-17h47.37.040_Marker.csv'
fs = 250  # sampling frequency

eeg_df = pd.read_csv(eeg_path)
markers_df = pd.read_csv(marker_path)

# --- Prepare EEG data ---
eeg_data = eeg_df.loc[:, 'ch1':'ch8'].to_numpy().T  # shape: (n_channels, n_times)
ch_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)

# --- Create events array ---
# Convert marker timestamps to sample indices
marker_times = markers_df['TimeStamp'].values
marker_codes = markers_df['Code'].str.extract(r'(\d+)').astype(int).values

# Assuming EEG and markers have same reference timestamp (aligning on raw's first time)
start_time = eeg_df['TimeStamp'].iloc[0]
event_samples = ((marker_times - start_time) * fs).astype(int)

events = np.column_stack((event_samples, np.zeros(len(event_samples), dtype=int), marker_codes))

# --- Define event_id mapping ---
event_id = {'left': 0, 'right': 1}

# --- Create epochs ---
offset = 2 + 2 # instruct_time + prepare_time
tmin = -0.5 + offset  # pre-stimulus time in seconds ()
tmax = 3 + offset  # post-stimulus time in seconds (+ instruct_time + prepare_time)
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=(tmin, offset), preload=True)

# rename channels = to complete yourself
# extra = set channel_types = 'eeg', 'misc'
# ch_name_dct = {'ch1': 'bad', }
# epochs.rename_channels()
# pick channel with new name = location
epochs.pick_channels(["ch4", "ch1", "ch5"]) # Cz, C3, C4


# Now you can access epochs['event_0'] or epochs['event_1'], etc.
print(epochs)

# Get data and labels from the Epochs object
X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
y = epochs.events[:, 2]  # The event codes (labels) are in the third column

print(f'X shape : {X.shape}')
print(f'y shape : {y.shape}')

# wrap all this in a function 
# load_data(path, ) -> mne.Epochs or X, y 

# A. subject-dependent model

# Train on subject, test on same subject = in practice NEEDS CALIBRATION DATA

# for sub in subjects:
#     X, y = load_data(path_for_subject)
#     model.fit(X, y) or cross_validate(X,y)


# B. subject-indepent model 

# = same model for each subject 

# Train on subject, test on same subject = in practice NEEDS CALIBRATION DATA
# for loop over subjects and concatenate X, y = subject-indepent model

# LOAD all test data

# Xfull, yfull = [], []
# for sub in subjects:
#     X, y = load_data(path_for_subject)
#     Xfull.append(X)
#     yfull.append(y)
# Xfull = np.stack(Xfull) # shape (n_trials, n_channels, n_timepoints)
# yfull = np.stack(yfull) # shape (n_trials) e.g 30 x 2 x 5 = 300

# 1. already trained model
# model.score(Xfull, yfull, scoreing='balanced_accuracy')
