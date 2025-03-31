import pandas as pd
import numpy as np
import mne

# --- Load EEG and marker data ---
eeg_path = r'C:\Users\dorak\Documents\IMbci\data\expe1\v1\002_002_2025-03-24-17h47.37.040_ExG.csv'
marker_path = r'C:\Users\dorak\Documents\IMbci\data\expe1\v1\002_002_2025-03-24-17h47.37.040_Marker.csv'
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






# from https://mne.tools/stable/generated/mne.decoding.CSP.html



# sphinx_gallery_thumbnail_number = 6

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mne
from mne.datasets import sample
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)




clf = make_pipeline(
    Scaler(epochs.info),
    Vectorizer(),
    LogisticRegression(solver="liblinear"),  # liblinear is faster than lbfgs
)

scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None)

# Mean scores across cross-validation splits
score = np.mean(scores, axis=0)
print(f"Spatio-temporal: {100 * score:0.1f}%")



csp = CSP(n_components=3, norm_trace=False)
clf_csp = make_pipeline(csp, LinearModel(LogisticRegression(solver="liblinear")))
scores = cross_val_multiscore(clf_csp, X, y, cv=5, n_jobs=None)
print(f"CSP: {100 * scores.mean():0.1f}%")
