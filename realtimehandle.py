import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
import explorepy
import time
from explorepy.stream_processor import TOPICS
import mne

########################
# === SHARED FUNCTIONS ===
########################
def bandpass(trials, lo, hi, fs):
    b, a = butter(4, [lo/(fs/2), hi/(fs/2)], btype='band')
    for i in range(trials.shape[2]):
        trials[:, :, i] = filtfilt(b, a, trials[:, :, i], axis=1)
    return trials

def cov(trials):
    return np.mean([t @ t.T / t.shape[1] for t in np.moveaxis(trials, 2, 0)], axis=0)

def whitening(sigma):
    U, l, _ = np.linalg.svd(sigma)
    return U @ np.diag(l ** -0.5)

def csp(trials_l, trials_r):
    cov_l = cov(trials_l)
    cov_r = cov(trials_r)
    P = whitening(cov_l + cov_r)
    B, _, _ = np.linalg.svd(P.T @ cov_l @ P)
    return P @ B

def apply_mix(W, trials):
    return np.array([W.T @ trials[:, :, i] for i in range(trials.shape[2])]).transpose(1, 2, 0)

def extract_features(trials_csp, labels_in):
    n_trials = trials_csp.shape[2]
    features = []
    cleaned_labels = []
    for i in range(n_trials):
        trial = trials_csp[:, :, i]
        if np.any(np.isnan(trial)) or np.max(np.abs(trial)) > 500:
            continue
        f = []
        for ch in range(trial.shape[0]):
            signal = trial[ch, :]
            f.extend([
                np.log(np.var(signal)),
                np.sqrt(np.mean(signal**2)),
                kurtosis(signal),
                skew(signal)
            ])
        features.append(f)
        cleaned_labels.append(labels_in[i])
    return pd.DataFrame(features), np.array(cleaned_labels)

########################
# === TRAINING SECTION ===
########################
mat_path = r'BCICIV_1\BCICIV_calib_ds1d.mat'
mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
cnt = mat['cnt'].astype(np.float32)
mrk = mat['mrk']
nfo = mat['nfo']
fs_train = int(nfo.fs)

channels_of_interest = ['C3', 'Cz', 'C4']
channel_indices = [i for i, ch in enumerate(nfo.clab) if ch in channels_of_interest]
cnt = cnt[:, channel_indices]

start_offset, end_offset = int(0.0 * fs_train), int(2.5 * fs_train)
window = np.arange(start_offset, end_offset)
nsamples = len(window)
nchannels = cnt.shape[1]
event_onsets = mrk.pos
event_labels = mrk.y
labels = np.array([1 if y == 1 else 0 for y in event_labels])

trials = np.zeros((nchannels, nsamples, len(event_onsets)))
for i, onset in enumerate(event_onsets):
    if onset + end_offset < cnt.shape[0]:
        trials[:, :, i] = cnt[onset + window, :].T

trials = bandpass(trials, 8, 30, fs_train)
trials_left = trials[:, :, labels == 1]
trials_right = trials[:, :, labels == 0]

W = csp(trials_left, trials_right)
trials_csp = apply_mix(W, trials)
features_left, labels_left = extract_features(apply_mix(W, trials_left), np.ones(trials_left.shape[2]))
features_right, labels_right = extract_features(apply_mix(W, trials_right), np.zeros(trials_right.shape[2]))
data = pd.concat([features_left, features_right], ignore_index=True)
labels_combined = np.concatenate([labels_left, labels_right])
X = data
y = labels_combined
X = (X - X.mean()) / X.std()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
lda = LDA(priors=[0.5, 0.5])
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ CSP + Feature Engineering + LDA Accuracy: {acc * 100:.2f}%")

joblib.dump(lda, "trained_lda.pkl")
np.save("trained_csp.npy", W)

########################
# === REAL-TIME PREDICTION SECTION ===
########################
def main():
    # Headset Setup
    headset_name = 'Explore_84FE'
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
        
        if len(eeg_list) >= sample_rate * 3:  # 3 seconds of data
            eeg_data = np.concatenate(eeg_list, axis=-1)
            eeg_data = eeg_data[:, :sample_rate * 3]  # Keep only the last 3 seconds
            eeg_list.clear()
            
            # Preprocess data
            eeg_data = bandpass(eeg_data, 8, 30, sample_rate)
            
            # Apply CSP
            csp_data = W.T @ eeg_data
            
            # Extract features
            features = extract_features(csp_data)
            
            # Standardize features
            features = (features - X.mean()) / X.std()
            
            # Predict
            prediction = lda.predict(features)
            print(f"Prediction: {'Left' if prediction[0] == 1 else 'Right'}")

    explorer.stream_processor.subscribe(callback=eeg_callback, topic=TOPICS.raw_ExG)
    print('Streaming data... Press Ctrl+C to stop.')
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        explorer.stop_recording()
        explorer.disconnect()
        print('Stopped.')

if __name__ == "__main__":
    main()
