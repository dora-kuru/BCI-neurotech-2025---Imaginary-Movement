import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
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

def psd(trials, fs=250):
    nperseg = 128
    freqs = np.fft.rfftfreq(nperseg, 1/fs)
    psd_all = np.zeros((trials.shape[0], len(freqs), trials.shape[2]))
    for i in range(trials.shape[2]):
        for ch in range(trials.shape[0]):
            f, Pxx = welch(trials[ch,:,i], fs=fs, nperseg=nperseg)
            psd_all[ch,:,i] = np.interp(freqs, f, Pxx)
    return psd_all, freqs

def extract_features(trials_time, CSP_components: tuple, label, fs=250):
    df = pd.DataFrame(columns=[
    'psd_power_1', 'psd_max_1', 'psd_kurtosis_1', 'psd_skew_1',
    'time_logvar_1', 'time_rms_1', 'time_kurtosis_1', 'time_skew_1',
    'psd_power_2', 'psd_max_2', 'psd_kurtosis_2', 'psd_skew_2',
    'time_logvar_2', 'time_rms_2', 'time_kurtosis_2', 'time_skew_2',
    'label'
])

    component_1, component_2 = CSP_components
    trials_PSDs, freqs = psd(trials_time, fs=fs)
    for trial in range(trials_time.shape[2]):
        row = []
        for comp in [component_1, component_2]:
            psd_comp = trials_PSDs[comp,:,trial][(freqs>=8) & (freqs<=15)]
            power = np.trapz(psd_comp)
            peak = np.max(psd_comp)
            kurt_p = kurtosis(psd_comp)
            skew_p = skew(psd_comp)
            time_signal = trials_time[comp,:,trial]
            logvar = np.log(np.var(time_signal))
            rms = np.sqrt(np.mean(time_signal**2))
            kurt_t = kurtosis(time_signal)
            skew_t = skew(time_signal)
            row.extend([power, peak, kurt_p, skew_p, logvar, rms, kurt_t, skew_t])
        row.append(label)
        df.loc[len(df)] = row
    return df

########################
# === TRAINING SECTION ===
########################
def train_model():
    mat = loadmat(r'BCICIV_1/BCICIV_calib_ds1d.mat', struct_as_record=False, squeeze_me=True)
    cnt = mat['cnt'].astype(np.float32)
    mrk = mat['mrk']
    nfo = mat['nfo']
    fs = int(nfo.fs)
    channels = [i for i, ch in enumerate(nfo.clab) if ch in ['C3', 'Cz', 'C4']]
    cnt = cnt[:, channels]

    start, end = int(0.0 * fs), int(2.5 * fs)
    labels_raw = np.array([1 if y == 1 else 0 for y in mrk.y])
    trials = np.stack([cnt[mrk.pos[i]+start:mrk.pos[i]+end, :].T for i in range(len(mrk.pos)) if mrk.pos[i]+end < cnt.shape[0]], axis=2)
    labels_raw = labels_raw[:trials.shape[2]]

    trials = bandpass(trials, 8, 30, fs)
    trials_left = trials[:,:,labels_raw==1]
    trials_right = trials[:,:,labels_raw==0]

    W = csp(trials_left, trials_right)
    train = {0: apply_mix(W, trials_left), 1: apply_mix(W, trials_right)}
    Xy_train_0 = extract_features(train[0], (0, -1), 0, fs)
    Xy_train_1 = extract_features(train[1], (0, -1), 1, fs)
    Xy_train = pd.concat([Xy_train_0, Xy_train_1], ignore_index=True)

    X_train = Xy_train.drop('label', axis=1)
    y_train = Xy_train['label']
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    clf_params = {
        'loss': ['log_loss'],
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [2, 3]
    }

    base_clf = GBC()
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tuned_clf = GridSearchCV(estimator=base_clf, param_grid=clf_params, scoring='accuracy', cv=cv_splitter)
    tuned_clf.fit(X_train, y_train)

    print('Best params:', tuned_clf.best_params_)
    print('Train acc:', accuracy_score(y_train, tuned_clf.predict(X_train)))

    joblib.dump(tuned_clf, "trained_gbc.pkl")
    np.save("trained_csp.npy", W)
    joblib.dump(scaler, "scaler.pkl")

########################
# === REAL-TIME PREDICTION SECTION ===
########################


# Load trained models
W_loaded = np.load("trained_csp.npy")
gbc_loaded = joblib.load("trained_gbc.pkl")
scaler_loaded = joblib.load("scaler.pkl")

# Bandpass filter function
def bandpass(trials, lo, hi, fs):
    b, a = butter(4, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, trials, axis=0)

# PSD function
def psd(trials, fs=250):
    nperseg = 128
    freqs = np.fft.rfftfreq(nperseg, 1/fs)
    psd_all = np.zeros((trials.shape[0], len(freqs), trials.shape[2]))
    for i in range(trials.shape[2]):
        for ch in range(trials.shape[0]):
            f, Pxx = welch(trials[ch,:,i], fs=fs, nperseg=nperseg)
            psd_all[ch,:,i] = np.interp(freqs, f, Pxx)
    return psd_all, freqs

# Feature extraction function
def extract_features(trials_time, CSP_components: tuple, label, fs=250):
    df = pd.DataFrame(columns=[
        'psd_power_1', 'psd_max_1', 'psd_kurtosis_1', 'psd_skew_1',
        'time_logvar_1', 'time_rms_1', 'time_kurtosis_1', 'time_skew_1',
        'psd_power_2', 'psd_max_2', 'psd_kurtosis_2', 'psd_skew_2',
        'time_logvar_2', 'time_rms_2', 'time_kurtosis_2', 'time_skew_2',
        'label'
    ])

    component_1, component_2 = CSP_components
    trials_PSDs, freqs = psd(trials_time, fs=fs)
    for trial in range(trials_time.shape[2]):
        row = []
        for comp in [component_1, component_2]:
            psd_comp = trials_PSDs[comp,:,trial][(freqs>=8) & (freqs<=15)]
            power = np.trapz(psd_comp)
            peak = np.max(psd_comp)
            kurt_p = kurtosis(psd_comp)
            skew_p = skew(psd_comp)
            time_signal = trials_time[comp,:,trial]
            logvar = np.log(np.var(time_signal))
            rms = np.sqrt(np.mean(time_signal**2))
            kurt_t = kurtosis(time_signal)
            skew_t = skew(time_signal)
            row.extend([power, peak, kurt_p, skew_p, logvar, rms, kurt_t, skew_t])
        row.append(label)
        df.loc[len(df)] = row
    return df

# Real-time data acquisition and prediction
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
            csp_data = W_loaded.T @ eeg_data
            
            # Extract features
            features_df = extract_features(csp_data, (0, -1), None, sample_rate)
            features = features_df.drop('label', axis=1)
            
            # Standardize features
            features = scaler_loaded.transform(features)
            
            # Predict
            prediction = gbc_loaded.predict(features)
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
