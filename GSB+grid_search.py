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
import time

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

########################
# === TEST SECTION ===
########################
eeg_df = pd.read_csv(r'data/expe1/v1/004_002_2025-03-24-18h53.49.288_ExG.csv')
markers_df = pd.read_csv(r'data/expe1/v1/004_002_2025-03-24-18h53.49.288_Marker.csv')
fs_test = 250
eeg_data = eeg_df[["ch4", "ch1", "ch5"]].values.astype(np.float32)
timestamps = eeg_df["TimeStamp"].values
label_map = {"sw_94": 1, "sw_95": 0}

epochs, labels_test = [], []
epoch_len = int(3.0 * fs_test)
for time, code in zip(markers_df["TimeStamp"], markers_df["Code"]):
    if code in label_map:
        start = np.searchsorted(timestamps, time - 0.5)
        end = start + epoch_len
        if end <= len(eeg_data):
            epochs.append(eeg_data[start:end].T)
            labels_test.append(label_map[code])

epochs = np.stack(epochs, axis=2)
labels_test = np.array(labels_test)

# === BAD EPOCH REJECTION ===
# Step 1: Remove extreme peak-to-peak amplitude
ptp = np.ptp(epochs, axis=1).max(axis=0)  # max P2P value across channels per epoch
ptp_thresh = np.percentile(ptp, 95)
good_idx_ptp = np.where(ptp < ptp_thresh)[0]

# Step 2: Apply variance-based rejection on remaining
epochs_filtered = epochs[:, :, good_idx_ptp]
labels_filtered = labels_test[good_idx_ptp]

var = np.var(epochs_filtered, axis=1).mean(axis=0)
var_thresh = np.percentile(var, 95)
good_idx_var = np.where(var < var_thresh)[0]

# Final clean data
epochs = epochs_filtered[:, :, good_idx_var]
labels_test = labels_filtered[good_idx_var]


epochs = bandpass(epochs, 8, 30, fs_test)
test = apply_mix(W, epochs)
Xy_test = extract_features(test, (0, -1), None, fs_test)
X_test = scaler.transform(Xy_test.drop('label', axis=1))
y_test = labels_test[:X_test.shape[0]]

########################
# === TRAIN CLASSIFIER ===
########################
def tune_train_test_pipeline(base_clf, clf_params, x_train, y_train, x_test, y_test):
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tuned_clf = GridSearchCV(estimator=base_clf, param_grid=clf_params,
                              scoring='accuracy', cv=cv_splitter)
    tuned_clf.fit(x_train, y_train)

    print('Best params:', tuned_clf.best_params_)
    print('Train acc:', accuracy_score(y_train, tuned_clf.predict(x_train)))
    print('Test acc:', accuracy_score(y_test, tuned_clf.predict(x_test)))
    print('\nTest Classification Report:')
    print(classification_report(y_test, tuned_clf.predict(x_test)))

    cm = confusion_matrix(y_test, tuned_clf.predict(x_test))
    ConfusionMatrixDisplay(cm, display_labels=['Left', 'Right']).plot(cmap='Blues')
    plt.title("Confusion Matrix - Test")
    plt.show()
    return tuned_clf

clf_params = {
    'loss': ['log_loss'],
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [2, 3]
}

base_clf = GBC()
model = tune_train_test_pipeline(base_clf, clf_params, X_train, y_train, X_test, y_test)


