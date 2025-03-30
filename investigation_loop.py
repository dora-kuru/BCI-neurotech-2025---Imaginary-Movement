# Full rewrite of your code integrating workshop-based changes

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.stats import zscore, kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd

# === Load dataset ===
mat_path = r'C:\Users\dorak\Documents\IMbci\BCICIV_1\BCICIV_calib_ds1d.mat'
mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)

cnt = mat['cnt'].astype(np.float32)
mrk = mat['mrk']
nfo = mat['nfo']
fs = int(nfo.fs)
channel_names = list(nfo.clab)

# === Epoching from 0.5s to 2.5s ===
start, end = int(0.5 * fs), int(2.5 * fs)
window = np.arange(start, end)
nsamples = len(window)
nchannels = cnt.shape[1]

event_onsets = mrk.pos
event_labels = mrk.y
labels = np.array([1 if y == 1 else 0 for y in event_labels])

trials = np.zeros((nchannels, nsamples, len(event_onsets)))
for i, onset in enumerate(event_onsets):
    if onset + end < cnt.shape[0]:
        trials[:, :, i] = cnt[onset + window, :].T

# === Bandpass filtering (8–30Hz) ===
def bandpass(trials, lo, hi, fs):
    b, a = butter(4, [lo/(fs/2), hi/(fs/2)], btype='band')
    for i in range(trials.shape[2]):
        trials[:, :, i] = filtfilt(b, a, trials[:, :, i], axis=1)
    return trials

trials = bandpass(trials, 8, 30, fs)

# === Split trials ===
trials_left = trials[:, :, labels == 1]
trials_right = trials[:, :, labels == 0]

# === CSP ===
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

W = csp(trials_left, trials_right)

# === Apply CSP ===
def apply_mix(W, trials):
    return np.array([W.T @ trials[:, :, i] for i in range(trials.shape[2])]).transpose(1, 2, 0)

trials_csp = apply_mix(W, trials)

# === Feature extraction (RMS, log-var, kurtosis, skew) ===
def extract_features(trials_csp, label):
    n_trials = trials_csp.shape[2]
    features = []
    for i in range(n_trials):
        trial = trials_csp[:, :, i]
        f = []
        for ch in range(trial.shape[0]):
            signal = trial[ch, :]
            f.extend([
                np.log(np.var(signal)),
                np.sqrt(np.mean(signal**2)),
                kurtosis(signal),
                skew(signal)
            ])
        features.append(f + [label])
    return pd.DataFrame(features)

features_left = extract_features(apply_mix(W, trials_left), 1)
features_right = extract_features(apply_mix(W, trials_right), 0)

# === Combine & Normalize ===
data = pd.concat([features_left, features_right], ignore_index=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X = (X - X.mean()) / X.std()

# === Train/test split + LDA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
lda = LDA()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ CSP + Feature Engineering + LDA Accuracy: {acc * 100:.2f}%")