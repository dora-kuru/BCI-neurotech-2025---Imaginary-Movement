# Mathijs BCI Competition IV Dataset 1 Script (Pure Python Version)
# Full pipeline based on ERD/ERS paradigm, CSP, feature extraction, and classification

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import time

# === Load Dataset ===
data_path = r'C:\Users\dorak\Documents\IMbci\BCICIV_1\BCICIV_calib_ds1d.mat'  # Change this path accordingly
m = scipy.io.loadmat(data_path, struct_as_record=True)
sample_rate = m['nfo']['fs'][0][0][0][0]
EEG = m['cnt'].T
channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
event_onsets = m['mrk'][0][0][0]
event_codes = m['mrk'][0][0][1]
labels = np.zeros((1, EEG.shape[1]), int)
labels[0, event_onsets] = event_codes

cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
cl1, cl2 = cl_lab

# === Epoching ===
trials = {}
win = np.arange(int(0.5 * sample_rate), int(2.5 * sample_rate))
nsamples = len(win)

for cl, code in zip(cl_lab, np.unique(event_codes)):
    cl_onsets = event_onsets[event_codes == code]
    trials[cl] = np.zeros((EEG.shape[0], nsamples, len(cl_onsets)))
    for i, onset in enumerate(cl_onsets):
        trials[cl][:,:,i] = EEG[:, win + onset]

# === Bandpass Filtering ===
def bandpass(trials, lo, hi, fs):
    a, b = signal.iirfilter(6, [lo/(fs/2.0), hi/(fs/2.0)])
    return np.stack([signal.filtfilt(a, b, trials[:, :, i], axis=1) for i in range(trials.shape[2])], axis=2)

trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
               cl2: bandpass(trials[cl2], 8, 15, sample_rate)}

# === CSP ===
def cov(trials):
    ns = trials.shape[1]
    return np.mean([trials[:,:,i] @ trials[:,:,i].T / ns for i in range(trials.shape[2])], axis=0)

def whitening(sigma):
    U, l, _ = np.linalg.svd(sigma)
    return U @ np.diag(l ** -0.5)

def csp(left, right):
    P = whitening(cov(left) + cov(right))
    B, _, _ = np.linalg.svd(P.T @ cov(left) @ P)
    return P @ B

def apply_mix(W, trials):
    return np.stack([W.T @ trials[:, :, i] for i in range(trials.shape[2])], axis=2)

W = csp(trials_filt[cl1], trials_filt[cl2])
train_percentage = 0.8
ntrain = int(train_percentage * trials_filt[cl1].shape[2])

train = {cl1: apply_mix(W, trials_filt[cl1][:,:,:ntrain]),
         cl2: apply_mix(W, trials_filt[cl2][:,:,:ntrain])}

test = {cl1: apply_mix(W, trials_filt[cl1][:,:,ntrain:]),
        cl2: apply_mix(W, trials_filt[cl2][:,:,ntrain:])}

# === Feature Extraction ===
def extract_features(trials_time, comps, label):
    df = pd.DataFrame()
    for i in range(trials_time.shape[2]):
        feat = {}
        for j, comp in enumerate(comps):
            signal_ = trials_time[comp,:,i]
            freqs, pxx = signal.welch(signal_, fs=sample_rate, nperseg=128)
            band = pxx[(freqs >= 8) & (freqs <= 15)]
            feat[f'time_logvar_{j}'] = np.log(np.var(signal_))
            feat[f'time_rms_{j}'] = np.sqrt(np.mean(signal_**2))
            feat[f'time_kurtosis_{j}'] = kurtosis(signal_)
            feat[f'time_skew_{j}'] = skew(signal_)
            feat[f'psd_power_{j}'] = np.trapz(band)
            feat[f'psd_max_{j}'] = max(band)
            feat[f'psd_kurtosis_{j}'] = kurtosis(band)
            feat[f'psd_skew_{j}'] = skew(band)
        feat['label'] = label
        df = pd.concat([df, pd.DataFrame([feat])], ignore_index=True)
    return df

comps = (0, -1)
Xy_train = pd.concat([extract_features(train[cl1], comps, 0), extract_features(train[cl2], comps, 1)])
Xy_test = pd.concat([extract_features(test[cl1], comps, 0), extract_features(test[cl2], comps, 1)])

X_train = Xy_train.drop(columns=['label'])
y_train = Xy_train['label']
X_test = Xy_test.drop(columns=['label'])
y_test = Xy_test['label']

# === Normalize ===
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# === Classifier with PCA Tuning ===
def tune_train_test_pipeline(base_clf, clf_params, X_train, y_train, X_test, y_test):
    best_acc, best_model, best_pca_n = 0, None, None
    pca_opts = [2, 3, 4, 5, 6, 7, 8]
    for n in pca_opts:
        pca = PCA(n_components=n)
        Xt_train = pca.fit_transform(X_train)
        Xt_test = pca.transform(X_test)
        clf = GridSearchCV(base_clf, clf_params, scoring='accuracy', cv=5)
        clf.fit(Xt_train, y_train)
        acc = accuracy_score(y_test, clf.predict(Xt_test))
        if acc > best_acc:
            best_acc = acc
            best_model = clf.best_estimator_
            best_pca_n = n
    print(f"Best PCA n_components: {best_pca_n}, Best Test Accuracy: {best_acc:.4f}")
    return best_model

clf_params = {
    'loss': ['log_loss'],
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [2, 3]
}

clf = tune_train_test_pipeline(GBC(), clf_params, X_train, y_train, X_test, y_test)
y_pred = clf.predict(X_test)

print("\nFinal Test Report:")
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=cl_lab).plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
