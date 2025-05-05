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
# === TEST SECTION (Experimental CSV) ===
########################
eeg_path = r'data/expe1/v1/002_002_2025-03-24-17h47.37.040_ExG.csv'
marker_path = r'data/expe1/v1/002_002_2025-03-24-17h47.37.040_Marker.csv'

eeg_df = pd.read_csv(eeg_path)
markers_df = pd.read_csv(marker_path)
fs_test = 250

label_map = {"sw_94": 1, "sw_95": 0}
eeg_channels = eeg_df[["ch4", "ch1", "ch5"]].values.astype(np.float32)
timestamps = eeg_df["TimeStamp"].values

marker_times = markers_df["TimeStamp"].values
marker_codes = markers_df["Code"].values

start_shift = 0.5
epoch_duration = 3.0
n_samples_epoch = int(epoch_duration * fs_test)
epochs, labels = [], []

for time, code in zip(marker_times, marker_codes):
    if code in label_map:
        start = np.searchsorted(timestamps, time - start_shift)
        end = start + n_samples_epoch
        if end <= len(eeg_channels):
            epochs.append(eeg_channels[start:end, :])
            labels.append(label_map[code])

epochs = np.stack(epochs)
labels = np.array(labels)

epochs_test = np.transpose(epochs, (2, 1, 0))
epochs_test = bandpass(epochs_test, 8, 30, fs_test)

from sklearn.decomposition import FastICA

def clean_epochs_ica(epochs, threshold=150):
    cleaned = []
    for i in range(epochs.shape[2]):
        trial = epochs[:, :, i]
        ica = FastICA(n_components=trial.shape[0], random_state=42)
        try:
            S = ica.fit_transform(trial.T).T  # Shape: (components, samples)
            # Remove components with very high amplitude (likely artifacts)
            keep = [j for j in range(S.shape[0]) if np.max(np.abs(S[j])) < threshold]
            S_clean = S[keep, :]
            trial_clean = ica.mixing_[:, keep] @ S_clean
            cleaned.append(trial_clean)
        except Exception as e:
            print(f"⚠️ ICA failed on trial {i}: {e}")
            continue
    cleaned = np.stack(cleaned, axis=2)
    return cleaned

epochs_test = clean_epochs_ica(epochs_test, threshold=150)


W_loaded = np.load("trained_csp.npy")
lda_loaded = joblib.load("trained_lda.pkl")

csp_test = apply_mix(W_loaded, epochs_test)
X_test, labels = extract_features(csp_test, labels)
X_test = (X_test - X.mean()) / X.std()
y_pred_test = lda_loaded.predict(X_test)

print("\n🧪 Experimental Test Results:")
print(classification_report(labels, y_pred_test, target_names=["Right", "Left"], zero_division=0))

logvar_test = []
for i in range(X_test.shape[0]):
    logvar_test.append([X_test.iloc[i, 0], X_test.iloc[i, -4]])

logvar_test = np.array(logvar_test)
plt.figure(figsize=(7, 6))
plt.scatter(logvar_test[:, 0], logvar_test[:, 1], c=labels, cmap='coolwarm', edgecolors='k')
plt.xlabel('Log-Var CSP Component 1')
plt.ylabel('Log-Var CSP Component N')
plt.title('CSP Feature Space (Test Data)')
plt.grid(True)
plt.tight_layout()
plt.show()

from matplotlib.colors import ListedColormap
lda_vis = LDA(priors=[0.5, 0.5])
lda_vis.fit(logvar_test, labels)
xx, yy = np.meshgrid(
    np.linspace(logvar_test[:, 0].min() - 1, logvar_test[:, 0].max() + 1, 300),
    np.linspace(logvar_test[:, 1].min() - 1, logvar_test[:, 1].max() + 1, 300)
)
Z = lda_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
plt.scatter(logvar_test[:, 0], logvar_test[:, 1], c=labels, cmap='coolwarm', edgecolors='k')
plt.xlabel('Log-Var CSP Component 1')
plt.ylabel('Log-Var CSP Component N')
plt.title('LDA Decision Boundary on CSP Features')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n🧠 Real-Time Style Predictions (one-by-one):")
correct = 0

for i in range(len(X_test)):
    trial_features = X_test.iloc[i].values.reshape(1, -1)
    pred = lda_loaded.predict(trial_features)[0]
    true_label = labels[i]
    
    match = "✅" if pred == true_label else "❌"
    if pred == true_label:
        correct += 1

    print(f"Trial {i+1:>2} | True: {'Left' if true_label==1 else 'Right':>5} | Pred: {'Left' if pred==1 else 'Right':>5} | {match}")

print(f"\n📊 Simulated Real-Time Accuracy: {correct}/{len(X_test)} = {correct/len(X_test)*100:.2f}%")
