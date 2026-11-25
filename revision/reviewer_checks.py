"""Reproduce the supplemental reviewer-response analyses from the submitted generator."""
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from joblib import parallel_backend
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

SEED = 7
N_SAMPLES = 400
FS = 2000
N_TIME = 256
SNR_DB = 8
N_PNN_MODES = 4
MZI_PAIRS = [(0, 1), (2, 3), (1, 2), (0, 1), (2, 3), (1, 2)]
N_MZIS = len(MZI_PAIRS)
MLST_PNN_RESTARTS = 20
MLST_PNN_MAXITER = 800

def make_tone(fs, n, f0, snr_db, rng):
    t = np.arange(n) / fs

    clean = np.sin(2 * np.pi * f0 * t)

    signal_power = np.mean(clean**2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=n
    )

    return clean + noise

def make_chirp(fs, n, f_start, f_end, snr_db, rng):
    t = np.arange(n) / fs

    # Use the time spanned by the sampled points.
    T = t[-1] if t[-1] > 0 else 1.0
    chirp_rate = (f_end - f_start) / T

    phase = 2 * np.pi * (
        f_start * t
        + 0.5 * chirp_rate * t**2
    )

    clean = np.sin(phase)

    signal_power = np.mean(clean**2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=n
    )

    return clean + noise

def features_from_signal(x, fs):
    # One-sided FFT
    fft_values = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)

    # IMPORTANT:
    # Use POWER spectrum so the implementation matches the manuscript:
    # P[k] = |X[k]|^2
    power = np.abs(fft_values)**2

    # Numerical safeguard
    power = power + 1e-15

    # Normalized spectral power distribution
    p = power / np.sum(power)

    spectral_centroid = np.sum(freqs * p)

    spectral_bandwidth = np.sqrt(
        np.sum(
            ((freqs - spectral_centroid)**2) * p
        )
    )

    peak_frequency = freqs[np.argmax(power)]

    time_domain_energy = np.mean(x**2)

    return np.array(
        [
            spectral_centroid,
            spectral_bandwidth,
            peak_frequency,
            time_domain_energy,
        ],
        dtype=float
    )

def build_dataset(
    n_samples=N_SAMPLES,
    fs=FS,
    n=N_TIME,
    snr_db=SNR_DB,
    seed=SEED
):
    if n_samples % 2 != 0:
        raise ValueError("n_samples must be even for a balanced binary dataset.")

    rng = np.random.default_rng(seed)

    X_features = []
    y = []

    # Class 0: noisy single-tone sinusoids
    for _ in range(n_samples // 2):
        f0 = rng.uniform(50, 450)

        x = make_tone(
            fs=fs,
            n=n,
            f0=f0,
            snr_db=snr_db,
            rng=rng
        )

        X_features.append(
            features_from_signal(x, fs)
        )
        y.append(0)

    # Class 1: noisy linear chirps
    for _ in range(n_samples // 2):
        f_start = rng.uniform(30, 200)
        f_end = rng.uniform(250, 600)

        x = make_chirp(
            fs=fs,
            n=n,
            f_start=f_start,
            f_end=f_end,
            snr_db=snr_db,
            rng=rng
        )

        X_features.append(
            features_from_signal(x, fs)
        )
        y.append(1)

    return np.vstack(X_features), np.asarray(y, dtype=int)

def mzi_block(theta, phi):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [np.exp(1j * phi) * c, s],
            [-s, np.exp(-1j * phi) * c],
        ],
        dtype=complex
    )

def mzi_mesh_matrix(
    thetas,
    phis,
    n_modes=N_PNN_MODES
):
    U = np.eye(
        n_modes,
        dtype=complex
    )

    for (i, j), theta, phi in zip(
        MZI_PAIRS,
        thetas,
        phis
    ):
        G = np.eye(
            n_modes,
            dtype=complex
        )

        G[np.ix_([i, j], [i, j])] = (
            mzi_block(theta, phi)
        )

        U = G @ U

    return U

def unpack_pnn_params(params):
    params = np.asarray(
        params,
        dtype=float
    )

    thetas = params[:N_MZIS]

    phis = params[
        N_MZIS:2 * N_MZIS
    ]

    gain = params[-2]
    bias = params[-1]

    return (
        thetas,
        phis,
        gain,
        bias
    )

def pnn_forward(
    params,
    X
):
    thetas, phis, gain, bias = (
        unpack_pnn_params(params)
    )

    U = mzi_mesh_matrix(
        thetas,
        phis
    )

    # Each row of X is one four-mode optical input.
    fields = (
        U @ np.asarray(X).T
    ).T

    intensities = (
        np.abs(fields)**2
    )

    differential_intensity = (
        np.sum(
            intensities[:, 2:4],
            axis=1
        )
        -
        np.sum(
            intensities[:, 0:2],
            axis=1
        )
    )

    scores = (
        gain
        * differential_intensity
        + bias
    )

    return (
        scores,
        intensities,
        U
    )

def pnn_predict(
    params,
    X
):
    scores, _, _ = pnn_forward(
        params,
        X
    )

    return (
        scores >= 0
    ).astype(int)

def pnn_cross_entropy(
    params,
    X,
    y,
    l2_gain=1e-4
):
    scores, _, _ = pnn_forward(
        params,
        X
    )

    y = np.asarray(
        y,
        dtype=float
    )

    # Stable binary cross-entropy:
    # log(1 + exp(score)) - y*score
    loss = np.mean(
        np.logaddexp(
            0.0,
            scores
        )
        - y * scores
    )

    # Very small penalty prevents an unnecessarily huge
    # electronic readout gain.
    gain = params[-2]

    return (
        loss
        + l2_gain * gain**2
    )

def train_pnn_multistart_quiet(
    X_train,
    y_train,
    n_restarts=MLST_PNN_RESTARTS,
    maxiter=MLST_PNN_MAXITER,
    seed=SEED
):
    rng = np.random.default_rng(
        seed
    )

    bounds = (
        [(-np.pi, np.pi)] * (2 * N_MZIS)
        + [(-20.0, 20.0)]
        + [(-10.0, 10.0)]
    )

    best_result = None

    for _ in range(
        n_restarts
    ):
        initial_point = np.concatenate(
            [
                rng.uniform(
                    -np.pi,
                    np.pi,
                    size=2 * N_MZIS
                ),
                rng.uniform(
                    -1.0,
                    1.0,
                    size=1
                ),
                np.array([0.0])
            ]
        )

        result = minimize(
            pnn_cross_entropy,
            initial_point,
            args=(X_train, y_train),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "ftol": 1e-10,
                "gtol": 1e-7,
            }
        )

        if (
            best_result is None
            or result.fun < best_result.fun
        ):
            best_result = result

    return best_result


def traceless_features(x):
    diagonal = [x[:, j]**2 - x[:, 3]**2 for j in range(3)]
    cross = [x[:, i]*x[:, j] for i, j in combinations(range(4), 2)]
    return np.column_stack(diagonal + cross)


def mean_power_and_peak_ratio(x, fs):
    power = np.abs(np.fft.rfft(x))**2 + 1e-15
    return np.array([np.mean(x**2), power.max()/power.sum()])


def evaluate(seed):
    raw, y = build_dataset(seed=seed)
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=seed, stratify=y
    )
    fresh, yf = build_dataset(n_samples=10_000, seed=100_000 + seed)
    transformer = make_pipeline(StandardScaler(), PCA(n_components=4))
    train_pc = transformer.fit_transform(raw[train])
    print(seed, 'PCA', transformer[-1].explained_variance_ratio_, flush=True)
    amplitude = MinMaxScaler(feature_range=(-0.5, 0.5), clip=True)
    xt = amplitude.fit_transform(train_pc)
    xs, xf = (
        amplitude.transform(transformer.transform(raw[test])),
        amplitude.transform(transformer.transform(fresh)),
    )
    rows = []

    def record(name, train_predictions, test_predictions, fresh_predictions, settings='', size=''):
        rows.append((seed, name, np.mean(train_predictions == y[train]),
                     np.mean(test_predictions == y[test]), np.mean(fresh_predictions == yf),
                     settings, size))

    pnn = train_pnn_multistart_quiet(xt, y[train], seed=seed)
    record('PNN', *(pnn_predict(pnn.x, x) for x in (xt, xs, xf)), size=14)
    original = LogisticRegression(max_iter=4000, random_state=seed).fit(np.pi*(2*xt+1), y[train])
    record('Original logistic', *(original.predict(np.pi*(2*x+1)) for x in (xt, xs, xf)), size=5)
    lr = LogisticRegression(max_iter=4000)
    c = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    mlp = MLPClassifier(solver='lbfgs', activation='tanh', max_iter=5000, random_state=seed)
    alpha = [0.0001, 0.001, 0.01, 0.1, 1]
    angle = MinMaxScaler(feature_range=(0, 2*np.pi), clip=True)
    models = [
        ('Logistic', angle, [], lr, {'logisticregression__C': c}, 5),
        ('Traceless quadratic', amplitude, [FunctionTransformer(traceless_features)], lr,
         {'logisticregression__C': c}, 10),
        ('General quadratic', amplitude, [PolynomialFeatures(2, include_bias=False)], lr,
         {'logisticregression__C': c}, 15),
        ('QDA', amplitude, [], QuadraticDiscriminantAnalysis(reg_param=0.001), {}, 29),
        ('MLP 4-2-1', amplitude, [], mlp, {'mlpclassifier__hidden_layer_sizes': [(2,)],
                                           'mlpclassifier__alpha': alpha}, 13),
        ('RBF-SVM', angle, [], SVC(), {'svc__C': c, 'svc__gamma': ['scale', 0.01, 0.1, 1]}, ''),
        ('Random forest', angle, [], RandomForestClassifier(n_estimators=300, random_state=seed),
         {'randomforestclassifier__max_depth': [None, 4, 8],
          'randomforestclassifier__min_samples_leaf': [1, 2, 4],
          'randomforestclassifier__max_features': ['sqrt', None]}, ''),
        ('Small MLP', angle, [], mlp, {'mlpclassifier__hidden_layer_sizes': [(8,), (16,), (16, 8)],
                                      'mlpclassifier__alpha': alpha}, ''),
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for name, scaler, features, model, grid, size in models:
        pipeline = make_pipeline(StandardScaler(), PCA(n_components=4), scaler, *features, model)
        with parallel_backend('threading'):
            search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=6, error_score='raise').fit(raw[train], y[train])
        fitted = search.best_estimator_[-1]
        if name == 'RBF-SVM':
            size = f'{len(fitted.support_)} support vectors'
        if name == 'Random forest':
            size = f'{sum(t.tree_.node_count for t in fitted.estimators_)} tree nodes'
        if name == 'Small MLP':
            size = sum(a.size for a in [*fitted.coefs_, *fitted.intercepts_])
        record(name, *(search.predict(x) for x in (raw[train], raw[test], fresh)), search.best_params_, size)
    return rows


def spectral_control():
    generator = dict(build_dataset.__globals__, features_from_signal=mean_power_and_peak_ratio)
    from types import FunctionType
    extract = FunctionType(build_dataset.__code__, generator, argdefs=build_dataset.__defaults__)
    rows = []
    for seed in range(7, 98, 10):
        x, y = extract(seed=seed)
        tr, te = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=seed)
        fpr, tpr, thresholds = roc_curve(y[tr], -x[tr, 1])
        cutoff = thresholds[np.argmax(tpr-fpr)]
        below = -x[tr, 1] < cutoff
        threshold = (cutoff + (-x[tr, 1][below]).max())/2
        for snr in (-10, -5, 0, 5, 8, 10, 15, 20):
            fresh, yf = extract(n_samples=10_000, seed=100_000+seed, snr_db=snr)
            rows.append((seed, snr, threshold, roc_auc_score(y[tr], x[tr, 0]),
                         np.mean((-x[te, 1] >= threshold) == y[te]),
                         np.mean((-fresh[:, 1] >= threshold) == yf)))
    return rows


if __name__ == '__main__':
    output = Path(__file__).parent
    with (output/'spectral_control.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'test_snr_db', 'negative_peak_ratio_threshold',
                         'train_mean_power_auc', 'original_split_accuracy', 'fresh_accuracy'])
        writer.writerows(spectral_control())
    with (output/'classical_controls.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'model', 'train_accuracy', 'original_split_accuracy',
                         'fresh_accuracy', 'hyperparameters', 'model_size'])
        for seed in range(7, 98, 10):
            writer.writerows(evaluate(seed))
            f.flush()
