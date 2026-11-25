"""Referee 1 diagnostics for the submitted signal generator and seed-7 optical classifier."""
import csv
from pathlib import Path
from types import FunctionType

import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from reviewer_checks import build_dataset, MZI_PAIRS, mzi_block, train_pnn_multistart_quiet

HERE = Path(__file__).parent


def spectrum_features(x, fs):
    power = abs(np.fft.rfft(x))**2 + 1e-15
    peak = np.argmax(power)
    return np.r_[power[max(0, peak-1):peak+2].sum()/power.sum(), np.log(power)]


def rotation(t):
    return np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)] ])


def physical_block(theta, phi, splitter, loss_db):
    # The two ideal 50:50 couplers have opposite orientation in this port convention.
    basis = np.diag([1j, 1])
    arm_phase = np.diag(np.exp(1j*np.array([theta, -theta])))
    arms = arm_phase @ np.diag(10**(-loss_db/20))
    port_phase = np.diag(np.exp(.5j*np.array([phi, -phi])))
    return (port_phase @ basis @ rotation(splitter[1]) @ arms
            @ rotation(splitter[0]).T @ basis.conj().T @ port_phase)


def optical_intensities(x, weights, splitter, loss_db):
    u = np.eye(4, dtype=complex)
    for k, pair in enumerate(MZI_PAIRS):
        stage = np.eye(4, dtype=complex)
        stage[np.ix_(pair, pair)] = physical_block(weights[k], weights[6+k], splitter[k], loss_db[k])
        u = stage @ u
    return abs(x @ u.T)**2


def spectral_controls():
    generator = dict(build_dataset.__globals__, features_from_signal=spectrum_features)
    extract = FunctionType(build_dataset.__code__, generator, argdefs=build_dataset.__defaults__)
    rows = []
    for seed in range(7, 98, 10):
        x, y = extract(seed=seed)
        tr, te = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=seed)
        threshold = (x[tr][y[tr] == 0, 0].min() + x[tr][y[tr] == 1, 0].max())/2
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(x[tr, 1:], y[tr])
        for snr in (-10, -5, 0, 5, 8, 10, 15, 20):
            z, target = extract(n_samples=10_000, seed=100_000+seed, snr_db=snr)
            rows.append((seed, snr, threshold, np.mean((x[te, 0] < threshold)==y[te]),
                         lr.score(x[te, 1:], y[te]), np.mean((z[:, 0] < threshold)==target),
                         lr.score(z[:, 1:], target)))
    return rows


def photonic_controls():
    rng = np.random.default_rng(7)
    x, y = build_dataset()
    tr, te = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=7)
    fresh, yf = build_dataset(n_samples=10_000, seed=100_007)
    prep = make_pipeline(StandardScaler(), PCA(n_components=4),
                         MinMaxScaler(feature_range=(-.5, .5), clip=True))
    train = prep.fit_transform(x[tr])
    test = prep.transform(fresh)
    weights = train_pnn_multistart_quiet(train, y[tr]).x
    for k in range(6):
        np.testing.assert_allclose(
            physical_block(weights[k], weights[6+k], np.full(2, np.pi/4), np.zeros(2)),
            mzi_block(weights[k], weights[6+k]), atol=1e-12)
    combined = np.vstack([train, test])
    rows = []
    for draw in range(50):
        delta = rng.normal(size=(6, 2))
        arm_loss = rng.uniform(.1, .7, size=(6, 2))
        responsivity = np.maximum(0, 1+.05*rng.normal(size=4))
        read_noise = rng.normal(0, .005, size=combined.shape)
        for name, split, loss, detector in [
            ('ideal', 0., 0., False), ('couplers only', .03, 0., False),
            ('arm loss only', 0., 1., False), ('couplers and arm loss', .03, 1., False),
            ('combined with electronics', .03, 1., True),
        ]:
            i = optical_intensities(combined, weights, np.pi/4 + split*delta, loss*arm_loss)
            if detector:
                i = np.clip(i*responsivity+read_noise, 0, .25)
                i = .25*np.round(i*255/.25)/255
            difference = i[:, 2:].sum(axis=1) - i[:, :2].sum(axis=1)
            dtrain, dtest = np.split(difference, [len(train)])
            gain, bias = weights[-2:]
            def loss_function(gb):
                scores = gb[0]*dtrain+gb[1]
                return np.mean(np.logaddexp(0,scores)-y[tr]*scores)+.0001*gb[0]**2
            adjusted = minimize(loss_function, weights[-2:], method="L-BFGS-B",
                                bounds=[(-20,20),(-10,10)]).x
            rows.append((draw+1, name, np.mean((gain*dtest+bias>=0)==yf),
                         np.mean((adjusted[0]*dtest+adjusted[1]>=0)==yf)))
    powers = np.sum(test**2, axis=1)
    print("Fresh input power quantiles and extrema", np.quantile(powers,[0,.05,.5,.95,1]))
    return rows


if __name__ == "__main__":
    for name, header, rows in [
        ("referee1_photonic.csv", ["draw", "scenario", "frozen_accuracy", "readout_refit_accuracy"], photonic_controls()),
        ("referee1_spectral.csv", ["seed", "snr_db", "three_bin_threshold", "three_bin_original", "log_spectrum_original",
                                  "three_bin_fresh", "log_spectrum_fresh"], spectral_controls()),
    ]:
        with (HERE/name).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
