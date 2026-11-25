"""Evaluate bandwidth sensitivity of exact four-qubit ZZ fidelity kernels."""
import csv
from pathlib import Path

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from reviewer_checks import build_dataset

CIRCUIT = zz_feature_map(4, reps=1, entanglement="full", alpha=2.0)
RANGES = (0.25, 0.5, 1.0, 2.0, np.pi, 2*np.pi)
REGULARIZATION = (0.1, 1, 10, 100)


def encode(train, evaluation):
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=4), MinMaxScaler(clip=True))
    return pipeline.fit_transform(train), pipeline.transform(evaluation)


def states(x, width):
    return np.array([
        Statevector.from_instruction(CIRCUIT.assign_parameters(width*row)).data
        for row in x
    ])


def fidelity(left, right):
    return np.abs(left @ right.conj().T)**2


def evaluate(seed):
    x, y = build_dataset(seed=seed)
    xt, xs, yt, ys = train_test_split(x, y, stratify=y, test_size=.25, random_state=seed)
    scores = {(width, c): [] for width in RANGES for c in REGULARIZATION}
    for tr, val in StratifiedKFold(5, shuffle=True, random_state=seed).split(xt, yt):
        a, b = encode(xt[tr], xt[val])
        for width in RANGES:
            train_states, validation_states = states(a, width), states(b, width)
            k_train = fidelity(train_states, train_states)
            k_validation = fidelity(validation_states, train_states)
            for c in REGULARIZATION:
                model = SVC(C=c, kernel="precomputed").fit(k_train, yt[tr])
                scores[width, c].append(model.score(k_validation, yt[val]))
    a, b = encode(xt, xs)
    pairs = np.triu_indices(len(yt), k=1)
    same = yt[pairs[0]] == yt[pairs[1]]
    for label, candidates in (
        ("tuned range", list(scores)),
        ("fixed range 1", [(1., c) for c in REGULARIZATION]),
        ("fixed range 2pi", [(2*np.pi, c) for c in REGULARIZATION]),
    ):
        width, c = max(candidates, key=lambda pair: np.mean(scores[pair]))
        train_states, test_states = states(a, width), states(b, width)
        k_train = fidelity(train_states, train_states)
        k_test = fidelity(test_states, train_states)
        model = SVC(C=c, kernel="precomputed").fit(k_train, yt)
        yield (seed, label, width, c, np.mean(scores[width, c]), model.score(k_train, yt),
               model.score(k_test, ys), k_train[pairs][same].mean(), k_train[pairs][~same].mean())


if __name__ == "__main__":
    with Path(__file__).with_name("quantum_kernel_controls.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "model", "range", "C", "inner_cv_accuracy", "train_accuracy",
                         "test_accuracy", "within_class_fidelity", "between_class_fidelity"])
        for seed in range(7, 48, 10):
            writer.writerows(evaluate(seed))
            f.flush()
