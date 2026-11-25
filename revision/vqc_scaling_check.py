"""Tune ZZ input range for exact-statevector VQCs with fold-local preprocessing."""
import csv
from pathlib import Path

import numpy as np
from qiskit.circuit.library import n_local, zz_feature_map
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, train_test_split

from quantum_kernel_check import encode
from reviewer_checks import build_dataset

WIDTHS = (0.5, 1., 2., np.pi, 2*np.pi)


class ParityCircuit:
    def __init__(self, n, rotations, repetitions):
        self.circuit = n_local(n, rotations, "cx", entanglement="linear", reps=repetitions)
        indices = np.arange(2**n)
        params = {p: i for i, p in enumerate(self.circuit.parameters)}
        self.gates = []
        for instruction in self.circuit.data:
            gate = instruction.operation
            qubits = [self.circuit.find_bit(q).index for q in instruction.qubits]
            first = 1 << qubits[0]
            sign = 1 - 2*((indices & first) != 0)
            if gate.name == "cx":
                self.gates.append(("cx", indices ^ np.where(sign < 0, 1 << qubits[1], 0), None, None))
            else:
                self.gates.append((gate.name, indices ^ first, sign, params[gate.params[0]]))
        self.odd = np.array([i.bit_count() % 2 for i in range(2**n)], dtype=bool)

    @staticmethod
    def apply(state, gate, angle):
        name, permutation, sign, _ = gate
        if name == "cx":
            return state[:, permutation]
        if name == "ry":
            return np.cos(angle/2)*state - np.sin(angle/2)*sign*state[:, permutation]
        return state * np.exp(-0.5j*angle*sign)

    def forward(self, inputs, weights):
        states = [inputs]
        for gate in self.gates:
            parameter = gate[-1]
            angle = weights[parameter] if parameter is not None else 0
            states.append(self.apply(states[-1], gate, angle))
        return states

    def probabilities(self, inputs, weights):
        return np.sum(abs(self.forward(inputs, weights)[-1][:, self.odd])**2, axis=1)

    def loss(self, weights, inputs, labels):
        states = self.forward(inputs, weights)
        last = states[-1]
        p = np.sum(abs(last[:, self.odd])**2, axis=1)
        correct = np.where(labels, p, 1-p)
        denominator = correct + 1e-12
        adjoint = last * self.odd * ((1-2*labels)/denominator)[:, None] / len(labels)
        gradient = np.zeros_like(weights)
        for gate, out in zip(reversed(self.gates), reversed(states[1:])):
            name, permutation, sign, parameter = gate
            angle = 0
            if parameter is not None:
                angle = weights[parameter]
                derivative = (-0.5*sign*out[:, permutation] if name == 'ry' else -0.5j*sign*out)
                gradient[parameter] += 2*np.vdot(adjoint, derivative).real
            adjoint = self.apply(adjoint, gate, -angle)
        return -np.log(denominator).mean(), gradient

    def fit(self, inputs, labels, seed):
        rng = np.random.default_rng(seed)
        candidates = [
            minimize(self.loss, rng.uniform(-np.pi, np.pi, self.circuit.num_parameters),
                     args=(inputs, labels), jac=True, method="L-BFGS-B",
                     options={"maxiter": 250})
            for _ in range(3)
        ]
        return min(candidates, key=lambda fit: fit.fun).x


def encode_states(circuit, x, width):
    return np.array([Statevector.from_instruction(circuit.assign_parameters(width*row)).data
                     for row in x])


def evaluate(seed, n, raw, labels, tr, te, fresh, yf):
    feature_map = zz_feature_map(n, reps=2 if n == 2 else 1, entanglement="full", alpha=2.)
    vqc = ParityCircuit(n, ["ry", "rz"] if n == 2 else ["ry"], 4 if n == 2 else 1)
    xt, yt = raw[tr], labels[tr]
    scores = {w: [] for w in WIDTHS}
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(xt, yt)):
        x, z = encode(xt[a], xt[b])
        for width in WIDTHS:
            u = encode_states(feature_map, x[:, :n], width)
            v = encode_states(feature_map, z[:, :n], width)
            weights = vqc.fit(u, yt[a], seed+fold)
            scores[width].append(np.mean((vqc.probabilities(v, weights) > .5) == yt[b]))
    x, z = encode(xt, np.vstack([raw[te], fresh]))
    for name, width in [('tuned', max(WIDTHS, key=lambda w: np.mean(scores[w]))),
                        ('fixed 1', 1.), ('fixed 2pi', 2*np.pi)]:
        u = encode_states(feature_map, x[:, :n], width)
        v = encode_states(feature_map, z[:, :n], width)
        weights = vqc.fit(u, yt, seed)
        pred = vqc.probabilities(v, weights) > .5
        yield (seed, n, name, width, np.mean(scores[width]),
               np.mean((vqc.probabilities(u, weights) > .5) == yt),
               np.mean(pred[:len(te)] == labels[te]), np.mean(pred[len(te):] == yf))


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    with Path(__file__).with_name(f"vqc_scaling_{n}q.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "qubits", "setting", "range", "inner_cv_accuracy", "train_accuracy",
                         "original_test_accuracy", "fresh_test_accuracy"])
        for seed in range(7, 48, 10):
            raw, labels = build_dataset(seed=seed)
            tr, te = train_test_split(np.arange(len(labels)), test_size=.25, stratify=labels, random_state=seed)
            fresh, yf = build_dataset(n_samples=10_000, seed=100_000+seed)
            writer.writerows(evaluate(seed, n, raw, labels, tr, te, fresh, yf))
            f.flush()
