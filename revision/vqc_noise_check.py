"""Illustrative two-qubit gate and readout noise sensitivity, without backend calibration."""
import csv
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import zz_feature_map
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from sklearn.model_selection import train_test_split

from quantum_kernel_check import encode
from reviewer_checks import build_dataset
from vqc_scaling_check import ParityCircuit, encode_states

rows = []
x, y = build_dataset()
tr, te = train_test_split(np.arange(len(y)), stratify=y, test_size=.25, random_state=7)
a, b = encode(x[tr], x[te])
for n, rotations, reps in [(2, ["ry", "rz"], 4), (4, ["ry"], 1)]:
    vqc = ParityCircuit(n, rotations, reps)
    feature_map = zz_feature_map(n, reps=2 if n==2 else 1)
    width = 2*np.pi
    initial_states = encode_states(feature_map, a[:, :n], width)
    weights = vqc.fit(initial_states, y[tr], 7)
    symbolic = feature_map.compose(vqc.circuit.assign_parameters(weights))
    symbolic.measure_all()
    compiled = transpile(symbolic, basis_gates=["rz", "sx", "x", "cx"],
                         coupling_map=CouplingMap.from_line(n), optimization_level=3,
                         seed_transpiler=7)
    circuits = [compiled.assign_parameters(width*row[:n]).remove_final_measurements(inplace=False)
                for row in b]
    # Parity is invariant to permutations of the active output qubits.
    for circuit in circuits:
        circuit.save_probabilities()
    simulator = AerSimulator(method="density_matrix", max_parallel_threads=1)
    odd = vqc.odd
    for strength, readout in [(0.0, 0.0), (.005, .01), (.02, .03)]:
        model = NoiseModel()
        if strength:
            model.add_all_qubit_quantum_error(depolarizing_error(strength, 2), "cx")
        results = simulator.run(circuits, noise_model=model, shots=None).result()
        probs = np.array([results.data(i)["probabilities"] for i in range(len(circuits))])
        p = probs[:, odd].sum(axis=1)
        # Independent symmetric bit flips attenuate the measured parity contrast.
        p = .5 + (p-.5)*(1-2*readout)**n
        if strength == 0:
            ref = vqc.probabilities(encode_states(feature_map, b[:, :n], width), weights)
            np.testing.assert_allclose(p, ref, atol=1e-8)
        rows.append((n, strength, readout, np.mean((p>.5)==y[te]), np.abs(p-.5).mean()))
with Path(__file__).with_name("vqc_noise_sensitivity.csv").open("w", newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["qubits", "cx_depolarizing_parameter", "readout_flip_probability",
                     "expected_parity_accuracy", "mean_absolute_probability_margin"])
    writer.writerows(rows)
