"""Sensitivity of basis gate counts to a bidirectional linear coupling map."""
from qiskit import transpile
from qiskit.circuit.library import n_local, zz_feature_map
from qiskit.transpiler import CouplingMap

for n, fmap_reps, rotations, ansatz_reps in [
    (2, 2, ["ry", "rz"], 4),
    (4, 1, ["ry"], 1),
    (4, 1, ["ry", "rz"], 2),
]:
    circuit = zz_feature_map(n, reps=fmap_reps).compose(
        n_local(n, rotations, "cx", reps=ansatz_reps, entanglement="linear")
    )
    for label, coupling, level in [
        ("unconstrained level 0", None, 0),
        ("line level 0", CouplingMap.from_line(n), 0),
        ("line level 3", CouplingMap.from_line(n), 3),
    ]:
        compiled = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"],
                             coupling_map=coupling, optimization_level=level, seed_transpiler=7)
        print(n, ansatz_reps, circuit.num_parameters-n, label,
              compiled.depth(), compiled.count_ops().get("cx", 0))
