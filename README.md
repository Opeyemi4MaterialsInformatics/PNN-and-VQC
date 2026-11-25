# Classical, photonic, and quantum classifiers for synthetic signals

Reproduction materials for **Hardware-Aware Simulation Benchmark of Classical, Photonic, and Quantum Classifiers for Synthetic Signal Classification**. The benchmark uses generated noisy tones and chirps. The photonic and superconducting-style models are simulations. No external dataset is required.

## Files

- [Original benchmark notebook](Photonic_vs_Superconducting_QNN_MLST_STRENGTHENED.ipynb) generates the synthetic training and test sets and runs the original fixed-range VQC, classical and photonic comparisons. Its saved outputs and the CSVs in [results/original](results/original) record the original protocol.
- [revision](revision) contains the revision scripts, their result tables and the updated statistical plots. The revised analysis includes fold-local model selection, matched quadratic controls, larger fresh tests, SNR stress tests, VQC encoding and noise checks, and optical component-error sensitivity.
- [figures](figures) contains the revised vector workflow and mesh schematics.
- [requirements.txt](requirements.txt) pins the supplemental analysis environment and the notebook dependencies.

## Reproduce the revised analysis

Use Python 3.13.18 and run these commands from the repository root in a virtual environment.

```sh
python -m pip install -r requirements.txt
python revision/reviewer_checks.py
python revision/quantum_kernel_check.py
python revision/vqc_scaling_check.py 2
python revision/vqc_scaling_check.py 4
python revision/hardware_checks.py
python revision/vqc_noise_check.py
python revision/referee1_checks.py
python revision/reviewer_figures.py
python revision/manuscript_schematics.py
```

Most scripts update their corresponding CSVs in `revision/`. The routing script prints gate counts and depths to the terminal. The statistical plots use the revised CSVs and the two saved original perturbation and shot-sweep tables in `results/original/`. The schematic script writes to `figures/`.

The revised classical experiments use ten seeds, 300 training signals per seed, a 100-signal holdout and a separately generated 10,000-signal test set using seed `100_000 + seed`. The exact-statevector VQC encoding check uses five seeds and a different optimizer from the original finite-shot SPSA notebook. The photonic and quantum error channels use explicitly assumed parameters. The seed-7 optical readout refit uses the training partition only.

## Run the original notebook

Launch `python -m jupyterlab` from the repository root, open the canonical notebook, and select the Python kernel from the installed environment. The VQC restart and replication cells are computationally intensive. The notebook writes its generated CSVs to its working directory. Original saved outputs were generated under the versions reported in the Methods, while the pinned environment above is the newer environment used for the revision scripts. Exact notebook reruns may vary across package versions.

## License

The project code and associated documentation are available under the [MIT License](LICENSE).
