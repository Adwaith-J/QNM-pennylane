# QNM-pennylane

### Dashboard

This project includes a Streamlit dashboard for visualizing the existing classical and quantum Newton implementations.

Start the dashboard with:

```bash
streamlit run dashboard/app.py
```

Features:
- Run a new experiment using the existing implementation from `src/quantum_newton.py` and `src/classical_newton.py`
- Save the result as a structured JSON experiment file
- Load a previously saved experiment without rerunning the solver
- Inspect residual convergence, root trajectory, correction magnitude, and iteration table
- Compare classical and quantum Newton results using the actual saved output
- Display the available quantum configuration metadata without inventing values

Experiment results are saved under:

```text
experiments/results/
```

The saved file format is JSON and stores the experiment metadata, quantum configuration, classical comparison data, quantum Newton result, and per-iteration history. The dashboard loads this saved output instead of hard-coding results.
The repository includes `experiments/results/experiment_20261005_211502.json` as a sample saved experiment that can be loaded from the dashboard.

Validation:
- Existing tests continue to check the original algorithm behavior
- Person 4 validation covers the data pipeline, dashboards, experiment persistence, and saved-result loading
- The dashboard gracefully handles missing files, invalid JSON, empty histories, and unavailable optional metrics

Important caveat:
- The PennyLane implementation is a finite simulator run and should not be interpreted as asymptotic quantum speedup evidence.
- Simulator runtime is not proof of quantum advantage.
- The project retains classical baseline results for validation and comparison.
