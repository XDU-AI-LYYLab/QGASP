# Zero-Cost Proxy Framework for Quantum Architecture Search

Code release accompanying the manuscript. This archive ships the minimal,
self-contained implementation of the five-stage training-free architecture
search framework:

1. **Stage I - Hardware-Aware Uniform Sampling (HA-US)**: sample candidate
   circuits under the Hardware-Compatible Virtual Gate Set (HC-VGS) and the
   physical coupling map.
2. **Stage II - Multi-Dimensional Zero-Cost Proxy Extraction**: extract 26
   proxy features (topological & causal, centrality & spatial dynamics,
   information-theoretic divergence, noise-aware, expressibility &
   trainability) from each candidate without training.
3. **Stage III - Unsupervised Latent Sampling and Training Set Construction**:
   build a PCA basis of the proxy matrix and perform stratified grid sampling
   in the latent space to select a structurally diverse subset.
4. **Stage IV - Noise-Aware Architecture Evaluation**: evaluate the selected
   circuits under the local clock-scheduled noise injection model (analytical
   Kraus channels, two-pass ALAP scheduling, state-aware decoherence
   penalties) to obtain ground-truth performance labels.
5. **Stage V - Surrogate Training and Performance Validation via Top-K Gain**:
   train a lightweight XGBoost surrogate on the labeled pairs and assess its
   ranking capability with the Top-K Gain AUC criterion.

## Repository layout

```
manuscript_code/
├── README.md
├── requirements.txt
├── demo.py                  # runs all five stages end-to-end (no data required)
├── data_prepare.py          # Stages I & III: HA-US sampling, dataset transforms,
│                            #   PCA latent sampling, stratified bin sampling, Hamiltonians
├── quantum_api_v3.py        # Stages II & IV: 26-dim proxy extraction (DAG + Ray pipeline),
│                            #   expressibility/trainability/SNIP, noise injectors, VQC ansatz
├── compute_metric.py        # Stages IV & V: ground-truth evaluation pipelines
│                            #   (classification AUC / VQE energy) and Top-K Gain AUC
└── stage5_surrogate.py      # Stage V: XGBoost surrogate training + Top-K Gain validation
```

## Environment

A CUDA-capable GPU is recommended (Stage II expressibility/trainability and
Stage IV evaluation use batched statevector simulation). The code was tested
with Python 3.13, torch 2.10, PennyLane 0.44 (+ `pennylane-lightning-gpu`),
ray 2.56, qiskit 2.3 / qiskit-aer 0.17, xgboost 3.2.

```bash
pip install -r requirements.txt
```

Notes:

- `quantum_api_v3.py` imports `qiskit` (Aer, IBM-runtime fake provider,
  machine-learning) at module level; those packages are required.
- If a patched PennyLane build was used in the original environment, activate
  it as usual (e.g., add its path to `PYTHONPATH`).

## Quick start

```bash
# Run all five stages on 30 sampled architectures (4-qubit classification config)
python demo.py --qubits 4 --n 30

# Same for the 7-qubit VQE configuration
python demo.py --qubits 7 --n 30
```

The demo covers Stages I-III on the sampled pool, runs one clock-scheduled
noisy evaluation (Stage IV) on the first circuit, and trains/validates the
XGBoost surrogate with Top-K Gain (Stage V) using synthetic labels for
illustration.

## Stage-to-module mapping

| Stage | Method (manuscript) | Implementation |
|---|---|---|
| I | HA-US sampling | `data_prepare.decisions_generator` (HC-VGS, coupling-aware) |
| II | 26-dim zero-cost proxies | `quantum_api_v3.calculate_graph_proxies_ray` and the `compute_*` proxy functions |
| III | PCA latent stratified sampling | `data_prepare.NonlinearFeatureTransformer`, `compute_latent_boundaries`, `get_bin_indices`, `sample_from_bins`, `latent_space_stratified_sampling` |
| IV | Noise-aware evaluation | `quantum_api_v3` (AdvancedNISQInjector, TraditionalNISQInjector, make_noisy_qnode, DecisionsAnsatz, analyze_circuit_timing, VQCFeatureExtractor, QiskitVQCFeatureExtractor); `compute_metric` (VQCPipeline, VQCClassifier, train_and_evaluate, VQEModel, train_vqe_model, VQEPipeline) |
| V | Surrogate training + Top-K Gain | `stage5_surrogate.train_surrogate` / `train_and_validate`; `compute_metric.compute_relative_topk_gain_auc` |

## Using Stage IV and Stage V on real labels

Stage IV produces ground-truth labels for the sampled architectures:

```python
from compute_metric import VQCPipeline

pipeline = VQCPipeline(...)            # classification target (AUC)
# or VQEPipeline(...)                  # VQE target (ground-state energy)
labels = pipeline.by_decisions(decisions_list, mode="ray", ...)
```

Stage V then trains the surrogate on the labeled pairs:

```python
from stage5_surrogate import train_and_validate

model, metrics = train_and_validate(X_proxies, labels, test_size=0.2)
# metrics: topk_gain_auc_mean, topk_gain_auc_max, spearman
```

For VQE energy labels, negate the energies before training (larger is better).

## Data and experiment notebooks

Per the release scope, the precomputed proxy/target databases and the
experiment notebooks are maintained separately. The pool databases can be
regenerated with Stage I + Stage II using the exact pool-construction
configurations embedded in `data_prepare.py` and `demo.py`:

- 4-qubit classification: `x_num=(1,50)`, `param=(1,40)`, `ent=(1,40)`
  on the 4-qubit Casablanca topology;
- 7-qubit VQE: `x_num=(0,0)`, `param=(1,100)`, `ent=(1,30)` on the 7-qubit
  topology.

The feature naming convention (raw name -> readable name + LaTeX symbol) used
in the manuscript figures is listed in Appendix A of the manuscript.
