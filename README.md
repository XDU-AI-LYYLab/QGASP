# QGASP

**Training-free Quantum Architecture Search with Graph-theoretic Proxies**

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

QGASP identifies near-optimal quantum circuits for data-reuploading models
without variational training. By combining graph-theoretic proxies, PCA-based
stratified sampling, and a local clock-scheduled noise model, it achieves
competitive performance with only 110 real-device queries.

The framework operates in five stages: candidate generation under hardware
connectivity constraints, 26-dimensional proxy extraction from circuit DAGs,
PCA latent-space sampling, noise-aware labeling via analytical Kraus channels,
and XGBoost surrogate training with Top-K Gain validation.

## Quick Start

```bash
pip install -r requirements.txt
python demo.py --qubits 4 --n 30    # 4-qubit classification
python demo.py --qubits 7 --n 30    # 7-qubit VQE
```

## Requirements

- Python 3.13+
- CUDA-capable GPU recommended
- Key dependencies: PyTorch 2.10, PennyLane 0.44, Qiskit 2.3, XGBoost 3.2, Ray 2.56

See `requirements.txt` for the full list.

## Repository Layout

```text
├── README.md
├── requirements.txt
├── demo.py                  # End-to-end pipeline (all five stages)
├── data_prepare.py          # Stages I & III: sampling, PCA, stratified binning
├── quantum_api_v3.py        # Stages II & IV: proxy extraction, noise injection, VQC
├── compute_metric.py        # Stages IV & V: evaluation pipelines, Top-K Gain AUC
└── stage5_surrogate.py      # Stage V: XGBoost surrogate training
```

## Usage

### Stage IV: Ground-truth labels

```python
from compute_metric import VQCPipeline

pipeline = VQCPipeline(...)            # classification (AUC)
# or VQEPipeline(...)                  # VQE (ground-state energy)
labels = pipeline.by_decisions(decisions_list, mode="ray")
```

### Stage V: Surrogate training

```python
from stage5_surrogate import train_and_validate

model, metrics = train_and_validate(X_proxies, labels, test_size=0.2)
# metrics: topk_gain_auc_mean, topk_gain_auc_max, spearman
```

For VQE, negate energies before training (larger is better).

## Data

Precomputed proxy/target databases and experiment notebooks are maintained
separately. Pool databases can be regenerated via Stage I + II using the
configurations in `demo.py`:

- 4-qubit classification: Casablanca topology, `x_num=(1,50)`, `param=(1,40)`, `ent=(1,40)`
- 7-qubit VQE: 7-qubit topology, `x_num=(0,0)`, `param=(1,100)`, `ent=(1,30)`

Feature naming conventions are listed in Appendix A of the manuscript.

## Citation

```bibtex
@article{li2026qgasp,
  title={QGASP: Training-free Quantum Architecture Search via Graph-theoretic Proxies and Space Decoupling},
  author={Li, Yangyang and Deng, Yu and Li, Lingling and Shang, Ronghua and Jiao, Licheng},
  journal={Information Fusion},
  year={2026},
  note={Under review}
}
```

## License

MIT
