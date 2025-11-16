# Sparse Federated Learning with Byzantine-Robust Defenses

Official implementation of Byzantine-robust federated learning with sparse aggregation weights and multiple defense mechanisms.

## ✨ Features
* **Multiple Defense Mechanisms**
  * FedLAW (Our method)
  * Krum
  * Bulyan
  * Bulyan-Bucketing
  * Trimmed Mean
  * CCLIP
  * CCLIP-Bucketing
  * RFA (Robust Federated Averaging)
  * RFA-Bucketing
  * Coordinate-wise Median
  * Huber
  * No Defense (baseline)
* **Attack Types**
  * Label Flipping
  * Backdoor
  * Inverse Gradient
  * Double Attack
  * Lie Attack
* **Supported Datasets & Models**
  * MNIST → `ThreeLayerFC`
  * CIFAR-10 → `DeeperCIFARCNN`
  * CIFAR-100 → `DeeperCIFAR100CNN`
* **Multi-GPU Training** with parallel agents
* **Experiment Tracking** via Weights & Biases

## 🛠 Requirements
* Python ≥ 3.6
* PyTorch
* CUDA-enabled GPU (optional but recommended)
* Required packages: `wandb`, `PyYAML`, `joblib`

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. **Configure Your Experiment**

Create a YAML config file (see `configs/` for examples):

```yaml
training_config:
  project_name: "your_project"
  model_name: "DeeperCIFARCNN"  # or "ThreeLayerFC", "DeeperCIFAR100CNN"
  dataset_name: "CIFAR10"       # or "MNIST", "CIFAR100"
  aggregate_type: "fedavg"      # or "sparse"
  num_clients: 200
  fraction_malicious: 0.4
  total_epochs: 400             # 200 for MNIST, 400 for CIFAR
  attack_args:
    attack_type: "boost_gradient"  # or "flip_labels", "backdoor", "lie_attack"
  defence_args:
    defence_type: "trimmed_mean"   # or "krum", "bulyan", "bulyan_bucketing", "cclip", "cclip_bucketing", "rfa", "rfa_bucketing", "coord_median", "huber", "no_defence"
```

2. **Run Training**

```bash
python script_parallel.py \
  --config path/to/your/config.yaml \
  --num-agents 4 \
  --num-gpus 2 \
  --num-run-per-config 3
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | Required |
| `--num-agents` | Number of parallel training agents | 1 |
| `--num-gpus` | Number of GPUs to use | 0 |
| `--num-run-per-config` | Runs per configuration | 1 |
| `--ignore-default-params` | Skip default parameter adjustments for Krum/Trimmed Mean | False |

## 📁 Project Structure

```
sparse-federated-learning/
├── configs/                    # Configuration files
│   ├── mnist/                  # MNIST experiment configs
│   │   ├── backdoor/           # Backdoor attack configs
│   │   ├── double_attack/      # Double attack configs
│   │   ├── flip_labels/        # Label flipping attack configs
│   │   ├── inverse_gradient/   # Inverse gradient attack configs
│   │   └── lie_attack/         # Lie attack configs
│   └── cifar/                  # CIFAR experiment configs
│       ├── backdoor/           # Backdoor attack configs
│       ├── double_attack/      # Double attack configs
│       ├── flip_labels/        # Label flipping attack configs
│       ├── inverse_gradient/   # Inverse gradient attack configs
│       └── lie_attack/         # Lie attack configs
├── attacks/                    # Attack implementations
├── client/                     # Client side implementation
├── defence/                    # Defence mechanism implementations
├── models/                     # Model architectures
├── server/
│   ├── server_base.py          # Base class for federated learning
│   ├── server_fedavg.py        # Robust aggregation methods implementation
    └── server_sparse.py        # FedLAW implementation
├── run_agents.py               # Main training script
├── requirements.txt            # Project dependencies
└── README.md     
```

## 🔬 Experiment Types

### Attack Scenarios
- **Flip Labels**: Malicious clients flip training labels
- **Backdoor**: Targeted poisoning attack
- **Inverse Gradient**: Gradient manipulation attack
- **Double Attack**: Combined attack strategies
- **Lie Attack**: Coordinated false information attack

### Defense Methods
- **Krum**: Byzantine-robust aggregation
- **Bulyan**: Enhanced Byzantine resilience
- **Bulyan-Bucketing**: Bulyan with bucketing enhancement
- **Trimmed Mean**: Statistical outlier removal
- **CCLIP**: Coordinate-wise clipping defense
- **CCLIP-Bucketing**: CCLIP with bucketing enhancement
- **RFA**: Robust Federated Averaging
- **RFA-Bucketing**: RFA with bucketing enhancement
- **Coordinate-wise Median**: Coordinate-wise median aggregation
- **Huber**: Robust loss-based defense
- **No Defense**: Baseline comparison
- **FedLAW**: Byzantine-robust federated learning with sparse aggregation weights

## 📊 Experiment Tracking

The framework uses Weights & Biases (wandb) for experiment tracking. Configure your wandb account before running:

```bash
wandb login
```

## License

MIT License - see LICENSE file for details.
