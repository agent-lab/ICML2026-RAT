# Randomized Advantage Transformation (RAT)

**Computing Natural Policy Gradients via Direct Backpropagation**

*Accepted at ICML 2026*

---

## Overview

Natural policy gradients improve optimization by accounting for the geometry of distribution space, but their practical use is limited by the cost of estimating and inverting the Fisher matrix. **RAT** estimates Tikhonov-regularized natural policy gradients via direct backpropagation. By applying the Woodbury formula, we reformulate the regularized natural policy gradient as a vanilla policy gradient with a *transformed advantage*. RAT computes this transformation efficiently via randomized block Kaczmarz iterations on on-policy mini-batches, avoiding explicit Fisher construction, conjugate-gradient solvers, and architecture-specific approximations.

**Key properties:**
- No explicit Fisher matrix construction or inversion
- No conjugate-gradient inner loop
- Architecture-agnostic (MLP, CNN, ResNet)
- Supports both separate and shared actor-critic networks

---

## Installation

**Requirements:** Python 3.9+, PyTorch 2.1+, CUDA 12.1+ (recommended)

### Using Docker (recommended)

```bash
# Build the image
docker build -t rat:pytorch docker/

# Run a container
docker run --gpus all -v $(pwd):/workspace -w /workspace -it rat:pytorch /bin/bash
```

### Manual installation

```bash
# Install procgen
pip install gym3
pip install https://github.com/openai/procgen/archive/refs/tags/0.10.7.zip

# Install remaining dependencies
pip install -r requirements.txt
```

For MuJoCo environments, additional system libraries may be required. See the [MuJoCo documentation](https://mujoco.org/) for platform-specific instructions.

---

## Quick Start

RAT supports two actor-critic architectures:

### Separate actor-critic (independent policy and value networks)
```bash
python3 train_detach.py --config rat_mlp.yaml --env_name humanoid
```

### Shared actor-critic (shared backbone)
```bash
python3 train_shared.py --config rat_mlp_shared.yaml --env_name humanoid
```

### Baseline methods
```bash 
# fvp + cg for separate actor-critic
python3 train_detach.py --config fvp_mlp.yaml --env_name humanoid

# sophia (diagonal Fisher) for separate actor-critic
python3 train_detach.py --config diag_mlp.yaml --env_name humanoid

# sophia (diagonal Fisher) for shared actor-critic
python3 train_detach.py --config diag_mlp_shared.yaml --env_name humanoid

# ACTKR (kfac-based) for separate actor-critic
python3 train_detach.py --config kfac_mlp.yaml --env_name humanoid

# ACTKR (kfac-based) for shared actor-critic
python3 train_detach.py --config kfac_mlp_shared.yaml --env_name humanoid
```

### Common overrides

```bash
# Change environment
python3 train_detach.py --config rat_mlp.yaml --env_name ant

# Run on a specific GPU
python3 train_detach.py --config rat_mlp.yaml --device 0

# Override key hyperparameters
python3 train_detach.py --config rat_mlp.yaml \
    --env_name humanoid \
    --cg_damping 0.1 \
    --lr_pi 0.05 \
    --pi_epochs 4
```

---

## Supported Environments

### MuJoCo (continuous control)
`hopper`, `halfcheetah`, `walker2d`, `ant`, `humanoid`, `humanoidstandup`, `reacher`, `swimmer`, `invertedpendulum`, `inverteddoublependulum`

---

## Configuration

All hyperparameters are specified in YAML config files under `configs/`. Filenames follow the convention:

```
{algorithm}_{architecture}[_shared].yaml
```

### Algorithms

| Config prefix | Description |
|---------------|-------------|
| `rat`         | **RAT** (this work) — advantage-transformed NPG via Kaczmarz |
| `kfac`        | K-FAC natural gradient |
| `diag`        | Diagonal Fisher approximation |
| `fvp`         | Trust-region with conjugate-gradient solver |

### Architectures

| Config suffix | Description | Training script |
|--------------|-------------|-----------------|
| `mlp`        | MLP (MuJoCo, classic control), separate networks | `train_detach.py` |
| `mlp_shared` | MLP, shared backbone | `train_shared.py` |

### Key hyperparameters (`train_detach.py`)

```yaml
algo_config:
  cg_damping: 0.1          # Tikhonov regularization strength (λ)
  is_karzmarz: true        # Enable randomized Kaczmarz iterations
  norm_obj: adv            # Advantage normalization (adv | obj | ratio)
  optimizer: sgd           # Base optimizer (sgd | adam | rmsprop | kfac | ekfac)
  lr_pi: 0.05              # Policy learning rate
  lr_v: 0.001              # Value function learning rate
  pi_epochs: 4             # Policy update epochs per rollout
  pi_minibatches: 8        # Number of mini-batches per epoch
  clamp_ratio: true        # Clamp importance-sampling ratio
  max_ratio: 10.0
  min_ratio: 0.1
  grad: npg                # Gradient type (pg | npg)
  post_grad: fisher_clip   # Step clipping (fisher_clip | l2_clip | norm)
  max_grad_norm: 0.5
```

### Key hyperparameters (`train_shared.py`)

```yaml
algo_config:
  cg_damping: 0.1
  is_karzmarz: true
  norm_obj: adv
  optimizer: sgd
  lr: 0.05
  epochs: 4
  minibatches: 8
  vf_coef: 1.0             # Value loss coefficient in the joint loss
```

---

## Project Structure

```
.
├── train_detach.py        # Training — separate actor-critic
├── train_shared.py        # Training — shared actor-critic
├── configs/               # YAML hyperparameter configurations
├── utils/
│   ├── utils.py           # ActorCritic, SharedActorCritic, network builders
│   ├── runners.py         # On-policy rollout collection (GAE)
│   ├── cg.py              # Conjugate gradient solver
│   ├── popart.py          # PopArt adaptive value normalization
│   ├── running_mean_std.py
│   └── ...
├── kfac/                  # K-FAC and EK-FAC optimizer implementations
├── vec_env/               # Vectorized environment wrappers
├── tests/                 # Unit tests
└── docker/                # Docker build files
```

---

## Logging

Training logs are written to `logs/` and include:
- `progress.csv` — per-epoch training metrics
- TensorBoard event files — view with `tensorboard --logdir logs/`
- `config.yaml` — full hyperparameter snapshot for reproducibility
- `model.ckpt` — final model checkpoint

---

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{rat2026,
  title     = {Randomized Advantage Transformation ({RAT}): Computing Natural Policy Gradients via Direct Backpropagation},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning ({ICML})},
  year      = {2026},
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
