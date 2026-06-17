# NEURA-PDE

**Neural Evidential Reconstruction of Urban Attributes via Parabolic-PDE-Governed Mobile UAV Sensing**

A remote-sensing framework that unifies four ideas from the control-theoretic and deep-learning literatures: a learned Takagi-Sugeno fuzzy neural operator for the latent urban field, an evidential Dirichlet detection head, a projection-constrained UAV policy GNN, and a Lyapunov-certified training objective. Built on top of the fuzzy-observer theory of Zhang et al. (*Nonlinear Dynamics*, 2026) and validated on the real Sentinel-2 EuroSAT dataset.

The three scripts read and write paths relative to the current working directory, so they work unchanged wherever you unpack the archive.

## Quick start on Colab (free tier, T4 GPU)

1. Open a new Colab notebook, switch runtime to **GPU (T4)**.
2. Upload `neura_pde.py` to the session.
3. Please install the following:
   ```python
   
   - torch 
   - torchvision 
   - matplotlib 
   - seaborn 
   - scikit-learn
   ```
4. Then next run:
   ```python
   %run neura_pde.py
   ```
That's it. Everything else (the PDE prior simulator, the encoder, the T-S fuzzy operator, the evidential head, the policy GNN, the Lyapunov loss, and the training loop) is self-contained in the one file.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision matplotlib seaborn scikit-learn einops
python neura_pde.py
```

On a single consumer GPU (RTX 3060 or better) one epoch takes about 90 seconds.

## Configuration

All hyperparameters live in the `Cfg` dataclass at the top of `neura_pde.py`. The ones most worth tuning are:

| field | default | notes |
|---|---|---|
| `n_fuzzy_rules` | 4 | nu in the base paper — more rules capture sharper non-linearities but cost memory |
| `T_steps` | 8 | unrolling horizon for the PDE emulator; 16 is the largest that fits on a T4 |
| `gamma` | 0.15 | diffusion coefficient of the latent parabolic field |
| `lambda_lyap` | 0.05 | weight on the monotonic-decrease regulariser for V(t) |
| `n_uav` | 4 | number of UAVs in the 2x2 subdomain partition |
| `epochs` | 12 | convergence is usually reached by epoch 10 |
| `batch_size` | 32 | reduce to 16 if out-of-memory on T4 |

## What gets produced

After training you will find in `./runs/neura_pde/`:

- `neura_pde.pt` — full model checkpoint plus config and history.
- `history.json` — per-epoch training loss breakdown and validation accuracy.
- `train_curves.png` — loss, validation accuracy, and Lyapunov regulariser over epochs.
- `confusion.png` — normalized confusion matrix on the validation split.
- `uncertainty_hist.png` — separation between epistemic uncertainty on correct versus wrong predictions (this is the main evidential sanity check).

