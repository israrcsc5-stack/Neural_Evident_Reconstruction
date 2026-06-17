"""
===========================================================================
 NEURA-PDE  —  Neural Evidential Reconstruction of Urban Attributes via
               Parabolic-PDE-Governed Mobile UAV Sensing
===========================================================================
Tested on Colab T4 (16 GB). Peak memory ~9 GB with batch_size=32.
===========================================================================
"""
#for full code you can request at israrcsc5@gmail.com, and it will be made public once our paper got published.

# ===========================================================================
# CELL 1  —  install
# ===========================================================================
# In Colab, run this cell once:
# !pip install -q torch torchvision matplotlib seaborn scikit-learn einops

# ===========================================================================
# CELL 2  —  imports + config
# ===========================================================================
import os, math, json, random
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

@dataclass
class Cfg:
    img_size: int = 64
    n_classes: int = 10
    channels_rgb: int = 3
    channels_pde: int = 3       # 3 learned PDE prior channels (thermal / traffic / acoustic)
    feat_dim: int = 256
    n_fuzzy_rules: int = 4       # nu in the base paper
    n_uav: int = 4               # 2x2 partition like the paper's simulation
    gnn_nodes: int = 8           # policy GNN nodes
    T_steps: int = 8             # PDE unrolling horizon
    gamma: float = 0.15          # diffusion coefficient
    lambda_field: float = 0.30
    lambda_evid: float  = 0.10
    lambda_lyap: float  = 0.05
    lambda_policy: float = 0.05
    epochs: int = 12
    batch_size: int = 32
    lr: float = 3e-4
    wd: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_root: str = "./data"
    out_dir: str = "./runs/neura_pde"

CFG = Cfg()
os.makedirs(CFG.out_dir, exist_ok=True)
print(f"Device: {CFG.device}   |   img: {CFG.img_size}   |   classes: {CFG.n_classes}")

# ===========================================================================
# CELL 3  —  real EuroSAT (Sentinel-2)
# ===========================================================================
class EuroSATWithPDE(torch.utils.data.Dataset):
    """Wrap EuroSAT to additionally emit a synthetic but physics-grounded PDE
    prior derived from each image. The PDE prior simulates a transient urban
    thermal / traffic field seeded by the image intensity — this is the
    'latent field' the observer tries to reconstruct."""
    def __init__(self, root, split="train", transform=None, T=CFG.T_steps,
                 gamma=CFG.gamma):
        super().__init__()
        self.base = torchvision.datasets.EuroSAT(root=root, download=True,
                                                  transform=None)
        N = len(self.base)
        gen = torch.Generator().manual_seed(0)
        perm = torch.randperm(N, generator=gen).tolist()
        cut = int(0.8 * N)
        self.indices = perm[:cut] if split == "train" else perm[cut:]
        self.transform = transform
        self.T = T
        self.gamma = gamma

    def __len__(self): return len(self.indices)

    def _pde_prior(self, img_tensor):
        """Simulate T parabolic PDE time steps seeded by the RGB intensity.
        Returns (T, 3, H, W) — 3 physics channels: thermal, traffic, acoustic."""
        C, H, W = img_tensor.shape
        u = img_tensor.mean(0, keepdim=True)  # (1, H, W) initial seed
        thermal = u.clone() * 1.2 - 0.1
        traffic = u.clone() * 0.8
        acoustic = u.clone() * 0.5
        frames = torch.zeros(self.T, 3, H, W)
        dt = 0.05
        for t in range(self.T):
            for fld, alpha, nl in [(thermal, self.gamma, 0.03),
                                   (traffic, self.gamma*0.8, 0.05),
                                   (acoustic, self.gamma*1.2, 0.02)]:
                # 5-point laplacian
                lap = (F.pad(fld, (1,1,1,1), mode="replicate"))
                lap = (lap[:, :-2, 1:-1] + lap[:, 2:, 1:-1] +
                       lap[:, 1:-1, :-2] + lap[:, 1:-1, 2:] - 4 * fld)
                fld.add_(dt * (alpha * lap + nl * torch.sin(fld)))
            frames[t, 0] = thermal.squeeze(0)
            frames[t, 1] = traffic.squeeze(0)
            frames[t, 2] = acoustic.squeeze(0)
        return frames

    def __getitem__(self, i):
        img, lbl = self.base[self.indices[i]]
        if self.transform: img = self.transform(img)
        pde = self._pde_prior(img)
        return img, pde, lbl


def build_loaders(cfg):
    tf = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tr_ds = EuroSATWithPDE(cfg.data_root, "train", transform=tf)
    va_ds = EuroSATWithPDE(cfg.data_root, "val",   transform=tf)
    tr = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=2, pin_memory=True)
    va = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=2, pin_memory=True)
    return tr, va

# ===========================================================================
# CELL 4  —  symbolic PDE reference simulator (for ground truth)
# ===========================================================================
def simulate_true_field(img, T=CFG.T_steps, gamma=CFG.gamma):
    """Ground-truth PDE evolution used to supervise the T-S fuzzy operator.
    Returns (B, T, 3, H, W)."""
    B, C, H, W = img.shape
    u = img.mean(1, keepdim=True)
    t_ch = [u.clone()*1.2 - 0.1, u.clone()*0.8, u.clone()*0.5]
    out = torch.zeros(B, T, 3, H, W, device=img.device)
    dt = 0.05
    for t in range(T):
        for k, fld in enumerate(t_ch):
            pad = F.pad(fld, (1,1,1,1), mode="replicate")
            lap = (pad[:, :, :-2, 1:-1] + pad[:, :, 2:, 1:-1] +
                   pad[:, :, 1:-1, :-2] + pad[:, :, 1:-1, 2:] - 4*fld)
            nl = [0.03, 0.05, 0.02][k]
            alpha = gamma * [1.0, 0.8, 1.2][k]
            t_ch[k] = fld + dt * (alpha * lap + nl * torch.sin(fld))
            out[:, t, k] = t_ch[k].squeeze(1)
    return out

# ===========================================================================
# CELL 5  —  T-S fuzzy neural PDE operator
# ===========================================================================
class TSFuzzyOperator(nn.Module):
    """Learns a bank of nu linear operators Theta_p plus soft membership
    h_p(vartheta) such that the output is  sum_p  h_p * (Theta_p xi)  —
    the neural analogue of the T-S fuzzy local linear rule decomposition."""
    def __init__(self, in_ch=3, hidden=32, nu=CFG.n_fuzzy_rules):
        super().__init__()
        self.nu = nu
        self.rules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hidden, 3, padding=1),
                nn.GroupNorm(4, hidden), nn.GELU(),
                nn.Conv2d(hidden, in_ch, 3, padding=1),
            ) for _ in range(nu)
        ])
        # membership net: maps local field magnitude to softmax weights
        self.memb = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GroupNorm(4, hidden), nn.GELU(),
            nn.Conv2d(hidden, nu, 1)
        )

    def forward(self, xi):
        # xi: (B, 3, H, W)  current field snapshot
        w = F.softmax(self.memb(xi), dim=1)  # (B, nu, H, W)
        out = 0
        for p, rule in enumerate(self.rules):
            out = out + w[:, p:p+1] * rule(xi)
        return out, w  # updated field delta, memberships


class LatentFieldEmulator(nn.Module):
    """Unrolls  xi_{t+1} = xi_t + dt*(gamma*Laplacian xi + TSOp(xi))."""
    def __init__(self, gamma=CFG.gamma, dt=0.05, T=CFG.T_steps, nu=CFG.n_fuzzy_rules):
        super().__init__()
        self.gamma = gamma; self.dt = dt; self.T = T
        self.op = TSFuzzyOperator(nu=nu)
        # known 5-pt Laplacian kernel
        k = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]).view(1,1,3,3)
        self.register_buffer("lap_k", k)

    def laplacian(self, xi):
        B, C, H, W = xi.shape
        x = xi.reshape(B*C, 1, H, W)
        x = F.conv2d(F.pad(x, (1,1,1,1), mode="replicate"), self.lap_k)
        return x.reshape(B, C, H, W)

    def forward(self, xi0):
        xi = xi0
        frames = []
        membs = []
        for t in range(self.T):
            lap = self.laplacian(xi)
            delta, w = self.op(xi)
            xi = xi + self.dt * (self.gamma * lap + delta)
            frames.append(xi); membs.append(w)
        return torch.stack(frames, 1), torch.stack(membs, 1)

# ===========================================================================
# CELL 6  —  multi-modal encoder
# ===========================================================================
class MultiModalEncoder(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.rgb_stem = self._stem(cfg.channels_rgb, 64)
        self.pde_stem = self._stem(cfg.channels_pde, 64)
        self.cross = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.trunk = nn.Sequential(
            self._block(128, 128), self._block(128, 192), self._block(192, cfg.feat_dim),
        )

    def _stem(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c//2, 3, 2, 1), nn.GroupNorm(8, out_c//2), nn.GELU(),
            nn.Conv2d(out_c//2, out_c, 3, 2, 1), nn.GroupNorm(8, out_c), nn.GELU(),
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 2, 1), nn.GroupNorm(8, out_c), nn.GELU(),
        )

    def forward(self, rgb, pde_prior):
        """rgb: (B,3,H,W);  pde_prior: (B,3,H,W)  (last frame of the PDE prior)"""
        r = self.rgb_stem(rgb)    # (B,64,H/4,W/4)
        p = self.pde_stem(pde_prior)
        B, Cc, Hc, Wc = r.shape
        r_seq = r.flatten(2).transpose(1, 2)  # (B, N, 64)
        p_seq = p.flatten(2).transpose(1, 2)
        # pairwise cross-attn: r attends to p
        out_r, _ = self.cross(r_seq, p_seq, p_seq)
        out_p, _ = self.cross(p_seq, r_seq, r_seq)
        fused = torch.cat([out_r, out_p], dim=-1).transpose(1, 2).view(B, 128, Hc, Wc)
        F_ = self.trunk(fused)    # (B, feat_dim, H/32, W/32)
        return F_

# ===========================================================================
# CELL 7  —  Evidential detection head (Dirichlet)
# ===========================================================================
class EvidentialHead(nn.Module):
    def __init__(self, feat_dim=CFG.feat_dim, n_classes=CFG.n_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, F_):
        z = self.pool(F_).flatten(1)
        logits = self.fc(z)
        evidence = F.softplus(logits)         # non-negative
        alpha = evidence + 1.0                # Dirichlet concentration
        S = alpha.sum(1, keepdim=True)
        p = alpha / S                          # expected class probabilities
        u = alpha.shape[1] / S.squeeze(1)      # vacuity / epistemic uncertainty
        return dict(alpha=alpha, prob=p, uncertainty=u, evidence=evidence)

# ===========================================================================
# CELL 8  —  Projection-constrained UAV policy GNN
# ===========================================================================
class PolicyGNN(nn.Module):
    """GNN over the 8 policy nodes (4 UAVs x 2 logical slots for position +
    sensing weight). Outputs per-UAV velocity (vx, vy) that the projection
    operator will clip to the UAV's subdomain."""
    def __init__(self, n_nodes=CFG.gnn_nodes, feat_dim=CFG.feat_dim, hidden=64):
        super().__init__()
        self.n_nodes = n_nodes
        self.proj = nn.Linear(feat_dim, hidden)
        self.msg = nn.Sequential(
            nn.Linear(2*hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.upd = nn.GRUCell(hidden, hidden)
        self.vel_head = nn.Linear(hidden, 2)   # (vx, vy) per node
        self.sense_head = nn.Linear(hidden, 1) # sensing weight in [0,1]

    def forward(self, F_, rounds=3):
        # F_: (B, feat, h, w) — we pool and broadcast to 8 nodes as init state
        B = F_.size(0)
        g = F.adaptive_avg_pool2d(F_, 1).flatten(1)        # (B, feat)
        h = self.proj(g).unsqueeze(1).repeat(1, self.n_nodes, 1)  # (B, N, hid)
        # fully-connected graph (small N=8)
        for _ in range(rounds):
            # aggregate messages from all other nodes
            m_src = h.unsqueeze(2).expand(-1, -1, self.n_nodes, -1)
            m_dst = h.unsqueeze(1).expand(-1, self.n_nodes, -1, -1)
            mij = self.msg(torch.cat([m_src, m_dst], -1))   # (B, N, N, hid)
            m = mij.sum(2) / (self.n_nodes - 1)
            h = self.upd(m.reshape(-1, h.size(-1)), h.reshape(-1, h.size(-1))).reshape(B, self.n_nodes, -1)
        v = self.vel_head(h)                      # (B, N, 2)
        w = torch.sigmoid(self.sense_head(h)).squeeze(-1)   # (B, N)
        return v, w


def projection_op(pos, vel, lo, hi, delta=0.02):
    """Differentiable version of base-paper eq. (13-14). pos, vel: (B,).
    Keeps the UAV inside [lo, hi] via a smoothly damped velocity."""
    left  = (pos < lo + delta)
    right = (pos > hi - delta)
    mid   = (~left) & (~right)
    out = torch.zeros_like(vel)
    out = torch.where(mid, vel, out)
    # near left boundary: damp if vel<0
    out = torch.where(left & (vel < 0),
                      (1 + (pos - (lo + delta)) / delta) * vel, out)
    out = torch.where(left & (vel >= 0), vel, out)
    # near right boundary: damp if vel>0
    out = torch.where(right & (vel > 0),
                      (1 + ((hi - delta) - pos) / delta) * vel, out)
    out = torch.where(right & (vel <= 0), vel, out)
    return out

# ===========================================================================
# CELL 9  —  end-to-end NEURA-PDE
# ===========================================================================
class NeuraPDE(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.encoder = MultiModalEncoder(cfg)
        self.emulator = LatentFieldEmulator(gamma=cfg.gamma, T=cfg.T_steps, nu=cfg.n_fuzzy_rules)
        self.evid = EvidentialHead(cfg.feat_dim, cfg.n_classes)
        self.policy = PolicyGNN(cfg.gnn_nodes, cfg.feat_dim)

    def forward(self, rgb, pde_prior, uav_pos=None):
        # use the LAST frame of the simulated PDE prior as the stable field snapshot
        xi0 = pde_prior[:, 0]                     # initial (B,3,H,W)
        xi_true_hist = pde_prior                   # (B,T,3,H,W) — acts as GT
        # emulator forward
        xi_hat_hist, membs = self.emulator(xi0)    # (B,T,3,H,W)
        # encoder uses RGB + reconstructed final field
        F_ = self.encoder(rgb, xi_hat_hist[:, -1])
        # detection (evidential)
        det = self.evid(F_)
        # policy
        vel, sense_w = self.policy(F_)
        out = dict(F=F_, xi_hat=xi_hat_hist, xi_true=xi_true_hist,
                   membs=membs, det=det, vel=vel, sense_w=sense_w)
        return out

# ===========================================================================
# CELL 10  —  composite loss including Lyapunov regularizer
# ===========================================================================
def evidential_nll(alpha, y, n_classes):
    """Sensoy et al. evidential cross-entropy + KL regulariser."""
    y_oh = F.one_hot(y, n_classes).float()
    S = alpha.sum(1, keepdim=True)
    p = alpha / S
    # expected cross-entropy
    A = (y_oh * (torch.digamma(S) - torch.digamma(alpha))).sum(1).mean()
    # KL to uniform Dirichlet on mis-evidence
    alpha_tilde = y_oh + (1.0 - y_oh) * alpha
    kl = _kl_dirichlet_uniform(alpha_tilde, n_classes)
    return A + 0.01 * kl


def _kl_dirichlet_uniform(alpha, K):
    S = alpha.sum(1, keepdim=True)
    t1 = torch.lgamma(S).squeeze(1) - torch.lgamma(alpha).sum(1)
    t2 = -math.lgamma(K)
    t3 = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))).sum(1)
    return (t1 + t2 + t3).mean()


def lyapunov_loss(xi_true, xi_hat):
    """Discretized Lyapunov candidate:
       V(t) = 0.5*||e||^2 + 0.5*||grad e||^2  ; penalize non-monotonic V.
       Encourages V(t+1) <= V(t) (matches base paper's dV/dt <= 0)."""
    e = xi_true - xi_hat                     # (B, T, 3, H, W)
    v1 = 0.5 * (e ** 2).sum(dim=(-1, -2, -3))  # (B, T)
    # grad via finite differences
    ge_x = e[..., 1:, :] - e[..., :-1, :]
    ge_y = e[..., :, 1:] - e[..., :, :-1]
    v3 = 0.5 * ((ge_x ** 2).sum(dim=(-1, -2, -3)) + (ge_y ** 2).sum(dim=(-1, -2, -3)))
    V = v1 + 0.5 * v3                         # (B, T)
    # dV <= 0 penalty
    dV = V[:, 1:] - V[:, :-1]
    pen = F.relu(dV).mean()                   # only positive dV is bad
    # also minimize V at the last step
    return pen + 0.1 * V[:, -1].mean()


def compute_loss(out, labels, cfg=CFG):
    det_loss = evidential_nll(out["det"]["alpha"], labels, cfg.n_classes)
    field_loss = F.mse_loss(out["xi_hat"], out["xi_true"])
    lyap_loss = lyapunov_loss(out["xi_true"], out["xi_hat"])
    # policy regularizers
    vel = out["vel"]
    policy_loss = (vel ** 2).mean() * 0.1     # action smoothness
    # simple evidence regularization (avoid exploding evidence)
    alpha = out["det"]["alpha"]
    evid_loss = (alpha.mean() - 1.0).abs() * 0.01
    total = (det_loss
             + cfg.lambda_field * field_loss
             + cfg.lambda_lyap * lyap_loss
             + cfg.lambda_policy * policy_loss
             + cfg.lambda_evid * evid_loss)
    return dict(total=total, det=det_loss, field=field_loss,
                lyap=lyap_loss, policy=policy_loss, evid=evid_loss)

# ===========================================================================
# CELL 11  —  training loop
# ===========================================================================
@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    all_p, all_y, all_u = [], [], []
    tot = 0; correct = 0
    for rgb, pde, y in loader:
        rgb = rgb.to(cfg.device); pde = pde.to(cfg.device); y = y.to(cfg.device)
        out = model(rgb, pde)
        p = out["det"]["prob"]
        pred = p.argmax(1)
        correct += (pred == y).sum().item()
        tot += y.size(0)
        all_p.append(p.cpu()); all_y.append(y.cpu()); all_u.append(out["det"]["uncertainty"].cpu())
    return dict(acc=correct/tot,
                probs=torch.cat(all_p), labels=torch.cat(all_y),
                uncertainty=torch.cat(all_u))


def train(cfg=CFG):
    tr, va = build_loaders(cfg)
    model = NeuraPDE(cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    history = dict(train_loss=[], val_acc=[], lyap=[], field=[], det=[])
    for ep in range(cfg.epochs):
        model.train()
        agg = {k: 0.0 for k in ["total","det","field","lyap","policy","evid"]}
        nb = 0
        for rgb, pde, y in tr:
            rgb = rgb.to(cfg.device); pde = pde.to(cfg.device); y = y.to(cfg.device)
            out = model(rgb, pde)
            losses = compute_loss(out, y, cfg)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in agg: agg[k] += losses[k].item()
            nb += 1
        sched.step()
        for k in agg: agg[k] /= max(1, nb)
        ev = evaluate(model, va, cfg)
        history["train_loss"].append(agg["total"])
        history["val_acc"].append(ev["acc"])
        history["lyap"].append(agg["lyap"])
        history["field"].append(agg["field"])
        history["det"].append(agg["det"])
        print(f"ep {ep+1:02d}/{cfg.epochs}  loss={agg['total']:.3f}  "
              f"det={agg['det']:.3f}  field={agg['field']:.4f}  "
              f"lyap={agg['lyap']:.4f}  val_acc={ev['acc']:.4f}")
    # save
    ckpt = os.path.join(cfg.out_dir, "neura_pde.pt")
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "history": history}, ckpt)
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved: {ckpt}")
    return model, history, ev

# ===========================================================================
# CELL 12  —  evaluation plots (condensed) for full code you can request at israrcsc5@gmail.com, and it will be made public once our paper got published.
# ===========================================================================
def evaluation_plots(history, ev, out_dir=CFG.out_dir, classes=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report
    os.makedirs(out_dir, exist_ok=True)
    # training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], lw=2); axes[0].set_title("train loss"); axes[0].grid(alpha=0.3)
    axes[1].plot(history["val_acc"], lw=2, color="#2a8f4e"); axes[1].set_title("val acc"); axes[1].grid(alpha=0.3)
    axes[2].plot(history["lyap"], lw=2, color="#c24a3c"); axes[2].set_title(r"$\mathcal{L}_{Lyap}$"); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "train_curves.png"), dpi=150); plt.close()
    # confusion matrix
    y_true = ev["labels"].numpy(); y_pred = ev["probs"].argmax(1).numpy()
    C = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    color="white" if C[i,j]>0.5 else "black", fontsize=8)
    if classes is None:
        classes = [f"c{i}" for i in range(C.shape[0])]
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_title("Confusion matrix (EuroSAT val)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "confusion.png"), dpi=150); plt.close()
    # uncertainty histogram
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    correct_mask = (y_true == y_pred)
    u = ev["uncertainty"].numpy()
    ax.hist(u[correct_mask], bins=30, alpha=0.6, label="correct", color="#2a8f4e")
    ax.hist(u[~correct_mask], bins=30, alpha=0.6, label="wrong", color="#c24a3c")
    ax.set_xlabel("epistemic uncertainty  u"); ax.set_ylabel("count")
    ax.set_title("Uncertainty separation  (wrong predictions have higher u)")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "uncertainty_hist.png"), dpi=150); plt.close()
    print("Plots saved to", out_dir)


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    model, history, ev = train(CFG)
    EUROSAT_CLASSES = [
        "AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial",
        "Pasture","PermanentCrop","Residential","River","SeaLake",
    ]
    evaluation_plots(history, ev, classes=EUROSAT_CLASSES)
    print("Done.")
