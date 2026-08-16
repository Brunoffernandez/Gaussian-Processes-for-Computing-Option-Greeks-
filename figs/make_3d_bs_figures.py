"""Alternative 3-D visualisation of the Section 4.2 Black-Scholes autograd Greeks.

We retrain the same 2-D GPyTorch model as in `src/section_4_2_bs_autograd.py`
and render the resulting Price / Delta / Gamma / Vega surfaces as 3-D plots
with the analytical BS truth overlaid as a black wireframe, plus a 1-D slice
at sigma = 0.30. The paper's Figure 10 uses the flat 2-D heatmap style
(produced by the section script itself); this file offers a 3-D view for
presentations or extra intuition. The model is retrained here rather than
imported because the section script executes at module load and would fire
its own figure writes as a side effect.
"""

import os
import numpy as np
import scipy.stats as st
import torch
import gpytorch
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Black-Scholes closed-form helpers
def bs_call(S, K, r, T, sigma):
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return S * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)


def bs_delta(S, K, r, T, sigma):
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  return st.norm.cdf(d1)


def bs_gamma(S, K, r, T, sigma):
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  return st.norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, r, T, sigma):
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  return S * np.sqrt(T) * st.norm.pdf(d1)


# Parameters and training grid (same as section_4_2_bs_autograd.py)
K_strike, r, T = 100.0, 0.02, 1.0
S_lb, S_ub   = 60.0, 140.0
sig_lb, sig_ub = 0.10, 0.50

n_grid = 15
S_grid   = np.linspace(S_lb, S_ub, n_grid)
sig_grid = np.linspace(sig_lb, sig_ub, n_grid)
S_mesh, sig_mesh = np.meshgrid(S_grid, sig_grid)
X_train_np = np.column_stack([S_mesh.ravel(), sig_mesh.ravel()])
y_train_np = bs_call(X_train_np[:, 0], K_strike, r, T, X_train_np[:, 1])


def rescale(X):
  Xs = X.copy()
  Xs[:, 0] = (X[:, 0] - S_lb) / (S_ub - S_lb)
  Xs[:, 1] = (X[:, 1] - sig_lb) / (sig_ub - sig_lb)
  return Xs


X_train = torch.tensor(rescale(X_train_np))
y_train = torch.tensor(y_train_np)


class ExactGP2D(gpytorch.models.ExactGP):
  def __init__(self, X, y, likelihood):
    super().__init__(X, y, likelihood)
    self.mean_module  = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))

  def forward(self, x):
    return gpytorch.distributions.MultivariateNormal(
        self.mean_module(x), self.covar_module(x))


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGP2D(X_train, y_train, likelihood)
model.train(); likelihood.train()
opt = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
for _ in range(200):
  opt.zero_grad()
  loss = -mll(model(X_train), y_train)
  loss.backward(); opt.step()


# Test grid and Greeks by autograd
n_test = 50
S_test_grid   = np.linspace(70.0, 130.0, n_test)
sig_test_grid = np.linspace(0.15, 0.45, n_test)
S_te, sig_te  = np.meshgrid(S_test_grid, sig_test_grid)
X_test_np = np.column_stack([S_te.ravel(), sig_te.ravel()])
X_test = torch.tensor(rescale(X_test_np), requires_grad=True)

model.eval(); likelihood.eval()
with gpytorch.settings.fast_pred_var():
  pred = likelihood(model(X_test))
mu = pred.mean

g_all    = torch.autograd.grad(mu.sum(), X_test, create_graph=True)[0]
delta_gp = g_all[:, 0].detach().numpy() / (S_ub - S_lb)
vega_gp  = g_all[:, 1].detach().numpy() / (sig_ub - sig_lb)
gamma_rows = torch.autograd.grad(g_all[:, 0].sum(), X_test, retain_graph=False)[0]
gamma_gp = gamma_rows[:, 0].detach().numpy() / (S_ub - S_lb) ** 2
mu_np    = mu.detach().numpy()

# ground truth on the same grid
price_true = bs_call (X_test_np[:, 0], K_strike, r, T, X_test_np[:, 1])
delta_true = bs_delta(X_test_np[:, 0], K_strike, r, T, X_test_np[:, 1])
gamma_true = bs_gamma(X_test_np[:, 0], K_strike, r, T, X_test_np[:, 1])
vega_true  = bs_vega (X_test_np[:, 0], K_strike, r, T, X_test_np[:, 1])


def rs(arr):
  return arr.reshape(n_test, n_test)


# 3-D surfaces
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("white")

specs = [
    ("Price",  mu_np,    price_true, cm.viridis,  r"$C(S,\sigma)$"),
    ("Delta",  delta_gp, delta_true, cm.plasma,   r"$\Delta = \partial C/\partial S$"),
    ("Gamma",  gamma_gp, gamma_true, cm.inferno,  r"$\Gamma = \partial^2 C/\partial S^2$"),
    ("Vega",   vega_gp,  vega_true,  cm.magma,    r"$\nu = \partial C/\partial\sigma$"),
]

for col, (name, gp_arr, true_arr, cmap, label) in enumerate(specs):
  G  = rs(gp_arr); Tr = rs(true_arr); Er = np.abs(G - Tr)

  ax = fig.add_subplot(2, 4, col + 1, projection="3d")
  surf = ax.plot_surface(S_te, sig_te, G, cmap=cmap, linewidth=0, antialiased=True, alpha=0.92)
  ax.plot_wireframe(S_te, sig_te, Tr, color="black", linewidth=0.35, alpha=0.55)
  ax.set_xlabel("S", fontsize=9, labelpad=4)
  ax.set_ylabel(r"$\sigma$", fontsize=9, labelpad=4)
  ax.set_zlabel(label, fontsize=9, labelpad=4)
  ax.set_title(f"GP {name}", fontsize=11, fontweight="bold", pad=8)
  ax.tick_params(labelsize=7)
  fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, aspect=12)

  ax2 = fig.add_subplot(2, 4, col + 5, projection="3d")
  surf2 = ax2.plot_surface(S_te, sig_te, Er, cmap="RdYlGn_r", linewidth=0, antialiased=True, alpha=0.92)
  ax2.set_xlabel("S", fontsize=9, labelpad=4)
  ax2.set_ylabel(r"$\sigma$", fontsize=9, labelpad=4)
  ax2.set_zlabel("|GP - BS|", fontsize=9, labelpad=4)
  ax2.set_title(f"|GP - BS|  max={Er.max():.2e}", fontsize=10, color="darkred", pad=8)
  ax2.tick_params(labelsize=7)
  fig.colorbar(surf2, ax=ax2, shrink=0.45, pad=0.08, aspect=12)

fig.suptitle(
    "GP posterior-mean Greeks on the 2-D surface $(S,\\sigma)$\n"
    "Top row: GP surface (coloured) with BS truth (black wireframe) - "
    "Bottom row: absolute error",
    fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_2d_greeks_3d.png"), dpi=140, bbox_inches="tight")
plt.close()
print("Saved fig4_2d_greeks_3d.png")


# 1-D slice at sigma = 0.30
slice_idx = np.argmin(np.abs(sig_test_grid - 0.30))
sl = lambda arr: rs(arr)[slice_idx, :]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.patch.set_facecolor("white")
titles = [r"Price $C$",
          r"Delta $\partial C/\partial S$",
          r"Gamma $\partial^2 C/\partial S^2$",
          r"Vega $\partial C/\partial\sigma$"]
gp_arrs   = [mu_np, delta_gp, gamma_gp, vega_gp]
true_arrs = [price_true, delta_true, gamma_true, vega_true]

for ax, title, gp_a, tr_a in zip(axes, titles, gp_arrs, true_arrs):
  ax.plot(S_test_grid, sl(tr_a), "k-",  lw=2.2, label="BS exact")
  ax.plot(S_test_grid, sl(gp_a), "r--", lw=1.8, label="GP (autograd)")
  ax.set_xlabel("S", fontsize=10)
  ax.set_title(title, fontsize=11)
  ax.legend(fontsize=9)
  ax.grid(True, alpha=0.4)

fig.suptitle(r"1-D slice at $\sigma = 0.30$  -  GP vs Black-Scholes", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_2d_greeks_slice.png"), dpi=140, bbox_inches="tight")
plt.close()
print("Saved fig5_2d_greeks_slice.png")
