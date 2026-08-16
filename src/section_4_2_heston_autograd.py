"""Section 4.2.4: Heston pricing and Greeks via GPyTorch autograd, with credible bands.

We reproduce the book's Example-5 (pricing and Greeking with single GPs on a
Heston surface), obtaining the Greeks by autograd on the GP posterior mean
rather than by differentiating the kernel by hand. Training prices come from
the COS method (Fang & Oosterlee 2008), the same pricer the book uses; the
parameters follow the book's Table 3.1. We fit the GP with n = 50, 100, 150
training points (book Fig 3.6), fit a 2-D GP on (S, volatility), and extract
Delta and Vega by autograd. Each Greek is drawn with a 95% credible band
propagated from the posterior covariance through a central finite-difference
operator - the closed-form GP hedging-risk margin.
"""

import os
import sys
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

# make the local Heston COS pricer importable when running this file directly
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from pricers.heston_cos import bs_call, heston_call_cos, heston_vec  # noqa: E402

torch.set_default_dtype(torch.float64)
np.random.seed(0)
torch.manual_seed(0)
plt.rcParams.update({"axes.grid": True})

OUTPUT_DIR = os.path.join(os.path.dirname(_HERE), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# book Table 3.1 parameters
kappa, theta, volvol, r = 0.1, 0.15, 0.1, 0.002
K, T, rho = 100.0, 2.0, -0.9
V0 = 0.1                                   # initial variance (book Table 3.2)

# sanity check: vol-of-vol -> 0 with v0 = theta must collapse to BS(sqrt(theta))
_h = heston_call_cos(100, 100, r, T, theta, kappa, theta, 1e-4, 0.0)
_b = bs_call(100, 100, r, T, np.sqrt(theta))
print(f"COS BS-limit check: Heston={_h:.6f}  BS={_b:.6f}  diff={abs(_h-_b):.2e}\n")


class ExactGP1D(gpytorch.models.ExactGP):
  '''Our 1-D exact GP on price-vs-S: constant mean and scaled Matern-5/2 kernel.'''
  def __init__(self, X, y, likelihood):
    super().__init__(X, y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5))

  def forward(self, x):
    return gpytorch.distributions.MultivariateNormal(
        self.mean_module(x), self.covar_module(x))


S_lb, S_ub = 60.0, 200.0          # underlying range (matches book Fig 3.6 axis)


def rescale(S):
  '''We rescale S to [0, 1] so the GP length-scale is well conditioned.'''
  return (S - S_lb) / (S_ub - S_lb)


def fit_gp(n_train, seed=0):
  '''
    We draw n_train random spots, price them with COS, and fit the 1-D GP by
    maximising the marginal log likelihood with 200 Adam iterations.
  '''
  rng = np.random.default_rng(seed)
  S_train = np.sort(rng.uniform(S_lb, S_ub, n_train))
  y_train = heston_vec(S_train, K, r, T, V0, kappa, theta, volvol, rho)
  X = torch.tensor(rescale(S_train)).unsqueeze(-1)
  y = torch.tensor(y_train)
  likelihood = gpytorch.likelihoods.GaussianLikelihood()
  model = ExactGP1D(X, y, likelihood)
  model.train(); likelihood.train()
  opt = torch.optim.Adam(model.parameters(), lr=0.1)
  mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
  for _ in range(200):
    opt.zero_grad(); loss = -mll(model(X), y); loss.backward(); opt.step()
  model.eval(); likelihood.eval()
  return model, likelihood, S_train, y_train


# predicted Heston price with confidence band, n = 50, 100, 150 (book Fig 3.6)
S_test = np.linspace(S_lb, S_ub, 250)
price_exact = heston_vec(S_test, K, r, T, V0, kappa, theta, volvol, rho)

fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
fig.patch.set_facecolor("white")
for ax, n_train in zip(axes, [50, 100, 150]):
  model, likelihood, S_train, y_train = fit_gp(n_train)
  Xte = torch.tensor(rescale(S_test)).unsqueeze(-1)
  with torch.no_grad():
    pred = likelihood(model(Xte))
    mean = pred.mean.numpy()
    lo, hi = pred.confidence_region()
    lo, hi = lo.numpy(), hi.numpy()

  ax.plot(S_train, y_train, "k+", markersize=6, label="Observed")
  ax.plot(S_test, price_exact, "k--", lw=1.6, label="Exact (test)")
  ax.plot(S_test, mean, color="mediumblue", lw=1.8, label="Mean")
  ax.fill_between(S_test, lo, hi, color="tab:blue", alpha=0.40, label="Confidence")

  ax.set_xlabel("S"); ax.set_ylabel("V")
  ax.set_title(f"n = {n_train} training points")
  ax.legend(fontsize=10)

fig.suptitle("Predicted Heston call prices (COS) with GP confidence band - "
             "n = 50, 100, 150 training points", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "heston_price_fits.png"), dpi=140, bbox_inches="tight")
plt.close()
print("saved heston_price_fits.png")


# Heston price surface over (S, volatility): 2-D GP on (S, vol) -> price
vol_lb, vol_ub = 0.20, 0.60        # volatility = sqrt(variance) axis
ng = 22
Sg = np.linspace(S_lb, S_ub, ng)
volg = np.linspace(vol_lb, vol_ub, ng)
Sm, vm = np.meshgrid(Sg, volg)
Xtr_np = np.column_stack([Sm.ravel(), vm.ravel()])
ytr_np = np.array([heston_call_cos(S, K, r, T, vol ** 2, kappa, theta, volvol, rho)
                   for S, vol in Xtr_np])


def rescale2(X):
  '''We rescale the (S, vol) inputs to [0, 1] per axis.'''
  Z = X.copy()
  Z[:, 0] = (X[:, 0] - S_lb) / (S_ub - S_lb)
  Z[:, 1] = (X[:, 1] - vol_lb) / (vol_ub - vol_lb)
  return Z


class ExactGP2D(gpytorch.models.ExactGP):
  '''Our 2-D exact GP on (S, vol) -> price: Matern-5/2 ARD kernel (length-scale per input).'''
  def __init__(self, X, y, likelihood):
    super().__init__(X, y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))

  def forward(self, x):
    return gpytorch.distributions.MultivariateNormal(
        self.mean_module(x), self.covar_module(x))


Xtr = torch.tensor(rescale2(Xtr_np)); ytr = torch.tensor(ytr_np)
lik2 = gpytorch.likelihoods.GaussianLikelihood()
m2 = ExactGP2D(Xtr, ytr, lik2)
m2.train(); lik2.train()
opt = torch.optim.Adam(m2.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik2, m2)
for _ in range(400):
  opt.zero_grad(); loss = -mll(m2(Xtr), ytr); loss.backward(); opt.step()
m2.eval(); lik2.eval()

nt = 40
Sf = np.linspace(S_lb, S_ub, nt)
volf = np.linspace(vol_lb, vol_ub, nt)
Smf, vmf = np.meshgrid(Sf, volf)
Xte_np = np.column_stack([Smf.ravel(), vmf.ravel()])
with torch.no_grad():
  gp_surf = m2(torch.tensor(rescale2(Xte_np))).mean.numpy().reshape(nt, nt)
true_surf = np.array([heston_call_cos(S, K, r, T, vol ** 2, kappa, theta, volvol, rho)
                      for S, vol in Xte_np]).reshape(nt, nt)

fig = plt.figure(figsize=(9, 6)); fig.patch.set_facecolor("white")
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Smf, vmf, true_surf, color="forestgreen", alpha=0.55, linewidth=0)
ax.plot_wireframe(Smf, vmf, gp_surf, color="darkred", linewidth=0.4)
ax.set_xlabel("S"); ax.set_ylabel("volatility"); ax.set_zlabel("V")
ax.set_title("Heston call price surface: analytical (green) vs GP (red wireframe)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "heston_surface.png"), dpi=140, bbox_inches="tight")
plt.close()
print("saved heston_surface.png")


# Greeks by autograd (Delta, Vega) with 95% credible bands, n = 100. Delta
# comes from differentiating the 1-D price GP in S; Vega from differentiating
# the 2-D GP in vol. Reference values are central finite differences on COS.
model, likelihood, S_train, y_train = fit_gp(100)
S_g = np.linspace(80.0, 160.0, 80)
N = len(S_g)

# Delta: autograd of the posterior mean w.r.t. the (rescaled) S input
Xg = torch.tensor(rescale(S_g)).unsqueeze(-1).requires_grad_(True)
mean_g = model(Xg).mean
delta_gp = torch.autograd.grad(mean_g.sum(), Xg, create_graph=False)[0][:, 0].detach().numpy() / (S_ub - S_lb)

# Delta variance: we propagate the GP posterior covariance through a central
# finite-difference operator. Evaluating f at x+h and x-h jointly gives the
# covariance terms we need.
h = 1e-4
Xg_rescaled = rescale(S_g)
X_p = torch.tensor(Xg_rescaled + h).unsqueeze(-1)
X_m = torch.tensor(Xg_rescaled - h).unsqueeze(-1)
X_joint = torch.cat([X_p, X_m], dim=0)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
  cov_joint = model(X_joint).covariance_matrix

var_p = torch.diag(cov_joint[:N, :N])
var_m = torch.diag(cov_joint[N:, N:])
cov_pm = torch.diag(cov_joint[:N, N:])

# Var((f(x+h) - f(x-h)) / 2h) = (Var(f(x+h)) + Var(f(x-h)) - 2 Cov(...)) / 4 h^2
var_delta = (var_p + var_m - 2 * cov_pm) / (4 * h ** 2)
std_delta = torch.sqrt(var_delta).numpy() / (S_ub - S_lb)

delta_lo = delta_gp - 1.96 * std_delta
delta_hi = delta_gp + 1.96 * std_delta

hS = 0.5  # bump for the finite-difference Delta reference
Cp = heston_vec(S_g + hS, K, r, T, V0, kappa, theta, volvol, rho)
Cm = heston_vec(S_g - hS, K, r, T, V0, kappa, theta, volvol, rho)
delta_fd = (Cp - Cm) / (2 * hS)

# Vega: autograd of the 2-D posterior mean w.r.t. the volatility input at vol0
vol0 = np.sqrt(V0)
Xv_np = np.column_stack([S_g, np.full_like(S_g, vol0)])
Xv = torch.tensor(rescale2(Xv_np), requires_grad=True)
mv = m2(Xv).mean
gv = torch.autograd.grad(mv.sum(), Xv, create_graph=False)[0]
vega_gp = gv[:, 1].detach().numpy() / (vol_ub - vol_lb)

# Vega variance: same finite-difference-on-covariance trick, now along the vol axis
h_v = 1e-4
Xv_rescaled = rescale2(Xv_np)
Xv_p = Xv_rescaled.copy()
Xv_p[:, 1] += h_v
Xv_m = Xv_rescaled.copy()
Xv_m[:, 1] -= h_v

Xv_joint = torch.tensor(np.vstack([Xv_p, Xv_m]))
with torch.no_grad(), gpytorch.settings.fast_pred_var():
  cov_v_joint = m2(Xv_joint).covariance_matrix

var_v_p = torch.diag(cov_v_joint[:N, :N])
var_v_m = torch.diag(cov_v_joint[N:, N:])
cov_v_pm = torch.diag(cov_v_joint[:N, N:])

var_vega = (var_v_p + var_v_m - 2 * cov_v_pm) / (4 * h_v ** 2)
std_vega = torch.sqrt(var_vega).numpy() / (vol_ub - vol_lb)

vega_lo = vega_gp - 1.96 * std_vega
vega_hi = vega_gp + 1.96 * std_vega

hv = 1e-3  # bump for the finite-difference Vega reference
Vp = np.array([heston_call_cos(S, K, r, T, (vol0 + hv) ** 2, kappa, theta, volvol, rho) for S in S_g])
Vm = np.array([heston_call_cos(S, K, r, T, (vol0 - hv) ** 2, kappa, theta, volvol, rho) for S in S_g])
vega_fd = (Vp - Vm) / (2 * hv)

# each Greek: finite-difference reference, GP autograd mean, and 95% credible band
fig, ax = plt.subplots(1, 2, figsize=(13, 5)); fig.patch.set_facecolor("white")

ax[0].plot(S_g, delta_fd, "k-", lw=2, label="Exact (test)")
ax[0].plot(S_g, delta_gp, "r--", lw=1.6, label="Mean")
ax[0].fill_between(S_g, delta_lo, delta_hi, color="tab:red", alpha=0.25, label="Confidence")
ax[0].set_xlabel("S"); ax[0].set_title(r"Delta $\partial C/\partial S$"); ax[0].legend(fontsize=9)

ax[1].plot(S_g, vega_fd, "k-", lw=2, label="Exact (test)")
ax[1].plot(S_g, vega_gp, "r--", lw=1.6, label="Mean")
ax[1].fill_between(S_g, vega_lo, vega_hi, color="tab:red", alpha=0.25, label="Confidence")
ax[1].set_xlabel("S"); ax[1].set_title(r"Vega $\partial C/\partial\,\mathrm{vol}$"); ax[1].legend(fontsize=9)

fig.suptitle("Heston Greeks by autograd vs finite differences (n = 100)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "heston_greeks.png"), dpi=140, bbox_inches="tight")
plt.close()
print("saved heston_greeks.png")

# error table for the price fits (50, 100, 150)
print("\nn_train   price mean|err|   price max|err|")
for n_train in [50, 100, 150]:
  model, likelihood, S_train, y_train = fit_gp(n_train)
  Xte = torch.tensor(rescale(S_test)).unsqueeze(-1)
  with torch.no_grad():
    mean = likelihood(model(Xte)).mean.numpy()
  mask = (S_test >= S_train.min()) & (S_test <= S_train.max())  # interpolation region
  e = np.abs(mean[mask] - price_exact[mask])
  print(f"  {n_train:4d}      {e.mean():.3e}        {e.max():.3e}")

# error table for the Greeks (GP autograd vs finite differences)
print("\nGreek   mean|err|   max|err|")
for name, gp, fd in [("Delta", delta_gp, delta_fd), ("Vega", vega_gp, vega_fd)]:
  e = np.abs(gp - fd)
  print(f"  {name:6s} {e.mean():.3e}   {e.max():.3e}")
