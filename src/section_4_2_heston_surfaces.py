"""Section 4.2.4: reproduction of the book's Figure 3.3 - Heston price surfaces.

We build the gridded Heston call (top row) and put (bottom row) price surfaces
over (S, volatility), at three times to maturity T - t = 1.0, 0.5, 0.1. Within
each panel we fit a 2-D GP to the surface on a 30 x 30 grid, then evaluate it
out-of-sample on a finer 40 x 40 grid so that most test points are new to the
model. Prices come from the vectorised COS pricer (Fang & Oosterlee).
Parameters follow the book's Table 3.1 / 3.2.
"""

import os
import sys
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from pricers.heston_cos import heston_call_cos_vec, heston_put_cos_vec  # noqa: E402

torch.set_default_dtype(torch.float64)
np.random.seed(0)
torch.manual_seed(0)

OUTPUT_DIR = os.path.join(os.path.dirname(_HERE), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# book Table 3.1 / 3.2 parameters
kappa, theta, volvol, r = 0.1, 0.15, 0.1, 0.002
K, rho = 100.0, -0.9
S_lb, S_ub = 60.0, 200.0
vol_lb, vol_ub = 0.20, 0.60


def surface_prices(Sf, volf, tau, K, r, kappa, theta, volvol, rho, kind):
  '''
    We build the analytical price surface over the (S, volatility) grid for one
    maturity tau, pricing each volatility row with the vectorised COS pricer
    (volatility enters as the initial variance vol**2).
  '''
  Sm, vm = np.meshgrid(Sf, volf)
  Z = np.empty_like(Sm)
  for i, vol in enumerate(volf):
    if kind == "call":
      Z[i, :] = heston_call_cos_vec(Sf, K, r, tau, vol ** 2, kappa, theta, volvol, rho)
    else:
      Z[i, :] = heston_put_cos_vec(Sf, K, r, tau, vol ** 2, kappa, theta, volvol, rho)
  return Sm, vm, Z


def rescale2(X):
  '''We rescale the (S, vol) inputs to [0, 1] per axis, for GP conditioning.'''
  Z = X.copy()
  Z[:, 0] = (X[:, 0] - S_lb) / (S_ub - S_lb)
  Z[:, 1] = (X[:, 1] - vol_lb) / (vol_ub - vol_lb)
  return Z


class GP2(gpytorch.models.ExactGP):
  '''Our 2-D exact GP model: constant mean and a scaled Matern-5/2 ARD kernel.'''
  def __init__(self, X, y, lik):
    super().__init__(X, y, lik)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))

  def forward(self, x):
    return gpytorch.distributions.MultivariateNormal(
        self.mean_module(x), self.covar_module(x))


def fit_surface_gp(tau, kind):
  '''
    We fit the 2-D GP to one price surface (call or put) at maturity tau on a
    30 x 30 grid, optimising the hyperparameters by maximising the marginal log
    likelihood with 180 Adam iterations.
  '''
  n_tr = 30
  Sg = np.linspace(S_lb, S_ub, n_tr)
  volg = np.linspace(vol_lb, vol_ub, n_tr)
  Sm, vm = np.meshgrid(Sg, volg)
  base = np.column_stack([Sm.ravel(), vm.ravel()])
  _, _, Z = surface_prices(Sg, volg, tau, K, r, kappa, theta, volvol, rho, kind)
  X = torch.tensor(rescale2(base)); y = torch.tensor(Z.ravel())
  lik = gpytorch.likelihoods.GaussianLikelihood()
  model = GP2(X, y, lik)
  model.train(); lik.train()
  opt = torch.optim.Adam(model.parameters(), lr=0.1)
  mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
  for _ in range(180):
    opt.zero_grad(); loss = -mll(model(X), y); loss.backward(); opt.step()
  model.eval(); lik.eval()
  return model


taus = [1.0, 0.5, 0.1]
nt = 40
Sf = np.linspace(S_lb, S_ub, nt)
volf = np.linspace(vol_lb, vol_ub, nt)
Smf, vmf = np.meshgrid(Sf, volf)
test_base = np.column_stack([Smf.ravel(), vmf.ravel()])

fig = plt.figure(figsize=(16, 9)); fig.patch.set_facecolor("white")
report = []
sign_report = []

for col, tau in enumerate(taus):
  for row, kind in enumerate(["call", "put"]):
    model = fit_surface_gp(tau, kind)
    with torch.no_grad():
      gp_z = model(torch.tensor(rescale2(test_base))).mean.numpy().reshape(nt, nt)
    _, _, true_z = surface_prices(Sf, volf, tau, K, r, kappa, theta, volvol, rho, kind)
    ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
    # colour each panel by the measured sign of the GP error (red if the GP
    # sits on average above the analytical surface, green if below)
    signed_err = (gp_z - true_z).mean()
    colour = "indianred" if signed_err > 0 else "forestgreen"
    ax.plot_surface(Smf, vmf, true_z, color=colour, alpha=0.3, linewidth=0)
    ax.plot_wireframe(Smf, vmf, gp_z, color="black", linewidth=0.4, alpha=0.6)
    ax.set_xlabel("S", fontsize=8, labelpad=2); ax.set_ylabel("volatility", fontsize=8, labelpad=2)
    ax.set_zlabel("V", fontsize=8, labelpad=2); ax.tick_params(labelsize=6)
    ax.set_title(f"({chr(97+col)}) {kind.capitalize()}: T - t = {tau}", fontsize=11)
    e = np.abs(gp_z - true_z); report.append((tau, kind, e.mean(), e.max()))
    sign_report.append((tau, kind, (gp_z - true_z).mean()))
    print(f"  done: {kind} T-t={tau}")


fig.suptitle("Heston call (top) and put (bottom) price surfaces vs GP estimate, "
             "at T - t = 1.0, 0.5, 0.1\n(analytical = coloured surface, GP = black wireframe)",
             fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "heston_surfaces_fig33.png"), dpi=140, bbox_inches="tight")
plt.close()
print("saved heston_surfaces_fig33.png\n")

# absolute-error table per panel
print("maturity  type   mean|err|   max|err|")
for tau, kind, me, mx in report:
  print(f"  {tau:4.1f}   {kind:5s}  {me:.3e}   {mx:.3e}")

# signed-error diagnostic: helps sanity-check the colouring convention
print("\nmaturity  type   signed mean err")
for tau, kind, se in sign_report:
  print(f"  {tau:4.1f}   {kind:5s}  {se:+.3e}")
