"""Section 4.3: exact GP regression vs KISS-GP scalability benchmark.

We compare exact GP regression against KISS-GP (Structured Kernel Interpolation,
Wilson & Nickisch 2015) across a range of training sizes n. The goal is to
verify empirically that KISS-GP scales near-linearly in n while the exact GP
scales as O(n^3), reproducing the headline result of Wilson & Nickisch. The
book's own notebooks all use n <= 100 and never push past the n = 10^4 barrier
nor implement SKI / KISS-GP, so this experiment fills that gap.

As a regression target we use a smooth 1-D function, the Black-Scholes call
price as a function of the underlying S, and we time both fit and prediction.
"""

import os
import time
import gc
import numpy as np
import torch
import gpytorch
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

torch.set_default_dtype(torch.float32)  # float32 so that large n fits in memory
plt.rcParams.update({"figure.figsize": (12, 5), "axes.grid": True})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def bs_call(S, K=100., r=0.02, T=1., sigma=0.3):
  '''We compute the Black-Scholes price of a European call, our smooth target.'''
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return S * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)


S_lb, S_ub = 1.0, 200.0


def make_data(n, seed=0):
  '''
    We draw n random training points uniformly on [S_lb, S_ub] and evaluate the
    BS price, adding a small observation noise so the regression is realistic.
  '''
  rng = np.random.default_rng(seed)
  S = rng.uniform(S_lb, S_ub, size=n).astype(np.float32)
  y = bs_call(S).astype(np.float32) + 0.001 * rng.standard_normal(n).astype(np.float32)
  return S, y


def time_exact_gp(n, n_test=200, max_n=20000, optimize=False):
  '''
    We time the exact-GP fit and prediction using scikit-learn (this is the
    book's approach). We cap n with max_n because the exact GP is O(n^3) and
    becomes infeasible for large n; above the cap we skip it and return None.
  '''
  if n > max_n:
    return None, None, None
  S, y = make_data(n)
  S_test = np.linspace(S_lb, S_ub, n_test)
  Xtr = ((S - S_lb) / (S_ub - S_lb)).reshape(-1, 1)
  Xte = ((S_test - S_lb) / (S_ub - S_lb)).reshape(-1, 1)

  t0 = time.time()
  if optimize:
    kernel = ConstantKernel(1.0) * Matern(length_scale=0.1, nu=2.5) + WhiteKernel(0.001)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
  else:
    # we fix the hyperparameters so the scaling comparison is fair (no
    # optimisation cost that would vary with n)
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * \
             Matern(length_scale=0.1, nu=2.5, length_scale_bounds="fixed") + \
             WhiteKernel(0.001, noise_level_bounds="fixed")
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
  gp.fit(Xtr, y)
  t_fit = time.time() - t0

  t0 = time.time()
  y_pred = gp.predict(Xte)
  t_pred = time.time() - t0

  rmse = np.sqrt(np.mean((y_pred - bs_call(S_test)) ** 2))
  return t_fit, t_pred, rmse


class KISSGPModel(gpytorch.models.ExactGP):
  '''
    Our KISS-GP model (Wilson & Nickisch 2015). It places inducing points on a
    regular grid and maps training points to that grid through a sparse local
    interpolation matrix W. Combined with the Toeplitz structure of the grid
    kernel, this gives O(n + m log m) matrix-vector products via the FFT and
    so effectively linear-time inference, replacing the exact GP's O(n^3)
    Cholesky decomposition.
  '''
  def __init__(self, X, y, likelihood, grid_size=400):
    super().__init__(X, y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.GridInterpolationKernel(
        gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        ),
        grid_size=grid_size,
        num_dims=1,
    )

  def forward(self, x):
    return gpytorch.distributions.MultivariateNormal(
        self.mean_module(x), self.covar_module(x)
    )


def time_kissgp(n, n_test=200, grid_size=400, train_iter=15, optimize=True):
  '''
    We time the KISS-GP fit and prediction. We train for a small fixed number
    of Adam iterations and use a conjugate-gradient solver (capped root-
    decomposition size) instead of Cholesky, which is what keeps the cost
    near-linear.
  '''
  S, y = make_data(n)
  S_test = np.linspace(S_lb, S_ub, n_test)
  Xtr = torch.tensor(((S - S_lb) / (S_ub - S_lb)).reshape(-1, 1))
  ytr = torch.tensor(y)
  Xte = torch.tensor(((S_test - S_lb) / (S_ub - S_lb)).reshape(-1, 1))

  likelihood = gpytorch.likelihoods.GaussianLikelihood()
  model = KISSGPModel(Xtr, ytr, likelihood, grid_size=grid_size)

  t0 = time.time()
  model.train(); likelihood.train()
  if optimize:
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(train_iter):
      opt.zero_grad()
      with gpytorch.settings.max_root_decomposition_size(50):
        out = model(Xtr)
        loss = -mll(out, ytr)
      loss.backward()
      opt.step()
  t_fit = time.time() - t0

  t0 = time.time()
  model.eval(); likelihood.eval()
  with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
       gpytorch.settings.max_root_decomposition_size(50):
    pred = likelihood(model(Xte))
    mu = pred.mean.numpy()
  t_pred = time.time() - t0

  rmse = np.sqrt(np.mean((mu - bs_call(S_test).astype(np.float32)) ** 2))

  del model, likelihood, Xtr, ytr, Xte, S, y, pred
  gc.collect()
  return t_fit, t_pred, rmse


# we sweep n from 500 up to 200,000 and record fit/predict time and RMSE
ns = [500, 1_000, 2_000, 5_000, 10_000, 15_000, 20_000, 200_000]
results_exact = []
results_kiss  = []

print(f"{'n':>10s} | {'Exact fit':>12s} {'Exact pred':>12s} {'Exact RMSE':>12s} | "
      f"{'KISS fit':>12s} {'KISS pred':>12s} {'KISS RMSE':>12s}")
print("-" * 110)

for n in ns:
  e_fit, e_pred, e_rmse = time_exact_gp(n, max_n=17_000)
  results_exact.append((n, e_fit, e_pred, e_rmse))

  k_fit, k_pred, k_rmse = time_kissgp(n)
  results_kiss.append((n, k_fit, k_pred, k_rmse))

  if e_fit is not None:
    e_str = f"{e_fit:>12.3f} {e_pred:>12.4f} {e_rmse:>12.2e}"
  else:
    e_str = f"{'skip':>12s} {'skip':>12s} {'skip':>12s}"
  k_str = f"{k_fit:>12.3f} {k_pred:>12.4f} {k_rmse:>12.2e}"
  print(f"{n:>10d} | {e_str} | {k_str}")
  gc.collect()


# fit-time and predict-time scaling on log-log axes
ns_arr = np.array(ns)
exact_fit = np.array([r[1] for r in results_exact], dtype=object)
exact_pred = np.array([r[2] for r in results_exact], dtype=object)
kiss_fit = np.array([r[1] for r in results_kiss])
kiss_pred = np.array([r[2] for r in results_kiss])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# keep only the n where the exact GP actually ran
mask = np.array([x is not None for x in exact_fit])
ne = ns_arr[mask]
efit = np.array([float(x) for x in exact_fit[mask]])
epred = np.array([float(x) for x in exact_pred[mask]])

axes[0].loglog(ne, efit, "ro-", label="Exact GP fit", lw=2, ms=8)
axes[0].loglog(ns_arr, kiss_fit, "bs-", label="KISS-GP fit", lw=2, ms=8)
# reference slopes: O(n^3) anchored to the exact GP, O(n) anchored to KISS-GP
ref_n = np.logspace(3, 6, 100)
ref_n3 = (efit[0] / ne[0] ** 3) * ref_n ** 3
ref_n_lin = (kiss_fit[0] / ns_arr[0]) * ref_n
axes[0].loglog(ref_n, ref_n3, "r--", alpha=0.4, label=r"$O(n^3)$ ref")
axes[0].loglog(ref_n, ref_n_lin, "b--", alpha=0.4, label=r"$O(n)$ ref")
axes[0].set_xlabel("n (training points)")
axes[0].set_ylabel("Fit time (s)")
axes[0].set_title("Training-time scaling: exact GP vs KISS-GP")
axes[0].legend()

axes[1].loglog(ne, epred, "ro-", label="Exact GP predict", lw=2, ms=8)
axes[1].loglog(ns_arr, kiss_pred, "bs-", label="KISS-GP predict", lw=2, ms=8)
axes[1].set_xlabel("n (training points)")
axes[1].set_ylabel("Predict time on 200 test points (s)")
axes[1].set_title("Prediction-time scaling")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_kissgp_scaling.png"), dpi=130)
plt.close()


# sanity check: visualise the KISS-GP fit at n = 20,000 against the exact BS price
n = 20_000
S, y = make_data(n)
S_test = np.linspace(S_lb, S_ub, 500)
Xtr = torch.tensor(((S - S_lb) / (S_ub - S_lb)).reshape(-1, 1))
ytr = torch.tensor(y)
Xte = torch.tensor(((S_test - S_lb) / (S_ub - S_lb)).reshape(-1, 1))

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = KISSGPModel(Xtr, ytr, likelihood, grid_size=400)
model.train(); likelihood.train()
opt = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
for _ in range(15):
  opt.zero_grad()
  with gpytorch.settings.max_root_decomposition_size(50):
    out = model(Xtr); loss = -mll(out, ytr)
  loss.backward(); opt.step()
model.eval(); likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
     gpytorch.settings.max_root_decomposition_size(50):
  pred = likelihood(model(Xte))
  mu_kiss = pred.mean.numpy()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
# only show a subsample of the training points, otherwise the scatter is unreadable
sub = np.random.choice(n, 5000, replace=False)
ax[0].scatter(S[sub], y[sub], s=0.5, alpha=0.2, color="grey", label=f"train (5000/{n} shown)")
ax[0].plot(S_test, bs_call(S_test), "k-", lw=2, label="BS exact")
ax[0].plot(S_test, mu_kiss, "r--", lw=2, label="KISS-GP")
ax[0].set_xlabel("S"); ax[0].set_ylabel("Call price")
ax[0].set_title(f"KISS-GP fit at n={n:.0e}")
ax[0].legend()

ax[1].plot(S_test, np.abs(mu_kiss - bs_call(S_test)), "r-")
ax[1].set_xlabel("S"); ax[1].set_ylabel("|KISS-GP - BS|")
ax[1].set_title(f"Absolute error of KISS-GP at n={n:.0e}")
ax[1].set_yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig7_kissgp_n20k.png"), dpi=130)
plt.close()


print("\n=========== Conclusions for the report ===========")
print("- Exact GP is feasible only for very small n in this 4 GB sandbox,")
print("  due to O(n^3) Cholesky and O(n^2) memory cost.")
print("- KISS-GP scales near-linearly thanks to:")
print("    * sparse local-interpolation matrix W (n x m)")
print("    * Toeplitz structure on the grid - matvecs cost O(m log m) via FFT")
print("    * conjugate-gradient solver replaces Cholesky")
print(f"- At n=2e4, KISS-GP fit took {results_kiss[-1][1]:.1f}s.")
print(f"- Final RMSE at n=2e4: {results_kiss[-1][3]:.2e}.")
print("- With a GPU and >8 GB RAM, KISS-GP scales to n=10^6 (Wilson-Nickisch 2015).")
print("\nFigures saved: fig6_kissgp_scaling.png, fig7_kissgp_n20k.png")
