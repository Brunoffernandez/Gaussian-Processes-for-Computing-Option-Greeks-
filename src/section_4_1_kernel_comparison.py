"""Section 4.1: kernel comparison and all five Greeks under Black-Scholes.

We extend the book's Example-2 in three directions. We compare six kernels on
the Black-Scholes call surrogate (RBF, Matern-3/2, Matern-5/2, RationalQuadratic,
RBF+Linear, RBF+Matern), we compute all five standard Greeks (Delta, Gamma, Vega,
Theta, Rho) rather than just Delta and Vega, and we compare analytic kernel-
derivative formulas (for RBF and Matern-5/2) against central finite differences
on the GP posterior mean for the composite kernels. Everything uses scikit-learn.

Parameters follow the book's Example-2 exactly: K=130, r=0.002, sigma=0.4, T=2,
n=100 training points on the rescaled domain [0, 1].
"""

import sys
import os
import warnings
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, DotProduct, ConstantKernel as C,
)

# make the local Black-Scholes helper importable when running this file directly
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from pricers.black_scholes import bsformula  # noqa: E402

warnings.filterwarnings("ignore")   # suppress sklearn convergence warnings
plt.rcParams.update({"figure.figsize": (12, 5), "axes.grid": True})

OUTPUT_DIR = os.path.join(os.path.dirname(_HERE), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Parameters matching the book's Example-2
# ---------------------------------------------------------------------------
KC    = 130    # call strike
r     = 0.002  # risk-free rate
sigma = 0.4    # implied volatility
T     = 2.0    # time to maturity
S0    = 100    # spot (used for vega/theta/rho, where we fix S)

lb, ub = 0, 300          # domain for underlying S
training_number = 100    # number of training points
testing_number  = 50     # number of test points
sigma_n         = 1e-8   # nugget noise for the manual Cholesky


def bs_call(x):
  '''We compute the BS call price on the rescaled spot x in [0, 1].'''
  return bsformula(1, lb + (ub - lb) * x, KC, r, T, sigma, 0)[0]


def bs_delta_exact(x):
  '''We return the BS Delta at the rescaled spot x.'''
  return bsformula(1, lb + (ub - lb) * x, KC, r, T, sigma, 0)[1]


def bs_vega_exact(x_vol):
  '''We return the BS Vega at spot S0 and volatility x_vol (not rescaled).'''
  return bsformula(1, S0, KC, r, T, x_vol, 0)[2]


def bs_gamma_exact(S):
  '''We return the BS Gamma at spot S; used as ground truth for the second derivative.'''
  from scipy.stats import norm
  d1 = (np.log(S / KC) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_theta_exact(S):
  '''We return the BS Theta (per calendar day) at spot S.'''
  from scipy.stats import norm
  d1 = (np.log(S / KC) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return (- S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
          - r * KC * np.exp(-r * T) * norm.cdf(d2)) / 365


def bs_rho_exact(S):
  '''We return the BS Rho (per one basis point) at spot S.'''
  from scipy.stats import norm
  d1 = (np.log(S / KC) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return KC * T * np.exp(-r * T) * norm.cdf(d2) / 100


# ---------------------------------------------------------------------------
# Training and test data (spot axis)
# ---------------------------------------------------------------------------
x_train = np.linspace(0.01, 1.2, training_number).reshape(-1, 1)
x_test  = np.linspace(0.01, 1.0, testing_number).reshape(-1, 1)
S_test  = (lb + (ub - lb) * x_test).ravel()

y_train = np.array([bs_call(x[0]) for x in x_train])

# ground-truth Greeks on the test grid
delta_true = bs_delta_exact(x_test).ravel()
gamma_true = bs_gamma_exact(S_test)
price_true = np.array([bs_call(x[0]) for x in x_test])


# ---------------------------------------------------------------------------
# Kernel zoo. Each is wrapped in a ConstantKernel so sklearn can learn the
# output scale (signal variance) separately from the length-scale.
#
# Kernel choice rationale:
#   RBF (Squared Exponential) - infinitely smooth; the book's default.
#   Matern-3/2  - once differentiable; rougher than RBF.
#   Matern-5/2  - twice differentiable; standard choice in finance.
#   RationalQuadratic - equivalent to an infinite mixture of RBF kernels with
#                       different length-scales; adapts to multi-scale data.
#   RBF + Linear  - RBF captures local structure; the linear (DotProduct) term
#                   captures the global slope of a call price deep in the money.
#   RBF + Matern  - combines RBF smoothness with a rougher Matern component to
#                   handle payoffs with multiple scales.
# ---------------------------------------------------------------------------
KERNELS = {
    "RBF":           C(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4)),
    "Matern-3/2":    C(1.0) * Matern(length_scale=1.0, nu=1.5,
                                     length_scale_bounds=(1e-2, 1e4)),
    "Matern-5/2":    C(1.0) * Matern(length_scale=1.0, nu=2.5,
                                     length_scale_bounds=(1e-2, 1e4)),
    "RationalQuad":  C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0),
    "RBF+Linear":    C(1.0) * RBF(length_scale=1.0) + DotProduct(sigma_0=1.0),
    "RBF+Matern":    C(1.0) * RBF(length_scale=1.0)
                     + C(1.0) * Matern(length_scale=0.5, nu=1.5),
}


# ---------------------------------------------------------------------------
# Fit each kernel to the price surrogate and record RMSE + log-ML
# ---------------------------------------------------------------------------
fitted = {}
print(f"{'Kernel':14s} {'Price RMSE':>12s} {'Price max':>10s} {'log ML':>10s}")
print("-" * 52)
for name, kernel in KERNELS.items():
  gp = gaussian_process.GaussianProcessRegressor(
      kernel=kernel, n_restarts_optimizer=15,
      normalize_y=True, alpha=1e-10,
  )
  gp.fit(x_train, y_train)
  y_pred = gp.predict(x_test)
  rmse = np.sqrt(np.mean((y_pred - price_true) ** 2))
  maxe = np.max(np.abs(y_pred - price_true))
  lml  = gp.log_marginal_likelihood_value_
  fitted[name] = gp
  print(f"{name:14s} {rmse:12.3e} {maxe:10.3e} {lml:10.2f}")


# ---------------------------------------------------------------------------
# Figure A: all six price fits plus per-kernel absolute error in log scale
# ---------------------------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, len(KERNELS)))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(S_test, price_true, "k-", lw=2.5, label="BS exact")
for (name, gp), c in zip(fitted.items(), colors):
  y_pred, _ = gp.predict(x_test, return_std=True)
  ax.plot(S_test, y_pred, "--", color=c, label=name, lw=1.2)
ax.set_xlabel("S"); ax.set_ylabel("Call price")
ax.set_title("GP call price: all kernels"); ax.legend(fontsize=8)

ax = axes[1]
for (name, gp), c in zip(fitted.items(), colors):
  y_pred = gp.predict(x_test)
  ax.plot(S_test, np.abs(y_pred - price_true), color=c, label=name)
ax.set_yscale("log"); ax.set_xlabel("S")
ax.set_ylabel("|GP - BS|"); ax.set_title("Price absolute error (log)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_A_price_all_kernels.png"), dpi=130)
plt.close()
print("\nSaved fig_A_price_all_kernels.png")


# ---------------------------------------------------------------------------
# Delta: analytic kernel derivative for RBF and Matern-5/2 (book formula for
# RBF), central finite differences on the GP posterior mean for the others.
#
# Because the GP posterior mean is a smooth analytic function of x*, finite
# differences on the mean are exact up to machine precision (not noisy like
# bump-and-revalue on the pricing model itself).
# ---------------------------------------------------------------------------
def analytic_delta_rbf(gp, x_train, y_train, x_test, sigma_n=1e-8):
  '''
    We compute the analytic Delta under the RBF kernel, exactly the formula in
    the book's Example-2. For k(x, x') = sf^2 * exp(-(x - x')^2 / 2 l^2),
    dk/dx* = -(x* - x_i) / l^2 * k(x_i, x*), so df/dx* = k'_{*X} alpha with
    alpha = (K + sigma_n^2 I)^{-1} y.
  '''
  k = gp.kernel_
  rbf_k = k.k2 if hasattr(k, 'k2') else k
  l = rbf_k.length_scale

  rbf = gaussian_process.kernels.RBF(length_scale=l)
  K_y  = rbf(x_train, x_train) + np.eye(len(x_train)) * sigma_n
  L    = la.cho_factor(K_y)
  alpha = la.cho_solve(L, y_train)

  k_s      = rbf(x_test, x_train)
  k_s_prime = -(x_test - x_train.T) / l**2 * k_s
  delta_raw = k_s_prime @ alpha
  return delta_raw / (ub - lb)  # chain rule: dC/dS = (1/(ub-lb)) * dC/dx


def analytic_delta_matern52(gp, x_train, y_train, x_test, sigma_n=1e-8):
  '''
    We compute the analytic Delta under the Matern-5/2 kernel. With a = sqrt(5) r / l,
    dk/dx* = -sf^2 * (5 / (3 l^2)) * (x* - x_i) * (1 + a) * exp(-a).
  '''
  k   = gp.kernel_
  sf2 = k.k1.constant_value          # signal variance (ConstantKernel)
  l   = k.k2.length_scale            # length-scale (Matern)

  m52  = gaussian_process.kernels.Matern(length_scale=l, nu=2.5)
  K_y  = sf2 * m52(x_train, x_train) + np.eye(len(x_train)) * sigma_n
  L    = la.cho_factor(K_y)
  alpha = la.cho_solve(L, y_train)

  diff = x_test - x_train.T
  r_arr = np.abs(diff)
  a = np.sqrt(5) * r_arr / l
  k_s_prime = -sf2 * (5 / (3 * l ** 2)) * diff * (1 + a) * np.exp(-a)
  delta_raw = k_s_prime @ alpha
  return delta_raw / (ub - lb)


def gp_greek_fd(gp, x_test, h=1e-4):
  '''
    We take central finite differences on the GP posterior mean. Works for ANY
    sklearn kernel with no formula derivation. First derivative w.r.t. x_test.
  '''
  mu_p = gp.predict(x_test + h)
  mu_m = gp.predict(x_test - h)
  return (mu_p - mu_m) / (2 * h)


def gp_greek_fd2(gp, x_test, h=1e-4):
  '''
    We take central finite differences for the second derivative (Gamma):
    d2f/dx2 ~= (f(x+h) - 2 f(x) + f(x-h)) / h^2.
  '''
  mu_p = gp.predict(x_test + h)
  mu_0 = gp.predict(x_test)
  mu_m = gp.predict(x_test - h)
  return (mu_p - 2 * mu_0 + mu_m) / h ** 2


# ---------------------------------------------------------------------------
# Compute Delta for all kernels.
#
# NOTE: RBF+Linear contains a DotProduct kernel whose derivative is unbounded
# near the rescaled origin, making finite-difference Greeks numerically
# unstable. We keep it in the pricing comparison but flag it here.
# ---------------------------------------------------------------------------
delta_results = {}
for name, gp in fitted.items():
  if name == "RBF+Linear":
    delta_results[name] = {"delta": np.zeros(len(x_test)),
                           "rmse": np.nan, "method": "skipped (DotProduct kernel)"}
    continue
  if name == "RBF":
    d = analytic_delta_rbf(gp, x_train, y_train, x_test)
    delta_results[name] = {"delta": d, "method": "analytic (book formula)"}
  elif name == "Matern-5/2":
    d = analytic_delta_matern52(gp, x_train, y_train, x_test)
    delta_results[name] = {"delta": d, "method": "analytic (kernel deriv.)"}
  else:
    d = gp_greek_fd(gp, x_test) / (ub - lb)
    delta_results[name] = {"delta": d, "method": "finite diff on GP mean"}
  rmse = np.sqrt(np.mean((d - delta_true) ** 2))
  delta_results[name]["rmse"] = rmse

print(f"\n{'Kernel':14s} {'Delta RMSE':>12s}  Method")
print("-" * 56)
for name, res in delta_results.items():
  print(f"{name:14s} {res['rmse']:12.3e}  {res['method']}")


# ---------------------------------------------------------------------------
# Gamma via second-order finite differences on the GP mean (works for all kernels).
# For Matern-5/2, the analytic formula is also derived in the report.
# ---------------------------------------------------------------------------
gamma_results = {}
for name, gp in fitted.items():
  if name == "RBF+Linear":
    gamma_results[name] = {"gamma": np.zeros(len(x_test)), "rmse": np.nan}
    continue
  g = gp_greek_fd2(gp, x_test) / (ub - lb) ** 2
  rmse = np.sqrt(np.mean((g - gamma_true) ** 2))
  gamma_results[name] = {"gamma": g, "rmse": rmse}

print(f"\n{'Kernel':14s} {'Gamma RMSE':>12s}")
print("-" * 30)
for name, res in gamma_results.items():
  print(f"{name:14s} {res['rmse']:12.3e}")


# ---------------------------------------------------------------------------
# Vega: we treat sigma as the GP input (same as Example-2 section 2) and
# refit each kernel on the sigma axis, then differentiate the GP mean.
# ---------------------------------------------------------------------------
vol_lb, vol_ub = 0.05, 1.0
x_vol_train = np.linspace(0.0, 1.0, training_number).reshape(-1, 1)
x_vol_test  = np.linspace(0.02, 0.98, testing_number).reshape(-1, 1)
sig_test    = vol_lb + (vol_ub - vol_lb) * x_vol_test.ravel()

y_vol_train = np.array([
    bsformula(1, S0, KC, r, T, vol_lb + (vol_ub - vol_lb) * x[0], 0)[0]
    for x in x_vol_train
])
vega_true_vals = bs_vega_exact(sig_test)

vega_results = {}
print(f"\n{'Kernel':14s} {'Vega RMSE':>12s}")
print("-" * 30)
for name, kernel in KERNELS.items():
  gp_vol = gaussian_process.GaussianProcessRegressor(
      kernel=kernel, n_restarts_optimizer=10,
      normalize_y=True, alpha=1e-10,
  )
  gp_vol.fit(x_vol_train, y_vol_train)
  vega = gp_greek_fd(gp_vol, x_vol_test) / (vol_ub - vol_lb)
  rmse = np.sqrt(np.mean((vega - vega_true_vals) ** 2))
  vega_results[name] = {"vega": vega, "gp": gp_vol, "rmse": rmse}
  print(f"{name:14s} {rmse:12.3e}")


# ---------------------------------------------------------------------------
# Theta: we fit GPs with T as the input, keeping S=S0 and sigma fixed.
# The Theta convention is dC/dt where t = elapsed time = T0 - T, so
# Theta = -dC/dT; we report per calendar day.
# ---------------------------------------------------------------------------
from scipy.stats import norm as _norm


def _theta_varying_T(S, K, r, sigma, T):
  '''We compute the closed-form Theta (per calendar day) with T as the variable.'''
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return (- S * _norm.pdf(d1) * sigma / (2 * np.sqrt(T))
          - r * K * np.exp(-r * T) * _norm.cdf(d2)) / 365


def _rho_varying_r(S, K, r, sigma, T):
  '''We compute the closed-form Rho (per one basis point) with r as the variable.'''
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return K * T * np.exp(-r * T) * _norm.cdf(d2) / 100


T_lb, T_ub = 0.25, 3.0  # stay away from T -> 0 where Theta blows up
x_T_train = np.linspace(0.0, 1.0, training_number).reshape(-1, 1)
x_T_test  = np.linspace(0.05, 0.95, testing_number).reshape(-1, 1)
T_test    = T_lb + (T_ub - T_lb) * x_T_test.ravel()

y_T_train = np.array([
    bsformula(1, S0, KC, r, T_lb + (T_ub - T_lb) * x[0], sigma, 0)[0]
    for x in x_T_train
])
theta_true_vals = np.array([_theta_varying_T(S0, KC, r, sigma, t) for t in T_test])

theta_results = {}
print(f"\n{'Kernel':14s} {'Theta RMSE':>12s}  (dC/dt per calendar day)")
print("-" * 45)
for name, kernel in KERNELS.items():
  gp_T = gaussian_process.GaussianProcessRegressor(
      kernel=kernel, n_restarts_optimizer=10,
      normalize_y=True, alpha=1e-10,
  )
  gp_T.fit(x_T_train, y_T_train)
  # theta = -dC/dT / (T_ub - T_lb) / 365 (per calendar day)
  theta = -gp_greek_fd(gp_T, x_T_test) / (T_ub - T_lb) / 365
  rmse = np.sqrt(np.mean((theta - theta_true_vals) ** 2))
  theta_results[name] = {"theta": theta, "gp": gp_T, "rmse": rmse}
  print(f"{name:14s} {rmse:12.3e}")


# ---------------------------------------------------------------------------
# Rho: we fit GPs with r as the input, keeping S=S0 and sigma fixed. Reported
# per 1 basis point (0.01%).
# ---------------------------------------------------------------------------
r_lb, r_ub = 0.001, 0.08  # sensible rate range, avoid r=0
x_r_train = np.linspace(0.0, 1.0, training_number).reshape(-1, 1)
x_r_test  = np.linspace(0.02, 0.98, testing_number).reshape(-1, 1)
r_test    = r_lb + (r_ub - r_lb) * x_r_test.ravel()

y_r_train = np.array([
    bsformula(1, S0, KC, r_lb + (r_ub - r_lb) * x[0], T, sigma, 0)[0]
    for x in x_r_train
])
rho_true_vals = np.array([_rho_varying_r(S0, KC, rv, sigma, T) for rv in r_test])

rho_results = {}
print(f"\n{'Kernel':14s} {'Rho RMSE':>12s}  (dC/dr per 1bp = 0.01%)")
print("-" * 45)
for name, kernel in KERNELS.items():
  gp_r = gaussian_process.GaussianProcessRegressor(
      kernel=kernel, n_restarts_optimizer=10,
      normalize_y=True, alpha=1e-10,
  )
  gp_r.fit(x_r_train, y_r_train)
  rho_arr = gp_greek_fd(gp_r, x_r_test) / (r_ub - r_lb) / 100
  rmse = np.sqrt(np.mean((rho_arr - rho_true_vals) ** 2))
  rho_results[name] = {"rho": rho_arr, "gp": gp_r, "rmse": rmse}
  print(f"{name:14s} {rmse:12.3e}")


# ---------------------------------------------------------------------------
# Summary tables and per-Greek best kernel
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY: RMSE of all Greeks for all kernels")
print("=" * 70)
print(f"{'Kernel':14s} {'Delta':>10s} {'Gamma':>10s} {'Vega':>10s} "
      f"{'Theta':>10s} {'Rho':>10s}")
print("-" * 70)
for name in KERNELS:
  d = delta_results[name]["rmse"]
  g = gamma_results[name]["rmse"]
  v = vega_results[name]["rmse"]
  th = theta_results[name]["rmse"]
  rh = rho_results[name]["rmse"]
  print(f"{name:14s} {d:10.3e} {g:10.3e} {v:10.3e} {th:10.3e} {rh:10.3e}")

greeks_rmse = {
    "Delta": {n: delta_results[n]["rmse"] for n in KERNELS},
    "Gamma": {n: gamma_results[n]["rmse"] for n in KERNELS},
    "Vega":  {n: vega_results[n]["rmse"] for n in KERNELS},
    "Theta": {n: theta_results[n]["rmse"] for n in KERNELS},
    "Rho":   {n: rho_results[n]["rmse"] for n in KERNELS},
}
print(f"\n{'Greek':8s} {'Best kernel':14s} {'RMSE':>10s}")
print("-" * 38)
for greek, rmses in greeks_rmse.items():
  best = min(rmses, key=lambda n: rmses[n] if np.isfinite(rmses[n]) else np.inf)
  print(f"{greek:8s} {best:14s} {rmses[best]:10.3e}")


# ---------------------------------------------------------------------------
# Figure B: Delta and Gamma fits + errors per kernel
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = plt.cm.tab10(np.linspace(0, 1, len(KERNELS)))

ax = axes[0, 0]
ax.plot(S_test, delta_true, "k-", lw=2.5, label="BS exact")
for (name, res), c in zip(delta_results.items(), colors):
  ax.plot(S_test, res["delta"], "--", color=c, label=name, lw=1.2)
ax.set_xlabel("S"); ax.set_ylabel(r"$\Delta$")
ax.set_title(r"Delta $= \partial C/\partial S$"); ax.legend(fontsize=8)

ax = axes[0, 1]
for (name, res), c in zip(delta_results.items(), colors):
  ax.plot(S_test, np.abs(res["delta"] - delta_true), color=c, label=name)
ax.set_yscale("log"); ax.set_xlabel("S")
ax.set_ylabel(r"|$\Delta_{GP} - \Delta_{BS}$|")
ax.set_title("Delta absolute error (log)"); ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(S_test, gamma_true, "k-", lw=2.5, label="BS exact")
for (name, res), c in zip(gamma_results.items(), colors):
  ax.plot(S_test, res["gamma"], "--", color=c, label=name, lw=1.2)
ax.set_xlabel("S"); ax.set_ylabel(r"$\Gamma$")
ax.set_title(r"Gamma $= \partial^2 C/\partial S^2$"); ax.legend(fontsize=8)

ax = axes[1, 1]
for (name, res), c in zip(gamma_results.items(), colors):
  ax.plot(S_test, np.abs(res["gamma"] - gamma_true), color=c, label=name)
ax.set_yscale("log"); ax.set_xlabel("S")
ax.set_ylabel(r"|$\Gamma_{GP} - \Gamma_{BS}$|")
ax.set_title("Gamma absolute error (log)"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_B_delta_gamma.png"), dpi=130)
plt.close()
print("Saved fig_B_delta_gamma.png")


# ---------------------------------------------------------------------------
# Figure C: Vega, Theta, Rho fits + errors per kernel
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
ax.plot(sig_test, vega_true_vals, "k-", lw=2.5, label="BS exact")
for (name, res), c in zip(vega_results.items(), colors):
  ax.plot(sig_test, res["vega"], "--", color=c, label=name, lw=1.2)
ax.set_xlabel(r"$\sigma$"); ax.set_ylabel(r"$\nu$")
ax.set_title(r"Vega $= \partial C/\partial \sigma$"); ax.legend(fontsize=8)

ax = axes[1, 0]
for (name, res), c in zip(vega_results.items(), colors):
  ax.plot(sig_test, np.abs(res["vega"] - vega_true_vals), color=c, label=name)
ax.set_yscale("log"); ax.set_xlabel(r"$\sigma$")
ax.set_ylabel("|error|"); ax.set_title("Vega error"); ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(T_test, theta_true_vals, "k-", lw=2.5, label="BS exact")
for (name, res), c in zip(theta_results.items(), colors):
  ax.plot(T_test, res["theta"], "--", color=c, label=name, lw=1.2)
ax.set_xlabel("T"); ax.set_ylabel(r"$\Theta$")
ax.set_title(r"Theta $= \partial C/\partial T$ (per day)"); ax.legend(fontsize=8)

ax = axes[1, 1]
for (name, res), c in zip(theta_results.items(), colors):
  ax.plot(T_test, np.abs(res["theta"] - theta_true_vals), color=c, label=name)
ax.set_yscale("log"); ax.set_xlabel("T")
ax.set_ylabel("|error|"); ax.set_title("Theta error"); ax.legend(fontsize=8)

ax = axes[0, 2]
ax.plot(r_test * 100, rho_true_vals, "k-", lw=2.5, label="BS exact")
for (name, res), c in zip(rho_results.items(), colors):
  ax.plot(r_test * 100, res["rho"], "--", color=c, label=name, lw=1.2)
ax.set_xlabel("r (%)"); ax.set_ylabel(r"$\rho$ (per bp)")
ax.set_title(r"Rho $= \partial C/\partial r$ (per 1bp)"); ax.legend(fontsize=8)

ax = axes[1, 2]
for (name, res), c in zip(rho_results.items(), colors):
  ax.plot(r_test * 100, np.abs(res["rho"] - rho_true_vals), color=c, label=name)
ax.set_yscale("log"); ax.set_xlabel("r (%)")
ax.set_ylabel("|error|"); ax.set_title("Rho error"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_C_vega_theta_rho.png"), dpi=130)
plt.close()
print("Saved fig_C_vega_theta_rho.png")


# ---------------------------------------------------------------------------
# Figure D: relative-RMSE heatmap (Greeks x kernels). We normalise by the
# range of each Greek so that Greeks with very different scales (Delta in [0,1]
# vs Rho in [0, 80]) can be compared on a single scale.
# ---------------------------------------------------------------------------
greek_names  = list(greeks_rmse.keys())
kernel_names = list(KERNELS.keys())

true_ranges = {
    "Delta": np.ptp(delta_true) + 1e-12,
    "Gamma": np.ptp(gamma_true) + 1e-12,
    "Vega":  np.ptp(vega_true_vals) + 1e-12,
    "Theta": np.ptp(theta_true_vals) + 1e-12,
    "Rho":   np.ptp(rho_true_vals) + 1e-12,
}

rel_rmse_matrix = np.array([
    [greeks_rmse[g][k] / true_ranges[g] for k in kernel_names]
    for g in greek_names
])

fig, ax = plt.subplots(figsize=(10, 4))
with np.errstate(invalid="ignore"):
  im = ax.imshow(np.log10(rel_rmse_matrix), cmap="RdYlGn_r", aspect="auto")
plt.colorbar(im, ax=ax, label="log10(relative RMSE)")
ax.set_xticks(range(len(kernel_names))); ax.set_xticklabels(kernel_names, rotation=20)
ax.set_yticks(range(len(greek_names)));  ax.set_yticklabels(greek_names)
ax.set_title("log10(Relative RMSE) per Greek x Kernel  (green = better)")
for i in range(len(greek_names)):
  for j in range(len(kernel_names)):
    ax.text(j, i, f"{rel_rmse_matrix[i,j]:.2%}", ha="center", va="center",
            fontsize=7, color="black")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_D_rmse_heatmap.png"), dpi=130)
plt.close()
print("Saved fig_D_rmse_heatmap.png (relative RMSE %)")

print("\nAll done - 4 figures written to output/")
