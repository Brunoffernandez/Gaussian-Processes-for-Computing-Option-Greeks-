"""GP from scratch with analytic Greeks - pedagogical companion to Section 4.1.

We implement a Gaussian-process regressor from scratch (no scikit-learn or
GPyTorch for the maths) so the reader can see exactly how the predictive
equations and kernel derivatives wire up. We use the RBF and Matern-5/2
kernels with their analytic derivatives, and recover Delta, Gamma and Vega on
a Black-Scholes call. The canonical Section 4.1 experiment lives in
`src/section_4_1_kernel_comparison.py`; this file exists to make the maths
transparent, not to reproduce the paper's tables.

The GP class here is intentionally minimal: Cholesky factorisation once, L-BFGS-B
hyperparameter optimisation, analytic first and second derivatives of the kernel.
"""

import os
import numpy as np
import scipy.stats as st
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams.update({"figure.figsize": (10, 5), "axes.grid": True})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Black-Scholes closed-form prices and Greeks (ground truth)
# ---------------------------------------------------------------------------
def bs_call(S, K, r, T, sigma):
  '''We compute the Black-Scholes call price.'''
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return S * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)


def bs_delta(S, K, r, T, sigma):
  '''We compute the Black-Scholes Delta.'''
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  return st.norm.cdf(d1)


def bs_gamma(S, K, r, T, sigma):
  '''We compute the Black-Scholes Gamma.'''
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  return st.norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, r, T, sigma):
  '''We compute the Black-Scholes Vega.'''
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  return S * np.sqrt(T) * st.norm.pdf(d1)


# ---------------------------------------------------------------------------
# Kernels and their derivatives (hand-coded)
# ---------------------------------------------------------------------------
def rbf_kernel(X1, X2, ell, sf):
  '''We compute the squared-exponential (RBF) kernel matrix.'''
  sqd = np.sum(X1 ** 2, 1)[:, None] + np.sum(X2 ** 2, 1)[None, :] - 2 * X1 @ X2.T
  return sf ** 2 * np.exp(-0.5 * sqd / ell ** 2)


def rbf_dk_dx(x_test, X_train, ell, sf):
  '''We compute the first derivative of the RBF kernel w.r.t. x_test.'''
  K = rbf_kernel(x_test, X_train, ell, sf)
  diff = x_test - X_train.T
  return -diff / ell ** 2 * K


def rbf_d2k_dx2(x_test, X_train, ell, sf):
  '''We compute the second derivative of the RBF kernel w.r.t. x_test.'''
  K = rbf_kernel(x_test, X_train, ell, sf)
  diff = x_test - X_train.T
  return (diff ** 2 / ell ** 2 - 1.0) / ell ** 2 * K


def matern52_kernel(X1, X2, ell, sf):
  '''We compute the Matern-5/2 kernel (twice differentiable, standard finance choice).'''
  sqd = np.sum(X1 ** 2, 1)[:, None] + np.sum(X2 ** 2, 1)[None, :] - 2 * X1 @ X2.T
  sqd = np.maximum(sqd, 0)
  r = np.sqrt(sqd)
  a = np.sqrt(5) * r / ell
  return sf ** 2 * (1.0 + a + a ** 2 / 3.0) * np.exp(-a)


def matern52_dk_dx(x_test, X_train, ell, sf):
  '''
    We compute the first derivative of Matern-5/2 w.r.t. x_test using the chain
    rule dk/dx = dk/dr * dr/dx. For 1-D, dr/dx = sign(x - x') and
    dk/dr = -sf^2 * (5 / (3 l^2)) * r * (1 + a) * exp(-a), a = sqrt(5) r / l.
  '''
  diff = x_test - X_train.T
  r = np.abs(diff)
  a = np.sqrt(5) * r / ell
  dk_dr = -sf ** 2 * (5.0 / (3 * ell ** 2)) * r * (1.0 + a) * np.exp(-a)
  return dk_dr * np.sign(diff)


def matern52_d2k_dx2(x_test, X_train, ell, sf):
  '''
    We compute the second derivative of Matern-5/2 w.r.t. x_test. Writing
    k(r) = sf^2 * h(a) with h(a) = (1 + a + a^2/3) exp(-a), the second
    derivative is d2k/dx2 = sf^2 * (5/l^2) * h''(a) with
    h''(a) = -(1 + a - a^2)/3 exp(-a).
  '''
  diff = x_test - X_train.T
  r = np.abs(diff)
  a = np.sqrt(5) * r / ell
  return sf ** 2 * (5.0 / (3 * ell ** 2)) * (-(1.0 + a - a ** 2)) * np.exp(-a)


# ---------------------------------------------------------------------------
# Minimal GP class with Cholesky-based prediction and hand-coded derivatives
# ---------------------------------------------------------------------------
class GP:
  '''
    Minimal exact GP regression. Predictive equations follow Rasmussen & Williams:
      E[f*]   = K_*X (K_XX + sigma_n^2 I)^{-1} y
      Var[f*] = K_** - K_*X (K_XX + sigma_n^2 I)^{-1} K_X*
    Hyperparameters are learned by maximising the log marginal likelihood with L-BFGS-B.
  '''

  def __init__(self, kernel="matern52", ell=1.0, sf=1.0, sigma_n=1e-3):
    self.kernel_name = kernel
    self.ell = ell
    self.sf = sf
    self.sigma_n = sigma_n
    if kernel == "matern52":
      self.k = matern52_kernel
      self.dk = matern52_dk_dx
      self.d2k = matern52_d2k_dx2
    elif kernel == "rbf":
      self.k = rbf_kernel
      self.dk = rbf_dk_dx
      self.d2k = rbf_d2k_dx2
    else:
      raise ValueError(kernel)

  def _build(self, X, y):
    '''We build the Cholesky factor once and reuse it for predictions and gradients.'''
    K = self.k(X, X, self.ell, self.sf) + self.sigma_n ** 2 * np.eye(len(X))
    K += 1e-10 * np.eye(len(X))  # tiny jitter for numerical stability
    self.L = la.cholesky(K, lower=True)
    self.alpha = la.cho_solve((self.L, True), y)
    self.X = X
    self.y = y
    return self

  def fit(self, X, y, optimize=True, n_restarts=5):
    '''We optimise the hyperparameters by maximising the log marginal likelihood.'''
    X = np.atleast_2d(X)
    if X.shape[0] != len(y):
      X = X.T
    y = np.asarray(y).ravel()

    if optimize:
      y_scale = np.std(y) + 1e-12

      def neg_lml(theta):
        theta = np.clip(theta, -8, 8)
        ell, sf, sn = np.exp(theta)
        try:
          K = self.k(X, X, ell, sf) + (sn ** 2 + 1e-8) * np.eye(len(X))
          L = la.cholesky(K, lower=True)
          alpha = la.cho_solve((L, True), y)
          val = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(X) * np.log(2 * np.pi)
          if not np.isfinite(val):
            return 1e10
          return val
        except (la.LinAlgError, ValueError, FloatingPointError):
          return 1e10

      bounds = [(-5, 5), (np.log(0.1 * y_scale), np.log(10 * y_scale)), (-8, 2)]
      best = (np.inf, None)
      rng = np.random.default_rng(0)
      for _ in range(n_restarts):
        theta0 = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        try:
          res = minimize(neg_lml, theta0, method="L-BFGS-B", bounds=bounds)
          if np.isfinite(res.fun) and res.fun < best[0]:
            best = (res.fun, res.x)
        except Exception:
          continue
      if best[1] is None:
        best = (0.0, np.array([0.0, np.log(y_scale), -3.0]))
      self.ell, self.sf, self.sigma_n = np.exp(best[1])

    return self._build(X, y)

  def predict(self, Xs, return_std=False):
    '''We predict the posterior mean (and optionally standard deviation) at Xs.'''
    Xs = np.atleast_2d(Xs)
    if Xs.shape[1] != self.X.shape[1]:
      Xs = Xs.T
    Ks = self.k(Xs, self.X, self.ell, self.sf)
    mu = Ks @ self.alpha
    if not return_std:
      return mu
    v = la.cho_solve((self.L, True), Ks.T)
    Kss = self.k(Xs, Xs, self.ell, self.sf)
    var = np.diag(Kss) - np.einsum("ij,ji->i", Ks, v)
    var = np.maximum(var, 0)
    return mu, np.sqrt(var)

  def first_derivative(self, Xs):
    '''
      We differentiate the posterior mean analytically:
      mu(x*) = sum_i alpha_i k(x_i, x*), so d mu / dx* = sum_i alpha_i dk/dx*.
    '''
    Xs = np.atleast_2d(Xs)
    if Xs.shape[1] != self.X.shape[1]:
      Xs = Xs.T
    dKs = self.dk(Xs, self.X, self.ell, self.sf)
    return dKs @ self.alpha

  def second_derivative(self, Xs):
    '''We compute the second derivative of the posterior mean.'''
    Xs = np.atleast_2d(Xs)
    if Xs.shape[1] != self.X.shape[1]:
      Xs = Xs.T
    d2Ks = self.d2k(Xs, self.X, self.ell, self.sf)
    return d2Ks @ self.alpha


# ---------------------------------------------------------------------------
# Demo: fit the GP on a Black-Scholes call, recover Delta and Gamma
# ---------------------------------------------------------------------------
K_strike = 130.0
r        = 0.002
T        = 2.0
sigma    = 0.4
S0       = 100.0

lb, ub = 0.0, 300.0
n_train, n_test = 100, 200

x_train_u = np.linspace(0.01, 1.2, n_train)
x_test_u  = np.linspace(0.01, 1.0, n_test)
S_train = lb + (ub - lb) * x_train_u
S_test  = lb + (ub - lb) * x_test_u
y_train = bs_call(S_train, K_strike, r, T, sigma)

gp_m = GP(kernel="matern52").fit(x_train_u.reshape(-1, 1), y_train)
print(f"Matern-5/2  ell={gp_m.ell:.4f}  sf={gp_m.sf:.4f}  sigma_n={gp_m.sigma_n:.6f}")

gp_r = GP(kernel="rbf").fit(x_train_u.reshape(-1, 1), y_train)
print(f"RBF         ell={gp_r.ell:.4f}  sf={gp_r.sf:.4f}  sigma_n={gp_r.sigma_n:.6f}")

mu_m, std_m = gp_m.predict(x_test_u.reshape(-1, 1), return_std=True)
mu_r, std_r = gp_r.predict(x_test_u.reshape(-1, 1), return_std=True)

C_true = bs_call(S_test, K_strike, r, T, sigma)
delta_true = bs_delta(S_test, K_strike, r, T, sigma)
gamma_true = bs_gamma(S_test, K_strike, r, T, sigma)

# chain rule: d/dS = (1/(ub - lb)) d/dx_u
delta_gp_m = gp_m.first_derivative(x_test_u.reshape(-1, 1)) / (ub - lb)
delta_gp_r = gp_r.first_derivative(x_test_u.reshape(-1, 1)) / (ub - lb)
gamma_gp_m = gp_m.second_derivative(x_test_u.reshape(-1, 1)) / (ub - lb) ** 2
gamma_gp_r = gp_r.second_derivative(x_test_u.reshape(-1, 1)) / (ub - lb) ** 2

# finite-difference Delta on the GP mean for comparison
h = 1e-3
mu_p = gp_m.predict(((x_test_u + h)).reshape(-1, 1))
mu_n = gp_m.predict(((x_test_u - h)).reshape(-1, 1))
delta_fd = (mu_p - mu_n) / (2 * h) / (ub - lb)
gamma_fd = (mu_p - 2 * mu_m + mu_n) / h ** 2 / (ub - lb) ** 2

# figure 1: price + uncertainty band, and Delta comparison
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(S_test, C_true, "k-", lw=2, label="BS exact")
ax[0].plot(S_test, mu_m, "r--", label="GP Matern-5/2")
ax[0].fill_between(S_test, mu_m - 2 * std_m, mu_m + 2 * std_m, alpha=0.2, color="red")
ax[0].scatter(S_train, y_train, c="k", marker="+", s=20, alpha=0.5, label="train")
ax[0].set_xlabel("S"); ax[0].set_ylabel("Call price"); ax[0].legend()
ax[0].set_title(f"Call price surrogate (n_train={n_train})")

ax[1].plot(S_test, delta_true, "k-", lw=2, label="BS exact")
ax[1].plot(S_test, delta_gp_m, "r--", label="GP analytic (Matern)")
ax[1].plot(S_test, delta_gp_r, "b:", label="GP analytic (RBF)")
ax[1].plot(S_test, delta_fd, "g-.", lw=1, label="GP finite diff")
ax[1].set_xlabel("S"); ax[1].set_ylabel(r"$\Delta$"); ax[1].legend()
ax[1].set_title(r"Delta = $\partial C / \partial S$")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "demo_fig1_price_and_delta.png"), dpi=130)
plt.close()

# figure 2: Gamma comparison and Delta error
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(S_test, gamma_true, "k-", lw=2, label="BS exact")
ax[0].plot(S_test, gamma_gp_m, "r--", label="GP analytic (Matern)")
ax[0].plot(S_test, gamma_gp_r, "b:", label="GP analytic (RBF)")
ax[0].plot(S_test, gamma_fd, "g-.", label="GP finite diff")
ax[0].set_xlabel("S"); ax[0].set_ylabel(r"$\Gamma$"); ax[0].legend()
ax[0].set_title(r"Gamma = $\partial^2 C / \partial S^2$")

ax[1].plot(S_test, np.abs(delta_gp_m - delta_true), "r-", label="Matern analytic")
ax[1].plot(S_test, np.abs(delta_gp_r - delta_true), "b-", label="RBF analytic")
ax[1].plot(S_test, np.abs(delta_fd - delta_true), "g-", label="finite diff")
ax[1].set_yscale("log")
ax[1].set_xlabel("S"); ax[1].set_ylabel(r"|$\Delta_{GP} - \Delta_{BS}$|"); ax[1].legend()
ax[1].set_title("Delta absolute error (log scale)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "demo_fig2_gamma_and_errors.png"), dpi=130)
plt.close()

# Vega via the same machinery (input is sigma instead of S)
vol_lb, vol_ub = 0.05, 1.0
vol_train = np.linspace(0.0, 1.0, n_train)
vol_test  = np.linspace(0.01, 0.99, n_test)
sig_train = vol_lb + (vol_ub - vol_lb) * vol_train
sig_test  = vol_lb + (vol_ub - vol_lb) * vol_test
y_train_vol = bs_call(S0, K_strike, r, T, sig_train)

gp_vol = GP(kernel="matern52").fit(vol_train.reshape(-1, 1), y_train_vol)
mu_vol, std_vol = gp_vol.predict(vol_test.reshape(-1, 1), return_std=True)
vega_gp = gp_vol.first_derivative(vol_test.reshape(-1, 1)) / (vol_ub - vol_lb)
vega_true = bs_vega(S0, K_strike, r, T, sig_test)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(sig_test, vega_true, "k-", lw=2, label="BS exact")
ax[0].plot(sig_test, vega_gp, "r--", label="GP analytic")
ax[0].set_xlabel(r"$\sigma$"); ax[0].set_ylabel(r"$\nu$"); ax[0].legend()
ax[0].set_title(r"Vega = $\partial C / \partial \sigma$")

ax[1].plot(sig_test, np.abs(vega_gp - vega_true), "r-")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"$\sigma$"); ax[1].set_ylabel("|GP - BS|")
ax[1].set_title("Vega absolute error (log scale)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "demo_fig3_vega.png"), dpi=130)
plt.close()

# summary
print("\n========== Summary ==========")
print(f"Training points: {n_train}, test points: {n_test}")
print(f"BS parameters: K={K_strike}, r={r}, T={T}, sigma={sigma}")
print()
print("Greek                Method           Max abs error    Mean abs error")
print("-" * 75)
print(f"Delta                Matern analytic  {np.max(np.abs(delta_gp_m - delta_true)):.6e}    {np.mean(np.abs(delta_gp_m - delta_true)):.6e}")
print(f"Delta                RBF analytic     {np.max(np.abs(delta_gp_r - delta_true)):.6e}    {np.mean(np.abs(delta_gp_r - delta_true)):.6e}")
print(f"Delta                FD on GP         {np.max(np.abs(delta_fd - delta_true)):.6e}    {np.mean(np.abs(delta_fd - delta_true)):.6e}")
print(f"Gamma                Matern analytic  {np.max(np.abs(gamma_gp_m - gamma_true)):.6e}    {np.mean(np.abs(gamma_gp_m - gamma_true)):.6e}")
print(f"Gamma                RBF analytic     {np.max(np.abs(gamma_gp_r - gamma_true)):.6e}    {np.mean(np.abs(gamma_gp_r - gamma_true)):.6e}")
print(f"Vega                 Matern analytic  {np.max(np.abs(vega_gp - vega_true)):.6e}    {np.mean(np.abs(vega_gp - vega_true)):.6e}")

print("\nFigures written to output/")
