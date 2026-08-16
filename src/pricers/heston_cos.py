"""COS-method Heston pricer (Fang & Oosterlee 2008) with vectorised call and put.

We price European call and put options under the Heston stochastic-volatility
model using the COS method, which recovers the option price from the
characteristic function of the log-price through a Fourier-cosine series
expansion. The characteristic function is written in the "little-trap" form
(Albrecher et al.), the numerically stable branch. Vectorised over spot for
speed; the scalar entry point is kept for readability and validation.
Parameters follow the book's Table 3.1 throughout the report.
"""

import numpy as np
import scipy.stats as st


def bs_call(S, K, r, T, sig):
  '''
    We compute the Black-Scholes call price. Used only as a sanity check: in the
    limit vol-of-vol -> 0 with v0 = theta and rho = 0 the Heston price must
    collapse to Black-Scholes with sigma = sqrt(theta).
  '''
  d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
  d2 = d1 - sig * np.sqrt(T)
  return S * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)


def heston_cf(u, r, T, v0, kappa, theta, sigma, rho):
  '''
    We compute the Heston characteristic function of the log-return
    y = log(S_T / S0), in the little-trap form of Albrecher et al. which avoids
    the discontinuity the original Heston formulation has for long maturities.
    S0 is absorbed into the payoff phase separately, so this function does not
    depend on S0.
  '''
  a = kappa * theta
  # d and g are the standard auxiliary terms of the Heston characteristic function
  d = np.sqrt((rho * sigma * 1j * u - kappa) ** 2 + sigma ** 2 * (1j * u + u ** 2))
  g = (kappa - rho * sigma * 1j * u - d) / (kappa - rho * sigma * 1j * u + d)
  e1 = np.exp(1j * u * r * T)
  e2 = np.exp(a / sigma ** 2 * ((kappa - rho * sigma * 1j * u - d) * T
                                - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
  e3 = np.exp(v0 / sigma ** 2 * (kappa - rho * sigma * 1j * u - d)
              * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
  return e1 * e2 * e3


def _chi(k, a, b, c, d):
  '''
    We compute the chi coefficients of Fang & Oosterlee: the cosine-series
    coefficients of exp(y) on the interval [c, d]. These build the call and put
    payoff coefficients together with _psi.
  '''
  w = k * np.pi / (b - a)
  return (1.0 / (1.0 + w ** 2)) * (
      np.cos(w * (d - a)) * np.exp(d) - np.cos(w * (c - a)) * np.exp(c)
      + w * np.sin(w * (d - a)) * np.exp(d) - w * np.sin(w * (c - a)) * np.exp(c))


def _psi(k, a, b, c, d):
  '''
    We compute the psi coefficients of Fang & Oosterlee: the cosine-series
    coefficients of the constant 1 on [c, d]. The k = 0 term is handled
    separately because the frequency w = 0 there.
  '''
  w = k * np.pi / (b - a)
  out = np.empty_like(w)
  out[0] = d - c
  out[1:] = (np.sin(w[1:] * (d - a)) - np.sin(w[1:] * (c - a))) / w[1:]
  return out


def _truncation_range(r, T, v0, kappa, theta, sigma, rho, L):
  '''
    We build the cosine-series truncation range [a, b] from the first two
    cumulants of the log-returns; L controls how many standard deviations of the
    return we keep. Widening [a, b] is essential when pricing deep-in/out-of-
    the-money options at short maturity, otherwise the cosine series collapses.
  '''
  c1 = r * T + (1 - np.exp(-kappa * T)) * (theta - v0) / (2 * kappa) - 0.5 * theta * T
  c2 = (1.0 / (8 * kappa ** 3)) * (
      sigma * T * kappa * np.exp(-kappa * T) * (v0 - theta) * (8 * kappa * rho - 4 * sigma)
      + kappa * rho * sigma * (1 - np.exp(-kappa * T)) * (16 * theta - 8 * v0)
      + 2 * theta * kappa * T * (-4 * kappa * rho * sigma + sigma ** 2 + 4 * kappa ** 2)
      + sigma ** 2 * ((theta - 2 * v0) * np.exp(-2 * kappa * T)
                      + theta * (6 * np.exp(-kappa * T) - 7) + 2 * v0)
      + 8 * kappa ** 2 * (v0 - theta) * (1 - np.exp(-kappa * T)))
  return c1, c2


def heston_call_cos(S0, K, r, T, v0, kappa, theta, sigma, rho, N=256, L=12):
  '''
    We compute the European call price under Heston via the COS method (scalar
    entry point). We work in log-return units y = log(S_T / S0) and shift the
    payoff so the strike sits at the origin.
  '''
  c1, c2 = _truncation_range(r, T, v0, kappa, theta, sigma, rho, L)
  a = c1 - L * np.sqrt(abs(c2))
  b = c1 + L * np.sqrt(abs(c2))
  k = np.arange(N)
  u = k * np.pi / (b - a)  # frequencies of the cosine expansion
  cf = heston_cf(u, r, T, v0, kappa, theta, sigma, rho)
  # call payoff coefficients: the payoff is nonzero on [0, b] in return units
  Uk = (2.0 / (b - a)) * (_chi(k, a, b, 0.0, b) - _psi(k, a, b, 0.0, b))
  # x = log(S0 / K) shifts the payoff so the strike sits at the origin
  x = np.log(S0 / K)
  Fk = np.real(cf * np.exp(1j * u * (x - a)))
  Fk[0] *= 0.5  # the first cosine term carries a 1/2 weight
  return K * np.exp(-r * T) * np.sum(Fk * Uk)


def heston_call_cos_vec(S0_arr, K, r, T, v0, kappa, theta, sigma, rho, N=256, L=12):
  '''
    We compute the European call price under Heston via COS, vectorised over an
    array of spots. The truncation range [a, b] is widened to cover the log-
    moneyness of every spot at once, so the cosine series stays convergent even
    for deep in/out-of-the-money options at short maturity.
  '''
  c1, c2 = _truncation_range(r, T, v0, kappa, theta, sigma, rho, L)
  x = np.log(np.asarray(S0_arr, dtype=float) / K)
  width = L * np.sqrt(abs(c2) + 1e-4)
  a = min(x.min(), 0.0) + c1 - width
  b = max(x.max(), 0.0) + c1 + width
  k = np.arange(N)
  u = k * np.pi / (b - a)
  cf = heston_cf(u, r, T, v0, kappa, theta, sigma, rho)
  Uk = (2.0 / (b - a)) * (_chi(k, a, b, 0.0, b) - _psi(k, a, b, 0.0, b))
  # phase shifts the payoff by the log-moneyness of each spot at once
  phase = np.exp(1j * np.outer(x - a, u))
  Fk = np.real(cf[None, :] * phase)
  Fk[:, 0] *= 0.5
  return K * np.exp(-r * T) * (Fk * Uk[None, :]).sum(1)


def heston_put_cos_vec(S0_arr, K, r, T, v0, kappa, theta, sigma, rho, N=512, L=14):
  '''
    We compute the European put price under Heston via COS directly (not via
    put-call parity), which stays stable for out-of-the-money puts. We use a
    larger N and L than the call because the put surface is harder to resolve at
    short maturity.
  '''
  c1, c2 = _truncation_range(r, T, v0, kappa, theta, sigma, rho, L)
  x = np.log(np.asarray(S0_arr, dtype=float) / K)
  width = L * np.sqrt(abs(c2) + 1e-4)
  a = min(x.min(), 0.0) + c1 - width
  b = max(x.max(), 0.0) + c1 + width
  k = np.arange(N)
  u = k * np.pi / (b - a)
  cf = heston_cf(u, r, T, v0, kappa, theta, sigma, rho)
  # put payoff coefficients: the payoff is nonzero on [a, 0] in return units
  Uk = (2.0 / (b - a)) * (-_chi(k, a, b, a, 0.0) + _psi(k, a, b, a, 0.0))
  phase = np.exp(1j * np.outer(x - a, u))
  Fk = np.real(cf[None, :] * phase)
  Fk[:, 0] *= 0.5
  return K * np.exp(-r * T) * (Fk * Uk[None, :]).sum(1)


def heston_vec(S_arr, K, r, T, v0, kappa, theta, sigma, rho, **kw):
  '''
    We evaluate the Heston call price over an array of spots. Kept as a
    scalar-in-a-loop wrapper so the older single-spot scripts still work; use
    heston_call_cos_vec when speed matters.
  '''
  return np.array([heston_call_cos(S, K, r, T, v0, kappa, theta, sigma, rho, **kw)
                   for S in S_arr])


if __name__ == "__main__":
  # book Table 3.1 parameters
  kappa, theta, sigma, r, K, T, rho = 0.1, 0.15, 0.1, 0.002, 100.0, 2.0, -0.9
  v0 = 0.1

  # validation 1: vol-of-vol -> 0 with v0 = theta and rho = 0 must collapse to BS(sqrt(theta))
  h = heston_call_cos(100, 100, r, T, theta, kappa, theta, 1e-4, 0.0)
  b = bs_call(100, 100, r, T, np.sqrt(theta))
  print(f"BS-limit check : Heston={h:.6f}  BS={b:.6f}  diff={abs(h-b):.2e}")

  # validation 2: price should be monotone increasing in S with sane magnitudes
  for S in [60, 80, 100, 120, 140]:
    print(f"  S={S:3d}  Heston call = {heston_call_cos(S,K,r,T,v0,kappa,theta,sigma,rho):.4f}")

  # validation 3: scalar and vectorised pricers must agree to machine precision
  S_arr = np.array([60., 80., 100., 120., 140.])
  scalar = np.array([heston_call_cos(S,K,r,T,v0,kappa,theta,sigma,rho) for S in S_arr])
  vector = heston_call_cos_vec(S_arr, K, r, T, v0, kappa, theta, sigma, rho)
  print(f"scalar vs vectorised max diff = {np.max(np.abs(scalar - vector)):.2e}")

  # validation 4: convergence of the cosine series as N grows
  for N in [64, 128, 256, 512]:
    print(f"  N={N:4d}  C(100)={heston_call_cos(100,K,r,T,v0,kappa,theta,sigma,rho,N=N):.8f}")
