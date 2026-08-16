"""Black-Scholes closed-form pricer with all Greeks.

We compute the European call/put price under the Black-Scholes model and return
it together with its analytical sensitivities (Delta, Vega, Gamma, Theta, Rho).
This module fills the role of the book's `BlackScholes.bsformula` helper, using
the same call signature and tuple return order so that the Section 4.1 script
runs against the reference values without change.
"""

import numpy as np
import scipy.stats as st


def bsformula(cp_flag, S, K, r, T, sigma, q=0.0):
  '''
    We compute the European option price and all standard Greeks under Black-Scholes.

    Parameters
    ----------
    cp_flag : int
      +1 for a call, -1 for a put.
    S, K, r, T, sigma : float or array-like
      Spot, strike, risk-free rate, time to maturity, volatility.
    q : float
      Continuous dividend yield (default 0).

    Returns
    -------
    tuple of six arrays (or scalars) in the order (price, delta, vega, gamma, theta, rho).
    The order matches the book's Example-2 convention: [0] = price, [1] = delta,
    [2] = vega, so downstream code that indexes into the tuple keeps working.
  '''
  S = np.asarray(S, dtype=float)
  d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  pdf_d1 = st.norm.pdf(d1)
  Nd1 = st.norm.cdf(cp_flag * d1)
  Nd2 = st.norm.cdf(cp_flag * d2)

  disc_r = np.exp(-r * T)
  disc_q = np.exp(-q * T)

  # price
  price = cp_flag * (S * disc_q * Nd1 - K * disc_r * Nd2)

  # first-order Greeks
  delta = cp_flag * disc_q * Nd1
  vega = S * disc_q * np.sqrt(T) * pdf_d1

  # second-order Greek
  gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))

  # theta (per year); the Section 4.1 script divides by 365 for per-day units
  theta = (- S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
           - cp_flag * r * K * disc_r * Nd2
           + cp_flag * q * S * disc_q * Nd1)

  # rho (per unit rate); the Section 4.1 script divides by 100 for per-basis-point
  rho = cp_flag * K * T * disc_r * Nd2

  return price, delta, vega, gamma, theta, rho


if __name__ == "__main__":
  # sanity check: reproduce a hand-known call price
  price, delta, vega, gamma, theta, rho = bsformula(1, 100.0, 100.0, 0.05, 1.0, 0.2)
  print(f"S=100 K=100 r=0.05 T=1 sigma=0.2 -> "
        f"price={price:.6f} delta={delta:.6f} vega={vega:.6f} "
        f"gamma={gamma:.6f} theta={theta:.6f} rho={rho:.6f}")
