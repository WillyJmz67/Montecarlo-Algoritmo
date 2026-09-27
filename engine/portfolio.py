# engine/portfolio.py
"""Construccion de cartera, rebalanceo y costes genericos."""
import numpy as np
import pandas as pd


def portfolio_paths_from_asset_paths(asset_paths: np.ndarray, weights: np.ndarray,
                                     rebalance: str = "none") -> np.ndarray:
    """Agrega trayectorias de activos a valor de cartera.

    - none: buy & hold (los pesos derivan).
    - monthly/weekly/daily: rebalanceo a pesos objetivo.
    """
    n_sims, n_steps, n_assets = asset_paths.shape
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() != 0 else np.ones(n_assets) / n_assets

    if rebalance == "none":
        # Normalizar por S0: cada activo aporta w_i * (P_t / P_0)
        s0 = asset_paths[:, :1, :]
        rel = asset_paths / np.maximum(s0, 1e-12)
        return (rel * w[None, None, :]).sum(axis=2)

    freq = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63}.get(rebalance, 21)
    # Rebalanceo discreto: reiniciar base en cada fecha de rebalanceo
    out = np.empty((n_sims, n_steps))
    out[:, 0] = 1.0
    # valor acumulado
    current_value = np.ones(n_sims)
    last_reset_prices = asset_paths[:, 0, :].copy()
    next_rb = freq
    for t in range(1, n_steps):
        rel = asset_paths[:, t, :] / np.maximum(last_reset_prices, 1e-12)
        out[:, t] = current_value * (rel * w[None, :]).sum(axis=1)
        if t == next_rb and t < n_steps - 1:
            current_value = out[:, t].copy()
            last_reset_prices = asset_paths[:, t, :].copy()
            next_rb += freq
    return out


def apply_generic_costs(portfolio_paths: np.ndarray, commission_bps: float = 0.0,
                        slippage_bps: float = 0.0, n_rebalances: int = 0) -> np.ndarray:
    total_bps = commission_bps + slippage_bps
    if total_bps <= 0:
        return portfolio_paths
    # Coste por rebalanceo aplicado de forma compuesta aprox.
    per_event = total_bps / 10_000.0
    # Al menos 1 evento (entrada) + rebalanceos
    events = max(1, n_rebalances + 1)
    factor = (1 - per_event) ** events
    # Aplicar progresivamente
    n_steps = portfolio_paths.shape[1]
    ramp = np.linspace(1.0, factor, n_steps)[None, :]
    return portfolio_paths * ramp


def apply_contributions(portfolio_paths: np.ndarray, monthly_contrib: float = 0.0,
                        initial_value: float = 1.0) -> np.ndarray:
    if not monthly_contrib:
        return portfolio_paths
    contrib_rel = monthly_contrib / max(initial_value, 1e-12)
    out = portfolio_paths.copy()
    # Aportacion cada 21 dias aprox.
    for t in range(21, out.shape[1], 21):
        out[:, t:] += contrib_rel * ((out.shape[1] - t) / out.shape[1] * 0 + 1)
    return out
