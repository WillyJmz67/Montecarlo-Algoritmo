# engine/metrics.py
"""Metricas de riesgo y rendimiento sobre simulaciones."""
import numpy as np


def calculate_var(final_values, initial_value, confidence=0.95) -> float:
    q = np.percentile(final_values, (1 - confidence) * 100)
    return float(initial_value - q)


def calculate_cvar(final_values, initial_value, confidence=0.95) -> float:
    thr = np.percentile(final_values, (1 - confidence) * 100)
    tail = final_values[final_values <= thr]
    if len(tail) == 0:
        return calculate_var(final_values, initial_value, confidence)
    return float(initial_value - tail.mean())


def max_drawdown(portfolio_paths: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(portfolio_paths, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(running_max > 0, (portfolio_paths - running_max) / running_max, 0.0)
    return np.min(dd, axis=1)


def percentile_table(final_values, initial_value):
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(final_values, qs)
    rets = (vals / initial_value - 1.0) * 100
    return qs, vals, rets


def sharpe_sortino(portfolio_paths: np.ndarray, risk_free: float = 0.0, periods: int = 252):
    """Sharpe y Sortino anualizados sobre retornos diarios simulados (media de sims)."""
    rets = portfolio_paths[:, 1:] / np.maximum(portfolio_paths[:, :-1], 1e-12) - 1.0
    mu = rets.mean()
    sigma = rets.std()
    downside = rets[rets < 0].std()
    sharpe = float((mu * periods - risk_free) / (sigma * np.sqrt(periods))) if sigma > 0 else 0.0
    sortino = float((mu * periods - risk_free) / (downside * np.sqrt(periods))) if downside and downside > 0 else 0.0
    return sharpe, sortino


def cagr(initial: float, final_median: float, years: float) -> float:
    if initial <= 0 or years <= 0:
        return 0.0
    return float((final_median / initial) ** (1 / years) - 1.0)
