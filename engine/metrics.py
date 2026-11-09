# engine/metrics.py
"""Métricas institucionales de riesgo y rendimiento sobre simulaciones Monte Carlo."""
import numpy as np
from scipy.stats import skew, kurtosis


def calculate_var(final_values, initial_value, confidence: float = 0.95) -> float:
    """Value at Risk (VaR) en unidades monetarias al nivel de confianza indicado."""
    q = np.percentile(final_values, (1.0 - confidence) * 100.0)
    return float(max(0.0, initial_value - q))


def calculate_cvar(final_values, initial_value, confidence: float = 0.95) -> float:
    """Conditional Value at Risk (Expected Shortfall / CVaR) en unidades monetarias."""
    thr = np.percentile(final_values, (1.0 - confidence) * 100.0)
    tail = final_values[final_values <= thr]
    if len(tail) == 0:
        return calculate_var(final_values, initial_value, confidence)
    return float(max(0.0, initial_value - tail.mean()))


def drawdown_paths(portfolio_paths: np.ndarray) -> np.ndarray:
    """Devuelve la matriz completa (n_sims, n_steps) de drawdowns en cada instante."""
    running_max = np.maximum.accumulate(portfolio_paths, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(running_max > 0, (portfolio_paths - running_max) / running_max, 0.0)
    return dd


def max_drawdown(portfolio_paths: np.ndarray) -> np.ndarray:
    """Máximo drawdown por trayectoria simulada (valores <= 0)."""
    return np.min(drawdown_paths(portfolio_paths), axis=1)


def percentile_table(final_values, initial_value):
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(final_values, qs)
    rets = (vals / max(initial_value, 1e-12) - 1.0) * 100.0
    return qs, vals, rets


def sharpe_sortino(portfolio_paths: np.ndarray, risk_free: float = 0.0, periods: int = 252):
    """Sharpe y Sortino anualizados sobre retornos diarios simulados.
    Corrige la desviación downside usando el semimomento inferior respecto a rf diario sobre N total.
    """
    rets = portfolio_paths[:, 1:] / np.maximum(portfolio_paths[:, :-1], 1e-12) - 1.0
    rf_daily = risk_free / periods
    excess = rets - rf_daily
    mu_excess = float(np.mean(excess)) * periods
    sigma_ann = float(np.std(rets)) * np.sqrt(periods)

    # Semidesviación inferior correcta sobre la totalidad de observaciones
    downside_diff = np.minimum(0.0, excess)
    downside_dev = float(np.sqrt(np.mean(downside_diff ** 2))) * np.sqrt(periods)

    sharpe = float(mu_excess / sigma_ann) if sigma_ann > 1e-12 else 0.0
    sortino = float(mu_excess / downside_dev) if downside_dev > 1e-12 else 0.0
    return sharpe, sortino


def cagr(initial: float, final_median: float, years: float) -> float:
    if initial <= 0 or final_median <= 0 or years <= 0:
        return 0.0
    return float((final_median / initial) ** (1.0 / years) - 1.0)


def advanced_risk_metrics(portfolio_paths: np.ndarray, base_capital: float,
                          years: float, risk_free: float = 0.0) -> dict:
    """Calcula métricas avanzadas: Calmar, Omega, Ulcer Index, Tail Ratio, Asimetría y Curtosis."""
    final = portfolio_paths[:, -1]
    total_rets = final / max(base_capital, 1e-12) - 1.0
    dd_matrix = drawdown_paths(portfolio_paths)
    mdd = np.min(dd_matrix, axis=1)

    med_final = float(np.median(final))
    cagr_val = cagr(base_capital, med_final, years)
    med_mdd = abs(float(np.median(mdd)))
    calmar = float(cagr_val / med_mdd) if med_mdd > 1e-6 else 0.0

    # Ulcer Index (RMS del drawdown porcentual)
    ulcer_index = float(np.sqrt(np.mean((dd_matrix * 100.0) ** 2)))

    # Omega Ratio respecto al umbral de preservar capital (retorno >= 0)
    gains = np.maximum(0.0, total_rets)
    losses = np.maximum(0.0, -total_rets)
    omega = float(np.sum(gains) / np.sum(losses)) if np.sum(losses) > 1e-12 else 99.9

    # Tail Ratio: |P95 retorno| / |P5 retorno|
    p95_ret = float(np.percentile(total_rets, 95))
    p5_ret = abs(float(np.percentile(total_rets, 5)))
    tail_ratio = float(abs(p95_ret) / p5_ret) if p5_ret > 1e-6 else 1.0

    sk = float(skew(total_rets)) if len(total_rets) > 3 else 0.0
    kt = float(kurtosis(total_rets, fisher=True)) if len(total_rets) > 3 else 0.0

    return {
        "calmar": calmar,
        "omega": omega,
        "ulcer_index": ulcer_index,
        "tail_ratio": tail_ratio,
        "skewness": sk,
        "kurtosis": kt,
    }
