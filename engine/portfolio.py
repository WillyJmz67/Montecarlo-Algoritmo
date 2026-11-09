# engine/portfolio.py
"""Construcción de cartera, rebalanceo, aportaciones periódicas compuestas y optimización cuántica de pesos."""
import numpy as np
from scipy.optimize import minimize


def portfolio_paths_from_asset_paths(asset_paths: np.ndarray, weights: np.ndarray,
                                     rebalance: str = "none") -> np.ndarray:
    """Agrega trayectorias de activos a valor de cartera normalizado (base = 1.0).

    - none: buy & hold (los pesos derivan con el mercado).
    - monthly/quarterly/weekly/daily: rebalanceo periódico a los pesos objetivo.
    """
    n_sims, n_steps, n_assets = asset_paths.shape
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones(n_assets) / n_assets

    if rebalance == "none":
        s0 = np.maximum(asset_paths[:, :1, :], 1e-12)
        rel = asset_paths / s0
        return (rel * w[None, None, :]).sum(axis=2)

    freq = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63}.get(rebalance, 21)
    out = np.empty((n_sims, n_steps), dtype=float)
    out[:, 0] = 1.0
    current_value = np.ones(n_sims, dtype=float)
    last_reset_prices = np.maximum(asset_paths[:, 0, :].copy(), 1e-12)
    next_rb = freq

    for t in range(1, n_steps):
        rel = asset_paths[:, t, :] / last_reset_prices
        out[:, t] = current_value * (rel * w[None, :]).sum(axis=1)
        if t == next_rb and t < n_steps - 1:
            current_value = out[:, t].copy()
            last_reset_prices = np.maximum(asset_paths[:, t, :].copy(), 1e-12)
            next_rb += freq
    return out


def apply_generic_costs(portfolio_paths: np.ndarray, commission_bps: float = 0.0,
                        slippage_bps: float = 0.0, n_rebalances: int = 0,
                        turnover_ratio: float = 0.15) -> np.ndarray:
    """Aplica costes de entrada (100% nocional) + costes por rotación en cada rebalanceo."""
    total_bps = float(commission_bps) + float(slippage_bps)
    if total_bps <= 0:
        return portfolio_paths
    per_event = total_bps / 10_000.0
    # Entrada inicial (100% de la cartera) + rebalanceos sobre la fracción rotada (turnover)
    entry_factor = 1.0 - per_event
    reb_factor = (1.0 - per_event * turnover_ratio) ** max(0, int(n_rebalances))
    total_factor = entry_factor * reb_factor
    n_steps = portfolio_paths.shape[1]
    ramp = np.linspace(entry_factor, total_factor, n_steps)[None, :]
    return portfolio_paths * ramp


def apply_contributions(portfolio_paths: np.ndarray, monthly_contrib: float = 0.0,
                        initial_value: float = 1.0) -> np.ndarray:
    """Aplica aportaciones periódicas cada 21 días hábiles compuestas con el retorno de la cartera."""
    if not monthly_contrib or monthly_contrib <= 0:
        return portfolio_paths

    out = portfolio_paths.copy()
    n_steps = out.shape[1]
    # Detectar si portfolio_paths ya está escalado en moneda (~initial_value) o normalizado (~1.0)
    start_mean = float(np.mean(out[:, 0]))
    is_normalized = abs(start_mean - 1.0) < 1e-2 and initial_value > 2.0
    contrib_amount = (monthly_contrib / max(initial_value, 1e-12)) if is_normalized else float(monthly_contrib)

    # Cada aportación realizada en el día t se invierte en la cartera y capitaliza desde t hasta el final
    base_paths = np.maximum(portfolio_paths, 1e-12)
    for t in range(21, n_steps, 21):
        growth_from_t = base_paths[:, t:] / base_paths[:, t:t + 1]
        out[:, t:] += contrib_amount * growth_from_t
    return out


def total_invested_capital(n_steps: int, initial_capital: float, monthly_contrib: float = 0.0) -> float:
    """Calcula el capital total desembolsado (inicial + suma de aportaciones mensuales)."""
    if not monthly_contrib or monthly_contrib <= 0:
        return float(initial_capital)
    n_contribs = len(range(21, int(n_steps), 21))
    return float(initial_capital + n_contribs * monthly_contrib)


# ============ OPTIMIZACIÓN CUÁNTICA DE CARTERAS ============

def covariance_from_corr(sigma_ann: np.ndarray, corr_matrix: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma_ann, dtype=float)
    corr = np.asarray(corr_matrix, dtype=float)
    return np.outer(sigma, sigma) * corr


def portfolio_stats(weights: np.ndarray, mu_ann: np.ndarray, cov_ann: np.ndarray,
                    risk_free: float = 0.0) -> dict:
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w) if np.sum(w) > 0 else np.ones_like(w) / len(w)
    ret = float(np.dot(w, mu_ann))
    vol = float(np.sqrt(max(np.dot(w.T, np.dot(cov_ann, w)), 1e-12)))
    sharpe = (ret - risk_free) / vol if vol > 0 else 0.0
    # Contribución marginal y porcentual al riesgo (Euler)
    mrc = np.dot(cov_ann, w) / vol if vol > 0 else np.zeros_like(w)
    rc = w * mrc
    rc_pct = rc / vol if vol > 0 else np.ones_like(w) / len(w)
    return {"return": ret, "volatility": vol, "sharpe": sharpe, "risk_contrib_pct": rc_pct}


def optimize_weights(mu_ann: np.ndarray, sigma_ann: np.ndarray, corr_matrix: np.ndarray,
                     risk_free: float = 0.0, method: str = "max_sharpe") -> np.ndarray:
    """Calcula pesos óptimos long-only (suma = 1, 0 <= w_i <= 1).
    Métodos soportados: 'equal', 'min_vol', 'max_sharpe', 'risk_parity'.
    """
    n = len(mu_ann)
    if n <= 1 or method == "equal":
        return np.ones(n, dtype=float) / max(1, n)

    cov = covariance_from_corr(sigma_ann, corr_matrix)
    w0 = np.ones(n, dtype=float) / n
    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if method == "min_vol":
        obj = lambda w: float(np.sqrt(max(np.dot(w.T, np.dot(cov, w)), 1e-12)))
    elif method == "risk_parity":
        def obj(w):
            w_eff = np.maximum(w, 1e-8)
            vol = np.sqrt(max(np.dot(w_eff.T, np.dot(cov, w_eff)), 1e-12))
            rc = w_eff * (np.dot(cov, w_eff) / vol)
            target = vol / n
            return float(np.sum((rc - target) ** 2))
    else:  # max_sharpe
        def obj(w):
            ret = np.dot(w, mu_ann)
            vol = np.sqrt(max(np.dot(w.T, np.dot(cov, w)), 1e-12))
            return -float((ret - risk_free) / vol)

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success or np.sum(res.x) <= 0:
        return w0
    w_opt = np.clip(res.x, 0.0, 1.0)
    return w_opt / np.sum(w_opt)


def efficient_frontier_curve(mu_ann: np.ndarray, sigma_ann: np.ndarray,
                             corr_matrix: np.ndarray, n_points: int = 35) -> tuple[np.ndarray, np.ndarray]:
    """Genera la curva de Frontera Eficiente (volatilidades, retornos)."""
    n = len(mu_ann)
    cov = covariance_from_corr(sigma_ann, corr_matrix)
    if n <= 1:
        return np.asarray(sigma_ann, dtype=float), np.asarray(mu_ann, dtype=float)

    w_min = optimize_weights(mu_ann, sigma_ann, corr_matrix, method="min_vol")
    r_min = float(np.dot(w_min, mu_ann))
    r_max = float(np.max(mu_ann))
    if r_max - r_min < 1e-5:
        vol_min = float(np.sqrt(max(np.dot(w_min.T, np.dot(cov, w_min)), 1e-12)))
        return np.array([vol_min]), np.array([r_min])

    target_rets = np.linspace(r_min, r_max * 0.995, n_points)
    vols, rets = [], []
    w0 = w_min.copy()
    bounds = [(0.0, 1.0) for _ in range(n)]

    for tr in target_rets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, target=tr: np.dot(w, mu_ann) - target},
        ]
        res = minimize(lambda w: float(np.dot(w.T, np.dot(cov, w))), w0,
                       method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            w_k = np.clip(res.x, 0.0, 1.0)
            w_k /= np.sum(w_k)
            vols.append(float(np.sqrt(max(np.dot(w_k.T, np.dot(cov, w_k)), 1e-12))))
            rets.append(float(np.dot(w_k, mu_ann)))
            w0 = w_k
    return np.array(vols), np.array(rets)
