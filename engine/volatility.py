# engine/volatility.py
"""Modelos de volatilidad: Histórica, EWMA y GARCH(1,1) (con motor MLE nativo SciPy y soporte arch)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _fit_garch11_scipy(returns_pct: np.ndarray) -> dict:
    """Ajusta un modelo GARCH(1,1) por Máxima Verosimilitud (MLE) usando SciPy.
    Ecuación de varianza: h_t = omega + alpha * e_{t-1}^2 + beta * h_{t-1}
    """
    eps = returns_pct - np.mean(returns_pct)
    var_uncond = float(np.var(eps))
    if var_uncond <= 1e-12:
        return {"vol_annual": 0.0, "forecast_annual": 0.0, "params": {}, "fallback": True}

    def neg_loglik(params):
        omega, alpha, beta = params
        if omega <= 1e-8 or alpha < 0.0 or beta < 0.0 or (alpha + beta) >= 0.999:
            return 1e12
        n = len(eps)
        h = np.empty(n, dtype=float)
        h[0] = var_uncond
        e2 = eps ** 2
        for t in range(1, n):
            h[t] = omega + alpha * e2[t - 1] + beta * h[t - 1]
        h = np.maximum(h, 1e-8)
        return 0.5 * np.sum(np.log(h) + e2 / h)

    init_params = [0.05 * var_uncond, 0.09, 0.86]
    bounds = [(1e-7, 2.0 * var_uncond), (0.001, 0.50), (0.40, 0.995)]
    res = minimize(neg_loglik, init_params, method="L-BFGS-B", bounds=bounds)
    omega, alpha, beta = res.x if res.success else init_params
    if alpha + beta >= 0.999:
        beta = max(0.0, 0.995 - alpha)

    h = np.empty(len(eps), dtype=float)
    h[0] = var_uncond
    e2 = eps ** 2
    for t in range(1, len(eps)):
        h[t] = omega + alpha * e2[t - 1] + beta * h[t - 1]

    last_var = float(h[-1])
    next_var = float(omega + alpha * e2[-1] + beta * last_var)
    vol_annual = float(np.sqrt(last_var * 252.0) / 100.0)
    fwd_annual = float(np.sqrt(next_var * 252.0) / 100.0)
    return {
        "vol_annual": vol_annual,
        "forecast_annual": fwd_annual,
        "params": {"omega": float(omega), "alpha": float(alpha), "beta": float(beta), "persistence": float(alpha + beta)},
        "engine": "scipy-mle",
        "fallback": False,
    }


class VolatilityModel:
    @staticmethod
    def historical(returns: pd.DataFrame) -> pd.Series:
        return returns.std() * np.sqrt(252)

    @staticmethod
    def ewma(returns: pd.DataFrame, span: int = 60) -> pd.Series:
        return returns.ewm(span=span, min_periods=min(span, len(returns))).std().iloc[-1] * np.sqrt(252)

    @staticmethod
    def ewma_series(returns: pd.DataFrame, span: int = 60) -> pd.DataFrame:
        """Devuelve la serie temporal completa de volatilidad anualizada EWMA."""
        return returns.ewm(span=span, min_periods=min(20, len(returns))).std() * np.sqrt(252)

    @staticmethod
    def garch_last(returns: pd.Series, p: int = 1, q: int = 1) -> dict:
        """Ajusta GARCH(1,1) y devuelve volatilidad anualizada condicional actual + forecast.
        Usa paquete `arch` si está disponible; de lo contrario emplea el optimizador MLE nativo en SciPy.
        """
        clean = (returns.dropna() * 100.0).astype(float)
        if len(clean) < 100:
            vol_d = float(clean.std() / 100.0) if len(clean) > 2 else 0.20 / np.sqrt(252.0)
            return {"vol_annual": float(vol_d * np.sqrt(252.0)), "forecast_annual": float(vol_d * np.sqrt(252.0)),
                    "params": {}, "fallback": True}

        try:
            from arch import arch_model
            model = arch_model(clean, vol="Garch", p=p, q=q, dist="t", rescale=False)
            res = model.fit(disp="off")
            last_var = float(res.conditional_volatility.iloc[-1] ** 2)
            vol_annual = float(np.sqrt(last_var * 252.0) / 100.0)
            fc = res.forecast(horizon=5)
            fwd = float(np.sqrt(fc.variance.values[-1, :].mean()) / 100.0 * np.sqrt(252.0))
            return {
                "vol_annual": vol_annual,
                "forecast_annual": fwd,
                "params": {k: float(v) for k, v in res.params.items()},
                "engine": "arch-t",
                "fallback": False,
            }
        except Exception:
            return _fit_garch11_scipy(clean.values)
