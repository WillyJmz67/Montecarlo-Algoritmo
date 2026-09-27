# engine/volatility.py
"""Modelos de volatilidad: historica, EWMA, GARCH."""
import numpy as np
import pandas as pd


class VolatilityModel:
    @staticmethod
    def historical(returns: pd.DataFrame) -> pd.Series:
        return returns.std() * np.sqrt(252)

    @staticmethod
    def ewma(returns: pd.DataFrame, span: int = 60) -> pd.Series:
        return returns.ewm(span=span, min_periods=min(span, len(returns))).std().iloc[-1] * np.sqrt(252)

    @staticmethod
    def garch_last(returns: pd.Series, p: int = 1, q: int = 1) -> dict:
        """Ajusta GARCH(1,1) t-student y devuelve vol anualizada actual + forecast."""
        try:
            from arch import arch_model
        except ImportError as e:
            raise ImportError("Falta 'arch'. Instala con: pip install arch") from e
        clean = (returns.dropna() * 100).astype(float)
        if len(clean) < 200:
            # Poco dato: fallback a EWMA simple
            vol_d = float(clean.std() / 100) if len(clean) > 2 else 0.2 / np.sqrt(252)
            return {"vol_annual": float(vol_d * np.sqrt(252)), "params": {}, "fallback": True}
        model = arch_model(clean, vol="Garch", p=p, q=q, dist="t", rescale=False)
        res = model.fit(disp="off")
        # Ultima varianza condicional
        last_var = float(res.conditional_volatility.iloc[-1] ** 2)  # en %^2
        vol_annual = float(np.sqrt(last_var * 252) / 100)
        fc = res.forecast(horizon=5)
        fwd = float(np.sqrt(fc.variance.values[-1, :].mean()) / 100 * np.sqrt(252))
        return {"vol_annual": vol_annual, "forecast_annual": fwd,
                "params": {k: float(v) for k, v in res.params.items()}, "fallback": False}
