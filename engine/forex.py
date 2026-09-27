# engine/forex.py
"""Ajustes especificos de Forex: spread, swap, apalancamiento, conversion."""
import numpy as np

PIP_SPECS = {
    "EURUSD=X": {"pip": 0.0001, "label": "Euro / Dolar EE.UU."},
    "GBPUSD=X": {"pip": 0.0001, "label": "Libra / Dolar EE.UU."},
    "AUDUSD=X": {"pip": 0.0001, "label": "Dolar australiano / USD"},
    "NZDUSD=X": {"pip": 0.0001, "label": "Dolar neozelandes / USD"},
    "USDCAD=X": {"pip": 0.0001, "label": "USD / Dolar canadiense"},
    "USDCHF=X": {"pip": 0.0001, "label": "USD / Franco suizo"},
    "USDJPY=X": {"pip": 0.01, "label": "USD / Yen japones"},
    "EURJPY=X": {"pip": 0.01, "label": "Euro / Yen"},
    "GBPJPY=X": {"pip": 0.01, "label": "Libra / Yen"},
    "EURGBP=X": {"pip": 0.0001, "label": "Euro / Libra"},
}

DEFAULT_LOT = 100_000


def pip_size(symbol: str) -> float:
    return PIP_SPECS.get(symbol, {}).get("pip", 0.0001)


def apply_forex_costs(portfolio_paths: np.ndarray, spread_bps: float = 1.0,
                      swap_bps_dia: float = 0.0, days: int = 0, leverage: float = 1.0) -> np.ndarray:
    """Aplica spread una sola vez + swap acumulado + escala por apalancamiento.

    leverage: 1 = sin apalancar. >1 amplifica desviaciones respecto al inicial.
    """
    adj = portfolio_paths * (1.0 - spread_bps / 10_000.0)
    if swap_bps_dia and days:
        # coste/ingreso lineal diario sobre el valor
        t = np.arange(adj.shape[1])[None, :]  # (1, days+1)
        adj = adj * (1.0 - (swap_bps_dia / 10_000.0) * t)
    if leverage and leverage != 1.0:
        init = adj[:, :1]
        adj = init + (adj - init) * leverage
    return adj
