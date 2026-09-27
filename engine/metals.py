# engine/metals.py
"""Ajustes especificos de metales y futuros: multiplicador, roll, margen."""
import numpy as np

FUTURES_SPECS = {
    "GC=F":  {"label": "Oro (COMEX)", "multiplier": 100, "margin": 8000, "roll_bps": 2.0},
    "SI=F":  {"label": "Plata (COMEX)", "multiplier": 5000, "margin": 9000, "roll_bps": 3.0},
    "PL=F":  {"label": "Platino (NYMEX)", "multiplier": 50, "margin": 4000, "roll_bps": 3.0},
    "HG=F":  {"label": "Cobre (COMEX)", "multiplier": 25000, "margin": 6000, "roll_bps": 4.0},
    "PA=F":  {"label": "Paladio (NYMEX)", "multiplier": 100, "margin": 7000, "roll_bps": 4.0},
    "XAUUSD=X": {"label": "Oro Spot", "multiplier": 1, "margin": 0, "roll_bps": 0.0},
    "XAGUSD=X": {"label": "Plata Spot", "multiplier": 1, "margin": 0, "roll_bps": 0.0},
}


def apply_futures_roll(paths: np.ndarray, roll_bps: float = 2.0, rolls_per_year: float = 6.0,
                       days: int = 252) -> np.ndarray:
    """Descuento continuo aproximado del coste de roll (contango)."""
    if not roll_bps or roll_bps <= 0:
        return paths
    annual_decay = (1 - roll_bps / 10_000.0) ** rolls_per_year
    daily_decay = annual_decay ** (1 / 252)
    t = np.arange(paths.shape[1])
    factor = daily_decay ** t  # (days+1,)
    return paths * factor[None, :, None] if paths.ndim == 3 else paths * factor[None, :]
