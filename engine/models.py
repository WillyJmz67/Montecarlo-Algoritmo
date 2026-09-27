# engine/models.py
"""Motor Monte Carlo multiactivo: GBM, Student-t, Bootstrap, Block Bootstrap + antithetic."""
import numpy as np
import pandas as pd
from typing import Optional


def _nearest_psd_cholesky(corr: np.ndarray) -> np.ndarray:
    n = corr.shape[0]
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Regularizacion diagonal progresiva
        for eps in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
            try:
                return np.linalg.cholesky(corr + np.eye(n) * eps)
            except np.linalg.LinAlgError:
                continue
        # Ultimo recurso: identidad (activos independientes)
        return np.eye(n)


def shrink_correlation(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage sobre covarianza -> correlacion. Fallback a historica."""
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(returns.values)
        cov = lw.covariance_
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        return np.clip(corr, -1, 1)
    except Exception:
        return returns.corr().values


class MonteCarloEngine:
    """Motor multiactivo reutilizable (Streamlit / FastAPI)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    # ---------- GBM normal ----------
    def simulate_gbm(self, S0, mu, sigma, corr_matrix, days, n_sims, dt=1/252,
                     antithetic: bool = False) -> np.ndarray:
        S0 = np.asarray(S0, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        n_assets = len(S0)
        L = _nearest_psd_cholesky(np.asarray(corr_matrix, dtype=float))
        drift = (mu - 0.5 * sigma ** 2) * dt
        vol_step = sigma * np.sqrt(dt)

        if antithetic and n_sims % 2 == 1:
            n_sims += 1  # par para pares Z/-Z

        Z = self.rng.standard_normal((n_sims, days, n_assets))
        if antithetic:
            half = n_sims // 2
            Z[half:] = -Z[:half]
        corr_Z = Z @ L.T

        paths = np.empty((n_sims, days + 1, n_assets))
        paths[:, 0, :] = S0
        for t in range(1, days + 1):
            paths[:, t, :] = paths[:, t - 1, :] * np.exp(drift + vol_step * corr_Z[:, t - 1, :])
        # Evitar colapsos numericos
        np.nan_to_num(paths, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        paths[paths < 0] = 0.0
        return paths

    # ---------- Student-t ----------
    def simulate_student_t(self, S0, mu, sigma, corr_matrix, days, n_sims,
                           df: float = 5.0, dt=1/252, antithetic: bool = False) -> np.ndarray:
        S0 = np.asarray(S0, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        n_assets = len(S0)
        L = _nearest_psd_cholesky(np.asarray(corr_matrix, dtype=float))
        drift = (mu - 0.5 * sigma ** 2) * dt
        vol_step = sigma * np.sqrt(dt)

        Z = self.rng.standard_normal((n_sims, days, n_assets))
        if antithetic and n_sims >= 2:
            half = n_sims // 2
            Z[half:half*2] = -Z[:half]
        corr_Z = Z @ L.T
        # Escala chi2 compartida por dia/sim para preservar dependencia de cola
        chi2 = self.rng.chisquare(max(df, 2.1), size=(n_sims, days, 1))
        t_shocks = corr_Z / np.sqrt(chi2 / df)

        paths = np.empty((n_sims, days + 1, n_assets))
        paths[:, 0, :] = S0
        for t in range(1, days + 1):
            paths[:, t, :] = paths[:, t - 1, :] * np.exp(drift + vol_step * t_shocks[:, t - 1, :])
        np.nan_to_num(paths, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        paths[paths < 0] = 0.0
        return paths

    # ---------- Bootstrap historico (vectorizado) ----------
    def simulate_historical_bootstrap(self, S0, historical_returns: pd.DataFrame,
                                      days, n_sims) -> np.ndarray:
        S0 = np.asarray(S0, dtype=float)
        arr = historical_returns.values.astype(float)
        arr = arr[~np.isnan(arr).any(axis=1)]
        n_hist = len(arr)
        if n_hist == 0:
            raise ValueError("Sin retornos historicos validos.")
        idx = self.rng.integers(0, n_hist, size=(n_sims, days))
        sampled = arr[idx]  # (n_sims, days, n_assets)
        paths = np.empty((n_sims, days + 1, len(S0)))
        paths[:, 0, :] = S0
        # Cumprod vectorizado por sim
        growth = np.cumprod(1.0 + sampled, axis=1)
        paths[:, 1:, :] = S0[None, None, :] * growth
        paths[paths < 0] = 0.0
        return paths

    # ---------- Block bootstrap ----------
    def simulate_block_bootstrap(self, S0, historical_returns: pd.DataFrame,
                                 days, n_sims, block_size: int = 20) -> np.ndarray:
        S0 = np.asarray(S0, dtype=float)
        df = historical_returns.dropna()
        arr = df.values.astype(float)
        n_hist, n_assets = arr.shape
        if n_hist < block_size:
            block_size = max(1, n_hist // 2)
        paths = np.empty((n_sims, days + 1, n_assets))
        paths[:, 0, :] = S0
        n_blocks = int(np.ceil(days / block_size))
        for s in range(n_sims):
            levels = [S0.copy()]
            cur = S0.copy()
            pos = 0
            for _ in range(n_blocks):
                start = int(self.rng.integers(0, max(1, n_hist - block_size + 1)))
                block = arr[start:start + block_size]
                take = min(len(block), days - pos)
                if take <= 0:
                    break
                cur = cur * np.cumprod(1.0 + block[:take], axis=0)[-1]
                # reconstruir paso a paso para guardar trayectoria
                base = levels[-1] if len(levels) == 1 else paths[s, pos, :]
                step_vals = base * np.cumprod(1.0 + block[:take], axis=0)
                paths[s, pos + 1:pos + 1 + take, :] = step_vals
                pos += take
            # seguridad: rellenar si quedo corto
            if pos < days:
                paths[s, pos + 1:, :] = paths[s, pos, :]
        paths[paths < 0] = 0.0
        return paths
