# engine/models.py
"""Motor Monte Carlo multiactivo vectorizado: GBM, Student-t, Jump-Diffusion (Merton),
Bootstrap histórico y Block Bootstrap + Antithetic Variates y Quasi-Monte Carlo (Sobol)."""
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc


def _nearest_psd_cholesky(corr: np.ndarray) -> np.ndarray:
    """Obtiene el factor de Cholesky de una matriz de correlación, proyectando a PSD si es necesario."""
    corr = np.asarray(corr, dtype=float)
    n = corr.shape[0]
    if n == 1:
        return np.ones((1, 1), dtype=float)
    # Asegurar simetría y diagonal unitaria
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Proyección espectral (recorte de autovalores negativos)
        try:
            eigvals, eigvecs = np.linalg.eigh(corr)
            eigvals = np.maximum(eigvals, 1e-6)
            psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            d = np.sqrt(np.maximum(np.diag(psd), 1e-12))
            psd = psd / np.outer(d, d)
            np.fill_diagonal(psd, 1.0)
            return np.linalg.cholesky(psd)
        except np.linalg.LinAlgError:
            for eps in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
                try:
                    return np.linalg.cholesky(corr + np.eye(n) * eps)
                except np.linalg.LinAlgError:
                    continue
            return np.eye(n)


def shrink_correlation(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage sobre covarianza -> matriz de correlación."""
    try:
        from sklearn.covariance import LedoitWolf
        clean = returns.dropna()
        if len(clean) < 5 or clean.shape[1] < 2:
            return returns.corr().fillna(0.0).values
        lw = LedoitWolf().fit(clean.values)
        cov = lw.covariance_
        d = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        corr = cov / np.outer(d, d)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        return corr
    except Exception:
        c = returns.corr().fillna(0.0).values
        np.fill_diagonal(c, 1.0)
        return c


class MonteCarloEngine:
    """Motor estocástico multiactivo 100% vectorizado (compatible con Streamlit / FastAPI)."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def _draw_standard_normals(self, n_sims: int, days: int, n_assets: int,
                               antithetic: bool = False, quasi_mc: bool = False) -> np.ndarray:
        """Genera shocks normales estándar (n_sims, days, n_assets) con soporte Antithetic y Sobol QMC."""
        if quasi_mc and days * n_assets <= 21201:
            dim = days * n_assets
            sampler = qmc.Sobol(d=dim, scramble=True, seed=self.seed)
            m = int(np.ceil(np.log2(max(2, n_sims))))
            u = sampler.random_base2(m=m)[:n_sims]
            u = np.clip(u, 1e-7, 1.0 - 1e-7)
            Z = norm.ppf(u).reshape(n_sims, days, n_assets)
            return Z

        if antithetic and n_sims >= 2:
            half = n_sims // 2
            Z_half = self.rng.standard_normal((half, days, n_assets))
            Z = np.empty((n_sims, days, n_assets), dtype=float)
            Z[:half] = Z_half
            Z[half:2 * half] = -Z_half
            if n_sims % 2 == 1:
                Z[-1] = self.rng.standard_normal((days, n_assets))
            return Z

        return self.rng.standard_normal((n_sims, days, n_assets))

    # ---------- 1. GBM Paramétrico (Vectorizado) ----------
    def simulate_gbm(self, S0, mu, sigma, corr_matrix, days: int, n_sims: int,
                     dt: float = 1 / 252, antithetic: bool = False,
                     quasi_mc: bool = False) -> np.ndarray:
        S0 = np.asarray(S0, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        n_assets = len(S0)
        L = _nearest_psd_cholesky(np.asarray(corr_matrix, dtype=float))

        drift = (mu - 0.5 * sigma ** 2) * dt
        vol_step = sigma * np.sqrt(dt)

        Z = self._draw_standard_normals(n_sims, days, n_assets, antithetic=antithetic, quasi_mc=quasi_mc)
        corr_Z = Z @ L.T

        log_rets = drift[None, None, :] + vol_step[None, None, :] * corr_Z
        cum_log_rets = np.cumsum(log_rets, axis=1)

        paths = np.empty((n_sims, days + 1, n_assets), dtype=float)
        paths[:, 0, :] = S0
        paths[:, 1:, :] = S0[None, None, :] * np.exp(np.clip(cum_log_rets, -50.0, 50.0))
        np.nan_to_num(paths, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(paths, 0.0)

    # ---------- 2. Student-t Multivariante Estandarizada (Colas Pesadas) ----------
    def simulate_student_t(self, S0, mu, sigma, corr_matrix, days: int, n_sims: int,
                           df: float = 5.0, dt: float = 1 / 252,
                           antithetic: bool = False, quasi_mc: bool = False) -> np.ndarray:
        """Simulación con innovaciones t-Student multivariantes de varianza unitaria.
        Normaliza por sqrt((df - 2) / df) para no distorsionar la volatilidad anual objetivo σ.
        """
        S0 = np.asarray(S0, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        n_assets = len(S0)
        df_eff = max(float(df), 2.05)
        L = _nearest_psd_cholesky(np.asarray(corr_matrix, dtype=float))

        drift = (mu - 0.5 * sigma ** 2) * dt
        vol_step = sigma * np.sqrt(dt)

        Z = self._draw_standard_normals(n_sims, days, n_assets, antithetic=antithetic, quasi_mc=quasi_mc)
        corr_Z = Z @ L.T

        # Escala Chi-cuadrado compartida por activo en cada día -> preserva dependencia de cola
        chi2 = self.rng.chisquare(df_eff, size=(n_sims, days, 1))
        # Factor de estandarización sqrt((df - 2)/df) para mantener Var(t_shock) = 1.0
        std_scale = np.sqrt((df_eff - 2.0) / df_eff)
        t_shocks = (corr_Z / np.sqrt(chi2 / df_eff)) * std_scale

        log_rets = drift[None, None, :] + vol_step[None, None, :] * t_shocks
        cum_log_rets = np.cumsum(log_rets, axis=1)

        paths = np.empty((n_sims, days + 1, n_assets), dtype=float)
        paths[:, 0, :] = S0
        paths[:, 1:, :] = S0[None, None, :] * np.exp(np.clip(cum_log_rets, -50.0, 50.0))
        np.nan_to_num(paths, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(paths, 0.0)

    # ---------- 3. Merton Jump-Diffusion (Saltos de Poisson + Difusión) ----------
    def simulate_jump_diffusion(self, S0, mu, sigma, corr_matrix, days: int, n_sims: int,
                                jump_intensity: float = 1.5, jump_mean: float = -0.04,
                                jump_std: float = 0.06, dt: float = 1 / 252,
                                antithetic: bool = False) -> np.ndarray:
        """Modelo de Merton con saltos compuestos de Poisson para capturar crashes abruptos.
        - jump_intensity (λ): número esperado de saltos por año.
        - jump_mean (μ_J): tamaño medio logarítmico del salto (ej. -4%).
        - jump_std (σ_J): desviación estándar del salto.
        """
        S0 = np.asarray(S0, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        n_assets = len(S0)
        L = _nearest_psd_cholesky(np.asarray(corr_matrix, dtype=float))

        # Compensador del drift por saltos: k = E[e^Y - 1]
        k = np.exp(jump_mean + 0.5 * jump_std ** 2) - 1.0
        drift = (mu - 0.5 * sigma ** 2 - jump_intensity * k) * dt
        vol_step = sigma * np.sqrt(dt)

        Z = self._draw_standard_normals(n_sims, days, n_assets, antithetic=antithetic)
        corr_Z = Z @ L.T

        # Eventos Poisson compartidos (shock sistémico) + componente idiosincrático
        n_jumps = self.rng.poisson(lam=max(0.0, jump_intensity * dt), size=(n_sims, days, 1))
        jump_sizes = self.rng.normal(loc=jump_mean, scale=jump_std, size=(n_sims, days, n_assets))
        jump_component = n_jumps * jump_sizes

        log_rets = drift[None, None, :] + vol_step[None, None, :] * corr_Z + jump_component
        cum_log_rets = np.cumsum(log_rets, axis=1)

        paths = np.empty((n_sims, days + 1, n_assets), dtype=float)
        paths[:, 0, :] = S0
        paths[:, 1:, :] = S0[None, None, :] * np.exp(np.clip(cum_log_rets, -50.0, 50.0))
        np.nan_to_num(paths, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(paths, 0.0)

    # ---------- 4. Bootstrap Histórico (Vectorizado) ----------
    def simulate_historical_bootstrap(self, S0, historical_returns: pd.DataFrame,
                                      days: int, n_sims: int) -> np.ndarray:
        S0 = np.asarray(S0, dtype=float)
        arr = historical_returns.values.astype(float)
        arr = arr[~np.isnan(arr).any(axis=1)]
        n_hist = len(arr)
        if n_hist == 0:
            raise ValueError("Sin retornos históricos válidos.")
        idx = self.rng.integers(0, n_hist, size=(n_sims, days))
        sampled = arr[idx]  # (n_sims, days, n_assets)
        paths = np.empty((n_sims, days + 1, len(S0)), dtype=float)
        paths[:, 0, :] = S0
        growth = np.cumprod(1.0 + sampled, axis=1)
        paths[:, 1:, :] = S0[None, None, :] * growth
        return np.maximum(paths, 0.0)

    # ---------- 5. Block Bootstrap Circular (100% Vectorizado) ----------
    def simulate_block_bootstrap(self, S0, historical_returns: pd.DataFrame,
                                 days: int, n_sims: int, block_size: int = 20) -> np.ndarray:
        """Block bootstrap completamente vectorizado mediante indexación matricial."""
        S0 = np.asarray(S0, dtype=float)
        arr = historical_returns.dropna().values.astype(float)
        n_hist, n_assets = arr.shape
        if n_hist == 0:
            raise ValueError("Sin retornos históricos válidos.")
        block_size = max(1, min(int(block_size), n_hist // 2 or 1))
        n_blocks = int(np.ceil(days / block_size))

        # Índices iniciales de cada bloque: (n_sims, n_blocks)
        starts = self.rng.integers(0, max(1, n_hist - block_size + 1), size=(n_sims, n_blocks))
        offsets = np.arange(block_size)[None, None, :]  # (1, 1, block_size)
        idx = (starts[:, :, None] + offsets).reshape(n_sims, n_blocks * block_size)[:, :days]

        sampled = arr[idx]  # (n_sims, days, n_assets)
        paths = np.empty((n_sims, days + 1, n_assets), dtype=float)
        paths[:, 0, :] = S0
        paths[:, 1:, :] = S0[None, None, :] * np.cumprod(1.0 + sampled, axis=1)
        return np.maximum(paths, 0.0)
