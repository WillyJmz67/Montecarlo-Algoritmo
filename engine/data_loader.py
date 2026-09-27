# engine/data_loader.py
"""Descarga (yfinance) + carga CSV + cache Parquet."""
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _clean_close(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance nuevo: columnas (Campo, Ticker)
        if "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
    elif "Close" in data.columns:
        pass
    # Aplanar columnas restantes
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all")


def fetch_yahoo(tickers, start, end=None, auto_adjust=True, use_cache=True) -> pd.DataFrame:
    import yfinance as yf
    key = f"yahoo_{'_'.join(sorted(tickers))}_{start}_{end}.parquet"
    f = CACHE_DIR / key.replace("=", "").replace("^", "")
    if use_cache and f.exists():
        try:
            return pd.read_parquet(f)
        except Exception:
            pass
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start, end=end, auto_adjust=auto_adjust,
                       progress=False, threads=True)
    prices = _clean_close(data)
    # Asegurar columnas = tickers pedidos
    if isinstance(prices, pd.DataFrame):
        # yfinance puede devolver 1 col sin nombre
        if len(prices.columns) == 1 and len(tickers) == 1:
            prices.columns = tickers
    if use_cache and not prices.empty:
        try:
            prices.to_parquet(f)
        except Exception:
            pass
    return prices.dropna(how="all")


def load_csv(file, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file, **kwargs)
    # Intentar: primera columna = fecha
    try:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0]).sort_index()
    except Exception:
        pass
    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
