# engine/data_loader.py
"""Descarga (yfinance) + carga CSV + caché Parquet con resolución robusta de rutas."""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _clean_close(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all")


def fetch_yahoo(tickers, start, end=None, auto_adjust=True, use_cache=True) -> pd.DataFrame:
    import yfinance as yf
    end_str = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    clean_tickers = [t.strip() for t in tickers if t.strip()]
    key = f"yahoo_{'_'.join(sorted(clean_tickers))}_{start}_{end_str}.parquet"
    f = CACHE_DIR / key.replace("=", "").replace("^", "")

    if use_cache and f.exists():
        try:
            return pd.read_parquet(f)
        except Exception:
            pass

    try:
        data = yf.download(clean_tickers, start=start, end=end_str, auto_adjust=auto_adjust,
                           progress=False, threads=True)
        prices = _clean_close(data)
        if isinstance(prices, pd.DataFrame):
            if len(prices.columns) == 1 and len(clean_tickers) == 1:
                prices.columns = clean_tickers
        if use_cache and not prices.empty:
            try:
                prices.to_parquet(f)
            except Exception:
                pass
        if not prices.empty:
            return prices.dropna(how="all")
    except Exception:
        pass

    # Fallback: buscar en caché algún archivo que contenga los tickers solicitados
    if use_cache:
        for cached_file in sorted(CACHE_DIR.glob("yahoo_*.parquet"), reverse=True):
            try:
                df_c = pd.read_parquet(cached_file)
                common = [t for t in clean_tickers if t in df_c.columns]
                if common:
                    sub = df_c[common].loc[str(start):str(end_str)]
                    if not sub.empty:
                        return sub.dropna(how="all")
            except Exception:
                continue
    return pd.DataFrame()


def load_csv(file, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file, **kwargs)
    try:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0]).sort_index()
    except Exception:
        pass
    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
