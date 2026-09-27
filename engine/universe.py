# engine/universe.py
"""Presets de tickers por clase de activo (Yahoo Finance)."""

ASSET_CLASSES = {
    "Acciones / ETFs": {
        "desc": "Renta variable: acciones individuales y ETFs. Usa precios ajustados por dividendos.",
        "tickers": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "VOO", "VTI", "AGG", "GLD"],
        "defaults": ["SPY", "QQQ", "AAPL"],
        "info": "Ajusta drift, volatilidad y dividendos. Los ETFs reparten dividendos: usa auto_adjust.",
    },
    "Indices": {
        "desc": "Indices bursatiles (ETFs espejo o futuros de indice). Tendencia + baja friccion.",
        "tickers": ["^GSPC", "^NDX", "^DJI", "^RUT", "^FTSE", "^N225", "^STOXX50E", "SPY", "QQQ", "DIA"],
        "defaults": ["^GSPC", "^NDX"],
        "info": "Algunos indices (^GSPC) no son invertibles directos; para cartera usa SPY/QQQ/DIA.",
    },
    "Forex": {
        "desc": "Pares de divisas. Drift cercano a 0, aplica spread, swap y apalancamiento.",
        "tickers": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X",
                    "AUDUSD=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"],
        "defaults": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
        "info": "Configura lote, apalancamiento, spread (bps) y swap diario. Convierte PnL a tu divisa base.",
    },
    "Metales": {
        "desc": "Oro, plata, platino, cobre: spot y futuros COMEX/NYMEX.",
        "tickers": ["GC=F", "SI=F", "PL=F", "HG=F", "PA=F", "XAUUSD=X", "XAGUSD=X", "GLD", "SLV"],
        "defaults": ["GC=F", "SI=F", "XAUUSD=X"],
        "info": "Los futuros tienen roll yield (contango/backwardation) y margen. El spot no.",
    },
    "Futuros": {
        "desc": "Futuros continuos: energia, agricultura, bonos e indices.",
        "tickers": ["ES=F", "NQ=F", "YM=F", "CL=F", "BZ=F", "NG=F", "GC=F", "SI=F", "ZC=F", "ZS=F", "ZB=F"],
        "defaults": ["ES=F", "NQ=F", "CL=F"],
        "info": "Contratos continuos con roll periodico. Ajusta roll_cost y multiplicador.",
    },
}

CLASS_TIPS = {
    "Acciones / ETFs": "- Usa 5+ anos de historia.\n- Student-t capta mejor las colas.\n- Considera expense ratio en ETFs.",
    "Indices": "- Los indices suavizan idiosincrasia.\n- Correlacion alta entre indices USA: usa shrinkage.\n- Bootstrap por bloques preserva regimenes.",
    "Forex": "- Pon drift = 0 (carry aparte).\n- Volatilidad EWMA reacciona mas rapido.\n- No olvides swap si mantienes dias/semanas.",
    "Metales": "- Oro/plata se disparan en crisis: Student-t o historico.\n- Futuros: activa coste de roll.\n- Spot vs futuro difieren por contango.",
    "Futuros": "- Cada futuro tiene multiplicador y margen.\n- El roll erosiona en contango.\n- Apalancamiento amplifica drawdowns.",
}
