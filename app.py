# app.py — MC Desk · Quant Terminal (Interfaz Institucional Multiactivo en Streamlit)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from engine.models import MonteCarloEngine, shrink_correlation
from engine.metrics import (
    calculate_var, calculate_cvar, max_drawdown, drawdown_paths,
    percentile_table, sharpe_sortino, cagr, advanced_risk_metrics,
)
from engine.volatility import VolatilityModel
from engine.portfolio import (
    portfolio_paths_from_asset_paths, apply_generic_costs, apply_contributions,
    total_invested_capital, covariance_from_corr, portfolio_stats,
    optimize_weights, efficient_frontier_curve,
)
from engine.forex import pip_size, apply_forex_costs
from engine.metals import FUTURES_SPECS, apply_futures_roll
from engine.universe import ASSET_CLASSES, CLASS_TIPS
from engine.data_loader import fetch_yahoo, load_csv
from engine.ui_theme import (
    QUANT_COLORS, ASSET_PALETTE, inject_custom_css, render_sidebar_brand,
    render_hero_header, render_kpi_grid, render_section_header, apply_quant_layout,
)

st.set_page_config(
    page_title="MC Desk · Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()
render_sidebar_brand()

# ============ SIDEBAR INSTITUCIONAL ============
asset_class = st.sidebar.selectbox(
    "Clase de Activo",
    list(ASSET_CLASSES.keys()),
    index=0,
    help="Selecciona el mercado principal para ajustar costes, apalancamiento y universo sugerido.",
)
cfg_class = ASSET_CLASSES[asset_class]
st.sidebar.caption(f"💡 {cfg_class['info']}")

# --- 1. Universo & Datos ---
with st.sidebar.expander("🌐 Universo & Datos Históricos", expanded=True):
    preset = st.multiselect(
        "Tickers sugeridos",
        cfg_class["tickers"],
        default=[t for t in cfg_class["defaults"] if t in cfg_class["tickers"]],
    )
    custom = st.text_input("Añadir símbolos (separados por coma)", placeholder="Ej: NVDA, GLD, EURUSD=X")
    extra = [t.strip().upper() for t in custom.split(",") if t.strip()] if custom else []
    tickers = sorted(set(preset + extra))

    fuente = st.radio("Fuente de precios", ["Yahoo Finance", "Subir CSV"], horizontal=True)
    up_prices = None
    if fuente == "Subir CSV":
        up = st.file_uploader("CSV (1ª col = Fecha, resto = Precios)", type=["csv"])
        if up is not None:
            try:
                up_prices = load_csv(up)
                tickers = [c for c in up_prices.columns if c in tickers] or list(up_prices.columns)
                st.success(f"CSV cargado: {len(up_prices):,} días · {len(tickers)} activos")
            except Exception as e:
                st.error(f"Error al leer CSV: {e}")

    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Desde", value=pd.to_datetime("2018-01-01"))
    end_date = col_d2.date_input("Hasta", value=pd.Timestamp.today())
    auto_adj = (
        st.checkbox("Ajustar por dividendos y splits", value=True)
        if asset_class == "Acciones / ETFs"
        else True
    )

if not tickers:
    st.warning("Selecciona al menos 1 activo en el panel lateral para iniciar la terminal.")
    st.stop()

# --- Descarga y validación temprana de precios para permitir optimización de pesos ---
@st.cache_data(show_spinner=False)
def get_prices(tickers_key, start, end, auto_adj_flag, fuente_nombre):
    if fuente_nombre == "Subir CSV":
        return None
    return fetch_yahoo(list(tickers_key), str(start), str(end), auto_adjust=auto_adj_flag)

prices_raw = (
    up_prices
    if (fuente == "Subir CSV" and up_prices is not None)
    else get_prices(tuple(tickers), str(start_date), str(end_date), auto_adj, fuente)
)
if prices_raw is None or (isinstance(prices_raw, pd.DataFrame) and prices_raw.empty):
    st.error("No se pudieron obtener datos históricos. Verifica los símbolos, el rango de fechas o tu conexión.")
    st.stop()

avail = [t for t in tickers if t in (prices_raw.columns if isinstance(prices_raw, pd.DataFrame) else [])]
missing = [t for t in tickers if t not in avail]
if missing:
    st.toast(f"Sin datos para: {', '.join(missing)} (excluidos)", icon="⚠️")
if not avail:
    st.error("Ninguno de los símbolos seleccionados devolvió series válidas.")
    st.stop()

prices = prices_raw[avail].dropna(how="all").ffill().dropna()
tickers = avail
S0 = prices.iloc[-1].values.astype(float)
returns = prices.pct_change().dropna()
if len(returns) < 40:
    st.error("Historial insuficiente (<40 sesiones). Amplía la fecha de inicio.")
    st.stop()

# --- 2. Motor Estocástico & Volatilidad ---
with st.sidebar.expander("⚙️ Motor Estocástico & Volatilidad", expanded=True):
    MODEL_LABELS = {
        "student_t": "Student-t (Colas Pesadas)",
        "gbm": "GBM Paramétrico (Normal)",
        "jump_diffusion": "Merton Jump-Diffusion (Saltos)",
        "historical": "Bootstrap Histórico",
        "block_bootstrap": "Bootstrap por Bloques",
    }
    model = st.selectbox("Modelo generador", list(MODEL_LABELS.keys()), index=0, format_func=lambda x: MODEL_LABELS[x])

    c_sim1, c_sim2 = st.columns(2)
    n_paths = c_sim1.number_input("Trayectorias", 500, 100_000, 10_000, step=2_500)
    horizon = c_sim2.number_input("Horizonte (días)", 10, 1260, 252, step=21)

    if model == "student_t":
        df_t = st.slider("Grados de libertad (ν)", 3, 15, 5, help="Menor ν = colas más pesadas (eventos extremos más frecuentes).")
    else:
        df_t = 5

    if model == "jump_diffusion":
        c_j1, c_j2 = st.columns(2)
        jump_lambda = c_j1.number_input("Saltos / año (λ)", 0.2, 10.0, 1.5, step=0.5)
        jump_mean = c_j2.number_input("Media salto (%)", -25.0, 15.0, -4.0, step=1.0) / 100.0
        jump_std = st.slider("Volatilidad del salto (%)", 1.0, 20.0, 6.0, step=0.5) / 100.0
    else:
        jump_lambda, jump_mean, jump_std = 1.5, -0.04, 0.06

    if model == "block_bootstrap":
        block_size = st.slider("Tamaño de bloque (días)", 5, 60, 20)
    else:
        block_size = 20

    VOL_LABELS = {
        "ewma": "EWMA (Ponderación Reciente)",
        "historica": "Histórica Muestral",
        "garch": "GARCH(1,1) Condicional",
    }
    vol_opt = st.selectbox("Estimador de Volatilidad (σ)", list(VOL_LABELS.keys()), index=0, format_func=lambda x: VOL_LABELS[x])
    ewma_span = st.slider("Ventana EWMA (días)", 20, 180, 60) if vol_opt == "ewma" else 60

    default_drift_idx = 1 if asset_class == "Forex" else 0
    drift_opt = st.selectbox(
        "Deriva esperada (Drift μ)",
        ["historico", "cero", "personalizado"],
        index=default_drift_idx,
        format_func=lambda x: {"historico": "Media histórica anualizada", "cero": "Neutral (μ = 0%)", "personalizado": "Personalizado fijo"}[x],
    )
    drift_custom = (
        st.number_input("Drift anual objetivo (%)", -20.0, 40.0, 7.0, step=0.5) / 100.0
        if drift_opt == "personalizado"
        else None
    )

    corr_opt = st.selectbox(
        "Matriz de Correlación",
        ["shrinkage", "historica"],
        index=0,
        format_func=lambda x: "Shrinkage Ledoit-Wolf (Robusta)" if x == "shrinkage" else "Histórica Muestral",
    )

    c_v1, c_v2 = st.columns(2)
    antithetic = c_v1.checkbox("Antithetic", value=True, help="Pares Z / -Z para reducir varianza muestral.") if model in ("gbm", "student_t", "jump_diffusion") else False
    quasi_mc = c_v2.checkbox("Sobol QMC", value=False, help="Secuencia cuasi-aleatoria de baja discrepancia.") if model in ("gbm", "student_t") else False
    seed = st.number_input("Semilla reproducible", 0, 99999, 42)

# --- Cálculo de parámetros anuales (μ, σ, Correlación) ---
if drift_opt == "cero":
    mu_ann = np.zeros(len(tickers), dtype=float)
elif drift_opt == "personalizado":
    mu_ann = np.full(len(tickers), float(drift_custom), dtype=float)
else:
    mu_ann = (returns.mean() * 252.0).values.astype(float)

garch_info = {}
if vol_opt == "historica":
    sigma_ann = VolatilityModel.historical(returns).values.astype(float)
    vol_note = "Volatilidad histórica equiponderada (√252)."
elif vol_opt == "ewma":
    sigma_ann = VolatilityModel.ewma(returns, span=ewma_span).values.astype(float)
    vol_note = f"Volatilidad exponencial EWMA (ventana λ = {ewma_span} sesiones)."
else:
    s_list = []
    for c in returns.columns:
        r_g = VolatilityModel.garch_last(returns[c])
        s_list.append(r_g["vol_annual"])
        garch_info[c] = r_g
    sigma_ann = np.array(s_list, dtype=float)
    vol_note = "Volatilidad condicional GARCH(1,1) estimada por máxima verosimilitud."

corr = shrink_correlation(returns) if corr_opt == "shrinkage" else returns.corr().fillna(0.0).values

# --- 3. Capital, Estrategia de Pesos & Rebalanceo ---
with st.sidebar.expander("💼 Capital & Asignación de Cartera", expanded=True):
    initial_capital = st.number_input("Capital Inicial ($)", 100.0, 100_000_000.0, 10_000.0, step=1_000.0)
    weight_mode = st.selectbox(
        "Estrategia de Pesos",
        ["custom", "equal", "risk_parity", "min_vol", "max_sharpe"],
        format_func=lambda x: {
            "custom": "🎛️ Manual (Sliders)",
            "equal": "⚖️ Equiponderada (1/N)",
            "risk_parity": "🛡️ Risk Parity (Igual Riesgo)",
            "min_vol": "📉 Mínima Volatilidad",
            "max_sharpe": "🚀 Máximo Ratio Sharpe",
        }[x],
    )

    if weight_mode == "custom":
        raw_w = []
        default_w = int(round(100 / len(tickers))) if tickers else 0
        for t in tickers:
            w_val = st.slider(f"{t} (%)", 0, 100, default_w, key=f"w_{t}_{asset_class}")
            raw_w.append(w_val)
        weights = np.array(raw_w, dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(tickers)) / len(tickers)
    else:
        weights = optimize_weights(mu_ann, sigma_ann, corr, risk_free=0.03, method=weight_mode)
        st.caption(" · ".join(f"**{t}**: {w:.1%}" for t, w in zip(tickers, weights)))

    rebalance = st.selectbox(
        "Frecuencia de Rebalanceo",
        ["monthly", "quarterly", "none", "weekly", "daily"],
        format_func=lambda x: {
            "none": "Buy & Hold (Sin rebalanceo)",
            "monthly": "Mensual (cada 21d)",
            "quarterly": "Trimestral (cada 63d)",
            "weekly": "Semanal (cada 5d)",
            "daily": "Diario",
        }[x],
    )

# --- 4. Costes, Fricciones & Objetivos ---
with st.sidebar.expander("💸 Costes, Fricciones & Objetivos", expanded=False):
    c_f1, c_f2 = st.columns(2)
    commission_bps = c_f1.number_input("Comisión (bps)", 0.0, 100.0, 5.0)
    slippage_bps = c_f2.number_input("Slippage (bps)", 0.0, 100.0, 2.0)
    monthly_contrib = st.number_input("Aportación Mensual ($)", 0.0, 500_000.0, 0.0, step=100.0)

    c_r1, c_r2 = st.columns(2)
    risk_free = c_r1.number_input("Tasa RF (%)", 0.0, 15.0, 3.5, step=0.25) / 100.0
    target_ret = c_r2.number_input("Meta Retorno (%)", -50.0, 300.0, 12.0, step=1.0) / 100.0

    fx_spread = fx_swap = fx_lev = None
    mt_roll = mt_rolls_year = None
    expense_bps = 0.0

    if asset_class == "Forex":
        fx_spread = st.number_input("Spread FX (bps)", 0.0, 50.0, 1.0)
        fx_swap = st.number_input("Swap diario (bps, + = coste)", -10.0, 10.0, 0.0)
        fx_lev = st.number_input("Apalancamiento (x)", 1.0, 100.0, 1.0, step=1.0)
    elif asset_class in ("Metales", "Futuros"):
        mt_roll = st.number_input("Coste Roll / vencimiento (bps)", 0.0, 100.0, 2.0)
        mt_rolls_year = st.number_input("Vencimientos (Rolls / año)", 0.0, 12.0, 6.0)
    elif asset_class == "Acciones / ETFs":
        expense_bps = st.number_input("TER / Expense Ratio anual (bps)", 0.0, 200.0, 5.0)

run_btn = st.sidebar.button("🚀 Recalcular Simulación", type="primary", use_container_width=True)

# ============ HEADER PRINCIPAL ============
render_hero_header(
    asset_class=asset_class,
    desc=cfg_class["desc"],
    model_label=MODEL_LABELS[model],
    vol_label=VOL_LABELS[vol_opt].split(" ")[0],
    n_assets=len(tickers),
    horizon=int(horizon),
    n_paths=int(n_paths),
    capital=float(initial_capital),
)

# ============ EJECUCIÓN DEL MOTOR MONTE CARLO ============
sim_signature = (
    tuple(tickers), tuple(np.round(weights, 4)), str(start_date), str(end_date),
    model, int(n_paths), int(horizon), int(seed), bool(antithetic), bool(quasi_mc),
    float(df_t), int(block_size), float(jump_lambda), float(jump_mean), float(jump_std),
    drift_opt, float(drift_custom or 0.0), vol_opt, int(ewma_span), corr_opt,
    rebalance, float(initial_capital), float(commission_bps), float(slippage_bps),
    float(monthly_contrib), float(expense_bps),
    float(fx_spread or 0.0), float(fx_swap or 0.0), float(fx_lev or 1.0),
    float(mt_roll or 0.0), float(mt_rolls_year or 0.0),
)

if run_btn or st.session_state.get("sim_sig") != sim_signature:
    eng = MonteCarloEngine(seed=int(seed))
    with st.spinner(f"Ejecutando {n_paths:,} simulaciones estocásticas ({MODEL_LABELS[model]})…"):
        if model == "gbm":
            paths = eng.simulate_gbm(S0, mu_ann, sigma_ann, corr, int(horizon), int(n_paths),
                                     antithetic=antithetic, quasi_mc=quasi_mc)
        elif model == "student_t":
            paths = eng.simulate_student_t(S0, mu_ann, sigma_ann, corr, int(horizon), int(n_paths),
                                           df=float(df_t), antithetic=antithetic, quasi_mc=quasi_mc)
        elif model == "jump_diffusion":
            paths = eng.simulate_jump_diffusion(S0, mu_ann, sigma_ann, corr, int(horizon), int(n_paths),
                                                jump_intensity=float(jump_lambda), jump_mean=float(jump_mean),
                                                jump_std=float(jump_std), antithetic=antithetic)
        elif model == "block_bootstrap":
            paths = eng.simulate_block_bootstrap(S0, returns, int(horizon), int(n_paths), block_size=int(block_size))
        else:
            paths = eng.simulate_historical_bootstrap(S0, returns, int(horizon), int(n_paths))

        if asset_class in ("Metales", "Futuros") and mt_roll:
            paths = apply_futures_roll(paths, roll_bps=float(mt_roll), rolls_per_year=float(mt_rolls_year), days=int(horizon))

        port_norm = portfolio_paths_from_asset_paths(paths, weights, rebalance=rebalance)
        port_val = port_norm * float(initial_capital)

        n_rb = 0 if rebalance == "none" else max(1, int(horizon / {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63}[rebalance]))
        port_val = apply_generic_costs(port_val, commission_bps, slippage_bps, n_rb)

        if expense_bps:
            t_axis = np.linspace(0.0, horizon / 252.0, port_val.shape[1])[None, :]
            port_val = port_val * ((1.0 - expense_bps / 10_000.0) ** t_axis)

        if asset_class == "Forex" and fx_spread is not None:
            port_val = apply_forex_costs(port_val, spread_bps=fx_spread, swap_bps_dia=fx_swap or 0.0,
                                         days=int(horizon), leverage=float(fx_lev or 1.0))

        if monthly_contrib and monthly_contrib > 0:
            port_val = apply_contributions(port_val, monthly_contrib, float(initial_capital))

        invested_total = total_invested_capital(port_val.shape[1], float(initial_capital), float(monthly_contrib))

    st.session_state["sim_sig"] = sim_signature
    st.session_state["res"] = dict(
        paths=paths,
        port=port_val,
        init_cap=float(initial_capital),
        invested_total=float(invested_total),
        tickers=list(tickers),
        weights=np.array(weights),
        mu=mu_ann.copy(),
        sigma=sigma_ann.copy(),
        corr=np.array(corr),
    )

R = st.session_state["res"]
paths = R["paths"]
port = R["port"]
init_cap = R["init_cap"]
invested_total = R["invested_total"]
final = port[:, -1]
years = max(horizon / 252.0, 1e-6)

# Métricas clave respecto al capital total desembolsado (inicial + aportaciones)
p_loss = float(np.mean(final < invested_total))
p_target = float(np.mean(final >= invested_total * (1.0 + target_ret)))
med = float(np.median(final))
mean_val = float(np.mean(final))
roi_med = (med / invested_total) - 1.0
roi_mean = (mean_val / invested_total) - 1.0

v95 = calculate_var(final, invested_total, 0.95)
c95 = calculate_cvar(final, invested_total, 0.95)
v99 = calculate_var(final, invested_total, 0.99)
c99 = calculate_cvar(final, invested_total, 0.99)

dd_matrix = drawdown_paths(port)
mdd = np.min(dd_matrix, axis=1)
sharpe, sortino = sharpe_sortino(port, risk_free)
adv_metrics = advanced_risk_metrics(port, invested_total, years, risk_free)
q, qv, qr = percentile_table(final, invested_total)

# ============ NAVEGACIÓN PRINCIPAL EN 5 PESTAÑAS ============
tab_sim, tab_risk, tab_opt, tab_data, tab_asset = st.tabs([
    "🔮 Simulación & Proyección",
    "⚠️ Riesgo, Cola & Stress Test",
    "⚖️ Optimización & Frontera Eficiente",
    "📊 Datos & Diagnóstico Histórico",
    "🎯 Inspector por Activo & Guía",
])

# -------------------------------------------------------------------------
# PESTAÑA 1: SIMULACIÓN & PROYECCIÓN
# -------------------------------------------------------------------------
with tab_sim:
    render_kpi_grid([
        {
            "label": "Capital Desembolsado",
            "value": f"${invested_total:,.0f}",
            "pill_text": f"Base ${init_cap:,.0f}",
            "pill_type": "neu",
            "sub": f"+${monthly_contrib:,.0f}/mes" if monthly_contrib > 0 else "Sin aportes extra",
            "accent": QUANT_COLORS["slate"],
        },
        {
            "label": "Mediana Terminal (P50)",
            "value": f"${med:,.0f}",
            "pill_text": f"{roi_med:+.1%}",
            "pill_type": "pos" if roi_med >= 0 else "neg",
            "sub": f"PnL: ${med - invested_total:+,.0f}",
            "accent": QUANT_COLORS["cyan"],
        },
        {
            "label": "Valor Esperado (Media)",
            "value": f"${mean_val:,.0f}",
            "pill_text": f"{roi_mean:+.1%}",
            "pill_type": "pos" if roi_mean >= 0 else "neg",
            "sub": f"CAGR P50: {cagr(invested_total, med, years):+.1%}",
            "accent": QUANT_COLORS["indigo"],
        },
        {
            "label": "Probabilidad de Pérdida",
            "value": f"{p_loss:.1%}",
            "pill_text": "Bajo control" if p_loss < 0.30 else ("Moderada" if p_loss < 0.50 else "Alerta"),
            "pill_type": "pos" if p_loss < 0.30 else ("warn" if p_loss < 0.50 else "neg"),
            "sub": f"Escenarios < ${invested_total:,.0f}",
            "accent": QUANT_COLORS["emerald"] if p_loss < 0.35 else QUANT_COLORS["rose"],
        },
        {
            "label": f"Prob. Meta (≥ +{target_ret:.0%})",
            "value": f"{p_target:.1%}",
            "pill_text": f"Meta ${invested_total*(1+target_ret):,.0f}",
            "pill_type": "pos" if p_target >= 0.40 else "warn",
            "sub": f"Horizonte {horizon} días",
            "accent": QUANT_COLORS["emerald"],
        },
        {
            "label": "Sharpe / Sortino Sim.",
            "value": f"{sharpe:.2f} / {sortino:.2f}",
            "pill_text": f"RF {risk_free:.1%}",
            "pill_type": "pos" if sharpe > 0.7 else "neu",
            "sub": f"MaxDD P50: {np.median(mdd):.1%}",
            "accent": QUANT_COLORS["violet"],
        },
    ])

    col_fan, col_paths = st.columns([1.15, 1.0])

    with col_fan:
        bands = {p: np.percentile(port, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)}
        x_days = np.arange(port.shape[1])

        fig_fan = go.Figure()
        # Banda externa P5 - P95
        fig_fan.add_trace(go.Scatter(x=x_days, y=bands[95], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_fan.add_trace(go.Scatter(
            x=x_days, y=bands[5], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(56, 189, 248, 0.10)", name="Banda P5–P95",
        ))
        # Banda intermedia P10 - P90
        fig_fan.add_trace(go.Scatter(x=x_days, y=bands[90], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_fan.add_trace(go.Scatter(
            x=x_days, y=bands[10], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(56, 189, 248, 0.20)", name="Banda P10–P90",
        ))
        # Banda central P25 - P75
        fig_fan.add_trace(go.Scatter(x=x_days, y=bands[75], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_fan.add_trace(go.Scatter(
            x=x_days, y=bands[25], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(56, 189, 248, 0.34)", name="Rango Intercuartil P25–P75",
        ))
        # Mediana y Media
        fig_fan.add_trace(go.Scatter(
            x=x_days, y=bands[50], mode="lines", name="Mediana (P50)",
            line=dict(color="#38BDF8", width=2.8),
        ))
        fig_fan.add_trace(go.Scatter(
            x=x_days, y=np.mean(port, axis=0), mode="lines", name="Media Esperada",
            line=dict(color="#A855F7", width=2.0, dash="dot"),
        ))
        fig_fan.add_hline(
            y=invested_total, line_dash="dash", line_color="rgba(244, 63, 94, 0.75)",
            annotation_text="Capital Invertido", annotation_position="bottom right",
        )
        apply_quant_layout(fig_fan, title="Cono de Probabilidad de Cartera (Percentiles P5 a P95)",
                           height=400, xaxis_title="Días de Simulación", yaxis_title="Valor de Cartera ($)")
        st.plotly_chart(fig_fan, use_container_width=True)

    with col_paths:
        fig_sp = go.Figure()
        n_sample = min(120, len(port))
        for i in range(n_sample):
            fig_sp.add_trace(go.Scatter(
                y=port[i], mode="lines",
                line=dict(color="rgba(129, 140, 248, 0.12)", width=0.8),
                showlegend=False, hoverinfo="skip",
            ))
        fig_sp.add_trace(go.Scatter(
            y=bands[95], mode="lines", name="P95 Alcista",
            line=dict(color="#10B981", width=1.8, dash="dot"),
        ))
        fig_sp.add_trace(go.Scatter(
            y=bands[50], mode="lines", name="Mediana (P50)",
            line=dict(color="#38BDF8", width=2.6),
        ))
        fig_sp.add_trace(go.Scatter(
            y=bands[5], mode="lines", name="P5 Bajista (VaR 95%)",
            line=dict(color="#F43F5E", width=1.8, dash="dot"),
        ))
        apply_quant_layout(fig_sp, title=f"Muestra Estocástica ({n_sample} trayectorias) + Cotas Extremas",
                           height=400, xaxis_title="Días de Simulación", yaxis_title="Valor ($)")
        st.plotly_chart(fig_sp, use_container_width=True)

    col_hist, col_dd = st.columns(2)
    with col_hist:
        p5_val = float(np.percentile(final, 5))
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=final, nbinsx=85, name="Escenarios Finales",
            marker=dict(color="rgba(56, 189, 248, 0.65)", line=dict(color="rgba(56, 189, 248, 0.9)", width=0.5)),
        ))
        fig_h.add_vline(x=invested_total, line_dash="dash", line_color="#F59E0B", annotation_text="Invertido")
        fig_h.add_vline(x=med, line_dash="solid", line_color="#10B981", annotation_text="Mediana")
        fig_h.add_vline(x=p5_val, line_dash="dot", line_color="#F43F5E", annotation_text="P5 (VaR 95%)")
        apply_quant_layout(fig_h, title="Distribución de Valor Final al Vencimiento",
                           height=340, xaxis_title="Capital Final ($)", yaxis_title="Frecuencia", show_legend=False)
        st.plotly_chart(fig_h, use_container_width=True)

    with col_dd:
        dd_p50 = np.median(dd_matrix, axis=0) * 100.0
        dd_p90 = np.percentile(dd_matrix, 10, axis=0) * 100.0
        dd_p95 = np.percentile(dd_matrix, 5, axis=0) * 100.0
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=x_days, y=dd_p95, mode="lines", name="Peor 5% (P95 DD)",
            line=dict(color="#F43F5E", width=1.6, dash="dot"),
        ))
        fig_dd.add_trace(go.Scatter(
            x=x_days, y=dd_p90, mode="lines", name="Peor 10% (P90 DD)",
            fill="tonexty", fillcolor="rgba(244, 63, 94, 0.12)",
            line=dict(color="#FB923C", width=1.5),
        ))
        fig_dd.add_trace(go.Scatter(
            x=x_days, y=dd_p50, mode="lines", name="Drawdown Mediano",
            fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.18)",
            line=dict(color="#38BDF8", width=2.4),
        ))
        apply_quant_layout(fig_dd, title="Curva Underwater (Evolución Temporal del Drawdown %)",
                           height=340, xaxis_title="Días de Simulación", yaxis_title="Drawdown (%)")
        st.plotly_chart(fig_dd, use_container_width=True)

    c_dl1, c_dl2 = st.columns([1, 3])
    c_dl1.download_button(
        "📥 Exportar Simulaciones (CSV)",
        pd.DataFrame({
            "valor_final_usd": np.round(final, 2),
            "retorno_neto_pct": np.round((final / invested_total - 1.0) * 100.0, 2),
            "max_drawdown_pct": np.round(mdd * 100.0, 2),
        }).to_csv(index=False),
        file_name=f"mc_simulacion_{asset_class.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# -------------------------------------------------------------------------
# PESTAÑA 2: RIESGO, COLA & STRESS TEST
# -------------------------------------------------------------------------
with tab_risk:
    render_kpi_grid([
        {
            "label": "VaR 95% (Pérdida P5)",
            "value": f"${v95:,.0f}",
            "pill_text": f"-{v95/invested_total:.1%}",
            "pill_type": "warn" if v95 / invested_total < 0.20 else "neg",
            "sub": "Confianza 95%",
            "accent": QUANT_COLORS["amber"],
        },
        {
            "label": "CVaR 95% (Expected Shortfall)",
            "value": f"${c95:,.0f}",
            "pill_text": f"-{c95/invested_total:.1%}",
            "pill_type": "neg",
            "sub": "Media del 5% peor",
            "accent": QUANT_COLORS["rose"],
        },
        {
            "label": "VaR 99% / CVaR 99%",
            "value": f"${v99:,.0f} / ${c99:,.0f}",
            "pill_text": f"-{c99/invested_total:.1%} cola",
            "pill_type": "neg",
            "sub": "Cisne negro (1% peor)",
            "accent": QUANT_COLORS["rose"],
        },
        {
            "label": "Calmar / Omega Ratio",
            "value": f"{adv_metrics['calmar']:.2f} / {adv_metrics['omega']:.2f}",
            "pill_text": "Omega > 1" if adv_metrics["omega"] >= 1.0 else "Omega < 1",
            "pill_type": "pos" if adv_metrics["omega"] >= 1.0 else "neg",
            "sub": f"Ulcer Index: {adv_metrics['ulcer_index']:.1f}",
            "accent": QUANT_COLORS["cyan"],
        },
        {
            "label": "Asimetría / Curtosis",
            "value": f"{adv_metrics['skewness']:+.2f} / {adv_metrics['kurtosis']:+.2f}",
            "pill_text": f"Tail {adv_metrics['tail_ratio']:.2f}x",
            "pill_type": "pos" if adv_metrics["tail_ratio"] >= 1.0 else "warn",
            "sub": "Morfología de cola",
            "accent": QUANT_COLORS["indigo"],
        },
    ])

    col_r1, col_r2 = st.columns([1.05, 1.15])
    with col_r1:
        render_section_header("Estructura Percentilar al Vencimiento", "Desde el escenario extremo bajista (P1) hasta el alcista (P99)")
        df_perc = pd.DataFrame({
            "Percentil": [f"P{q_i}" for q_i in q],
            "Escenario": [
                "Colapso Extremo (1%)", "VaR 95% (5%)", "Pesimista (10%)",
                "Conservador (25%)", "Central Mediana (50%)", "Favorable (75%)",
                "Optimista (90%)", "Euforia (95%)", "Outlier Alcista (99%)",
            ],
            "Capital Final ($)": qv,
            "PnL Neto ($)": qv - invested_total,
            "Retorno (%)": qr / 100.0,
        })
        st.dataframe(
            df_perc.style.format({
                "Capital Final ($)": "${:,.2f}",
                "PnL Neto ($)": "${:+,.2f}",
                "Retorno (%)": "{:+.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with col_r2:
        render_section_header("Diagnóstico Ejecutivo & Lectura Cuantitativa", "Evaluación automática de solvencia y riesgo de cola")
        if p_loss > 0.50:
            v_cls, v_icon, v_title = "rose", "🔴", f"Expectativa Central Negativa (P. Pérdida = {p_loss:.1%})"
            v_body = (
                "Más de la mitad de las trayectorias terminan por debajo del capital desembolsado. "
                "Revisa la deriva esperada (drift), reduce las fricciones operativas o disminuye el apalancamiento."
            )
        elif v95 > invested_total * 0.22:
            v_cls, v_icon, v_title = "amber", "🟡", f"Riesgo de Cola Elevado (VaR 95% = -${v95:,.0f} | -{v95/invested_total:.1%})"
            v_body = (
                f"Aunque la mediana proyecta {roi_med:+.1%}, en el 5% peor de los casos la pérdida media (CVaR 95%) "
                f"alcanza -${c95:,.0f}. Considera diversificar hacia activos descorrelacionados o usar pesos Risk Parity."
            )
        else:
            v_cls, v_icon, v_title = "emerald", "🟢", f"Perfil de Riesgo Contenido (VaR 95% = -${v95:,.0f} | Omega = {adv_metrics['omega']:.2f})"
            v_body = (
                f"La probabilidad de preservar capital es del {(1-p_loss):.1%} y la probabilidad de alcanzar tu meta "
                f"(+{target_ret:.0%}) es del {p_target:.1%}. Aun así, en el 1% extremo (CVaR 99%) la pérdida sería de -${c99:,.0f}."
            )

        st.markdown(
            f"""
            <div class="verdict-box {v_cls}">
                <div class="verdict-title">{v_icon} {v_title}</div>
                <div class="verdict-desc">{v_body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_mdd = go.Figure()
        fig_mdd.add_trace(go.Histogram(
            x=mdd * 100.0, nbinsx=65, name="Max Drawdown (%)",
            marker=dict(color="rgba(244, 63, 94, 0.62)", line=dict(color="rgba(244, 63, 94, 0.9)", width=0.5)),
        ))
        fig_mdd.add_vline(x=float(np.median(mdd) * 100.0), line_dash="solid", line_color="#38BDF8", annotation_text="Mediana DD")
        fig_mdd.add_vline(x=float(np.percentile(mdd, 5) * 100.0), line_dash="dot", line_color="#F59E0B", annotation_text="Peor 5% DD")
        apply_quant_layout(fig_mdd, title="Distribución de Máximo Drawdown por Trayectoria (%)",
                           height=255, xaxis_title="Máximo Drawdown (%)", yaxis_title="Simulaciones", show_legend=False)
        st.plotly_chart(fig_mdd, use_container_width=True)

    # --- Matriz de Stress Test Paramétrico ---
    render_section_header(
        "Matriz de Stress Test (Sensibilidad ante Shocks de Volatilidad y Drift)",
        "Retorno mediano proyectado (%) en el horizonte al alterar simultáneamente σ y μ de la cartera",
    )
    cov_curr = covariance_from_corr(sigma_ann, corr)
    p_stats = portfolio_stats(weights, mu_ann, cov_curr, risk_free)
    port_mu_base = p_stats["return"]
    port_vol_base = p_stats["volatility"]

    vol_mults = [0.75, 1.00, 1.25, 1.50, 2.00]
    drift_shocks = [-0.08, -0.04, 0.00, +0.04, +0.08]
    stress_grid = np.zeros((len(drift_shocks), len(vol_mults)))
    z_p5 = -1.64485  # cuantil 5% normal para aproximar escenario de estrés P5
    for i_d, d_sh in enumerate(drift_shocks):
        for j_v, v_m in enumerate(vol_mults):
            mu_s = port_mu_base + d_sh
            vol_s = port_vol_base * v_m
            # Retorno en P5 (estrés bajista al 95% de confianza)
            r_p5 = np.exp((mu_s - 0.5 * vol_s ** 2) * years + vol_s * np.sqrt(years) * z_p5) - 1.0
            stress_grid[i_d, j_v] = r_p5 * 100.0

    fig_stress = go.Figure(data=go.Heatmap(
        z=stress_grid,
        x=[f"σ × {m:.2f} ({port_vol_base*m:.1%})" for m in vol_mults],
        y=[f"μ {d:+.0%} ({port_mu_base+d:+.1%})" for d in drift_shocks],
        colorscale="RdYlGn",
        zmid=0.0,
        text=np.round(stress_grid, 1),
        texttemplate="%{text:+.1f}%",
        textfont={"family": "JetBrains Mono, monospace", "size": 12},
        colorbar=dict(title="Retorno P5 (%)"),
    ))
    apply_quant_layout(fig_stress, title="Stress Test de Cola (Retorno en Percentil 5% bajo Shocks de Mercado)",
                       height=320, xaxis_title="Multiplicador de Volatilidad (σ)", yaxis_title="Shock en Drift Anual (μ)")
    st.plotly_chart(fig_stress, use_container_width=True)

# -------------------------------------------------------------------------
# PESTAÑA 3: OPTIMIZACIÓN & FRONTERA EFICIENTE
# -------------------------------------------------------------------------
with tab_opt:
    render_section_header(
        "Arquitectura Cuántica de Cartera & Frontera Eficiente de Markowitz",
        "Comparativa entre tu asignación actual y las carteras institucionales óptimas",
    )
    cov_mat = covariance_from_corr(sigma_ann, corr)
    w_eq = optimize_weights(mu_ann, sigma_ann, corr, risk_free, "equal")
    w_mv = optimize_weights(mu_ann, sigma_ann, corr, risk_free, "min_vol")
    w_ms = optimize_weights(mu_ann, sigma_ann, corr, risk_free, "max_sharpe")
    w_rp = optimize_weights(mu_ann, sigma_ann, corr, risk_free, "risk_parity")

    st_curr = portfolio_stats(weights, mu_ann, cov_mat, risk_free)
    st_eq = portfolio_stats(w_eq, mu_ann, cov_mat, risk_free)
    st_mv = portfolio_stats(w_mv, mu_ann, cov_mat, risk_free)
    st_ms = portfolio_stats(w_ms, mu_ann, cov_mat, risk_free)
    st_rp = portfolio_stats(w_rp, mu_ann, cov_mat, risk_free)

    col_ef, col_rc = st.columns([1.2, 1.0])
    with col_ef:
        ef_vols, ef_rets = efficient_frontier_curve(mu_ann, sigma_ann, corr, n_points=35)
        fig_ef = go.Figure()
        if len(ef_vols) > 1:
            fig_ef.add_trace(go.Scatter(
                x=ef_vols * 100.0, y=ef_rets * 100.0, mode="lines",
                name="Frontera Eficiente", line=dict(color="#38BDF8", width=2.8),
            ))
        # Activos individuales
        fig_ef.add_trace(go.Scatter(
            x=sigma_ann * 100.0, y=mu_ann * 100.0, mode="markers+text",
            text=tickers, textposition="top center", name="Activos Individuales",
            marker=dict(size=10, color="#94A3B8", line=dict(width=1, color="#F8FAFC")),
        ))
        # Carteras destacadas
        for name, st_obj, col_m, sym in [
            ("Cartera Actual", st_curr, "#F59E0B", "diamond"),
            ("Máximo Sharpe", st_ms, "#10B981", "star"),
            ("Mínima Volatilidad", st_mv, "#818CF8", "circle"),
            ("Risk Parity (ERC)", st_rp, "#A855F7", "square"),
        ]:
            fig_ef.add_trace(go.Scatter(
                x=[st_obj["volatility"] * 100.0], y=[st_obj["return"] * 100.0],
                mode="markers", name=name,
                marker=dict(size=14, color=col_m, symbol=sym, line=dict(width=1.5, color="#0B0F17")),
            ))
        apply_quant_layout(fig_ef, title="Plano Riesgo–Retorno Anualizado & Frontera Eficiente",
                           height=390, xaxis_title="Volatilidad Anualizada σ (%)", yaxis_title="Retorno Esperado μ (%)")
        st.plotly_chart(fig_ef, use_container_width=True)

    with col_rc:
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Bar(
            x=tickers, y=weights * 100.0, name="Peso en Capital (%)",
            marker_color="#38BDF8",
        ))
        fig_rc.add_trace(go.Bar(
            x=tickers, y=st_curr["risk_contrib_pct"] * 100.0, name="Contribución al Riesgo (%)",
            marker_color="#F43F5E",
        ))
        fig_rc.update_layout(barmode="group")
        apply_quant_layout(fig_rc, title="Descomposición de Euler: Peso Capital vs. Riesgo Aportado",
                           height=390, xaxis_title="Activo", yaxis_title="Porcentaje (%)")
        st.plotly_chart(fig_rc, use_container_width=True)

    render_section_header("Tabla Comparativa de Estrategias de Asignación", "Puedes activar cualquiera de estas estrategias directamente en el panel izquierdo")
    df_strat = pd.DataFrame({
        "Estrategia": ["🎯 Cartera Actual", "⚖️ Equiponderada (1/N)", "🛡️ Risk Parity (ERC)", "📉 Mínima Volatilidad", "🚀 Máximo Sharpe"],
        "Retorno Esperado (μ)": [st_curr["return"], st_eq["return"], st_rp["return"], st_mv["return"], st_ms["return"]],
        "Volatilidad Anual (σ)": [st_curr["volatility"], st_eq["volatility"], st_rp["volatility"], st_mv["volatility"], st_ms["volatility"]],
        "Ratio Sharpe Teórico": [st_curr["sharpe"], st_eq["sharpe"], st_rp["sharpe"], st_mv["sharpe"], st_ms["sharpe"]],
        **{t: [weights[i], w_eq[i], w_rp[i], w_mv[i], w_ms[i]] for i, t in enumerate(tickers)},
    })
    fmt_dict = {
        "Retorno Esperado (μ)": "{:+.2%}",
        "Volatilidad Anual (σ)": "{:.2%}",
        "Ratio Sharpe Teórico": "{:.2f}",
        **{t: "{:.1%}" for t in tickers},
    }
    st.dataframe(df_strat.style.format(fmt_dict), use_container_width=True, hide_index=True)

# -------------------------------------------------------------------------
# PESTAÑA 4: DATOS & DIAGNÓSTICO HISTÓRICO
# -------------------------------------------------------------------------
with tab_data:
    render_kpi_grid([
        {
            "label": "Activos en Universo",
            "value": str(len(tickers)),
            "pill_text": asset_class,
            "pill_type": "neu",
            "sub": ", ".join(tickers[:4]) + ("…" if len(tickers) > 4 else ""),
            "accent": QUANT_COLORS["cyan"],
        },
        {
            "label": "Sesiones Históricas",
            "value": f"{len(prices):,}",
            "pill_text": f"{len(prices)/252:.1f} años",
            "pill_type": "pos",
            "sub": f"Desde {prices.index[0].date()}",
            "accent": QUANT_COLORS["indigo"],
        },
        {
            "label": "Último Cierre",
            "value": str(prices.index[-1].date()),
            "pill_text": fuente,
            "pill_type": "neu",
            "sub": "Datos sincronizados",
            "accent": QUANT_COLORS["emerald"],
        },
        {
            "label": "Volatilidad Cartera (σ)",
            "value": f"{st_curr['volatility']:.1%}",
            "pill_text": f"Media indiv. {np.mean(sigma_ann):.1%}",
            "pill_type": "pos" if st_curr["volatility"] < np.mean(sigma_ann) else "neu",
            "sub": f"Beneficio diversif.: {(np.dot(weights, sigma_ann) - st_curr['volatility']):.1%}",
            "accent": QUANT_COLORS["amber"],
        },
    ])

    col_p1, col_p2 = st.columns([1.25, 1.0])
    with col_p1:
        rebased = prices / prices.iloc[0] * 100.0
        fig_hist = go.Figure()
        for idx_t, t in enumerate(tickers):
            fig_hist.add_trace(go.Scatter(
                x=rebased.index, y=rebased[t], mode="lines", name=t,
                line=dict(color=ASSET_PALETTE[idx_t % len(ASSET_PALETTE)], width=2.0),
            ))
        apply_quant_layout(fig_hist, title="Evolución Histórica Comparada (Base 100 = Inicio del Periodo)",
                           height=370, xaxis_title="Fecha", yaxis_title="Índice Base 100")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_p2:
        ewma_df = VolatilityModel.ewma_series(returns, span=ewma_span) * 100.0
        fig_vol = go.Figure()
        for idx_t, t in enumerate(tickers):
            fig_vol.add_trace(go.Scatter(
                x=ewma_df.index, y=ewma_df[t], mode="lines", name=t,
                line=dict(color=ASSET_PALETTE[idx_t % len(ASSET_PALETTE)], width=1.6),
            ))
        apply_quant_layout(fig_vol, title=f"Régimen de Volatilidad Anualizada EWMA (λ={ewma_span}d)",
                           height=370, xaxis_title="Fecha", yaxis_title="Volatilidad Anual (%)")
        st.plotly_chart(fig_vol, use_container_width=True)

    col_d1, col_d2 = st.columns([1.15, 1.0])
    with col_d1:
        render_section_header("Parámetros Estimados por Activo", vol_note)
        df_params = pd.DataFrame({
            "Activo": tickers,
            "Último Precio": S0,
            "Drift Anual (μ)": mu_ann,
            "Volatilidad (σ)": sigma_ann,
            "Peso Cartera": weights,
            "Riesgo Euler": st_curr["risk_contrib_pct"],
        })
        st.dataframe(
            df_params.style.format({
                "Último Precio": "{:,.4f}" if asset_class == "Forex" else "${:,.2f}",
                "Drift Anual (μ)": "{:+.2%}",
                "Volatilidad (σ)": "{:.2%}",
                "Peso Cartera": "{:.1%}",
                "Riesgo Euler": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with col_d2:
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr, x=tickers, y=tickers,
            colorscale="RdBu_r", zmin=-1.0, zmax=1.0,
            text=np.round(corr, 2), texttemplate="%{text:.2f}",
            textfont={"family": "JetBrains Mono, monospace", "size": 12},
        ))
        apply_quant_layout(fig_corr, title=f"Matriz de Correlación ({'Ledoit-Wolf Shrinkage' if corr_opt == 'shrinkage' else 'Histórica'})",
                           height=340)
        st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------------------------------------------------------
# PESTAÑA 5: INSPECTOR POR ACTIVO & GUÍA
# -------------------------------------------------------------------------
with tab_asset:
    col_sel, col_info = st.columns([1, 2.5])
    with col_sel:
        sel = st.selectbox("Seleccionar activo para inspección profunda", R["tickers"], index=0)
    j = R["tickers"].index(sel)
    ap = paths[:, :, j]
    af = ap[:, -1]
    a_init = float(ap[:, 0].mean())
    a_med = float(np.median(af))

    render_kpi_grid([
        {
            "label": f"Spot / Cierre ({sel})",
            "value": f"{a_init:,.4f}" if asset_class == "Forex" else f"${a_init:,.2f}",
            "pill_text": f"Peso {R['weights'][j]:.1%}",
            "pill_type": "neu",
            "sub": f"σ anual: {R['sigma'][j]:.1%}",
            "accent": QUANT_COLORS["cyan"],
        },
        {
            "label": "Mediana Proyectada",
            "value": f"{a_med:,.4f}" if asset_class == "Forex" else f"${a_med:,.2f}",
            "pill_text": f"{(a_med/a_init - 1):+.1%}",
            "pill_type": "pos" if a_med >= a_init else "neg",
            "sub": f"μ anual: {R['mu'][j]:+.1%}",
            "accent": QUANT_COLORS["emerald"],
        },
        {
            "label": "Prob. Caída Individual",
            "value": f"{np.mean(af < a_init):.1%}",
            "pill_text": "Bajista" if np.mean(af < a_init) > 0.5 else "Alcista",
            "pill_type": "warn" if np.mean(af < a_init) > 0.5 else "pos",
            "sub": f"Horizonte {horizon}d",
            "accent": QUANT_COLORS["amber"],
        },
        {
            "label": "VaR 95% Unitario",
            "value": f"{calculate_var(af, a_init, 0.95):,.4f}" if asset_class == "Forex" else f"${calculate_var(af, a_init, 0.95):,.2f}",
            "pill_text": f"-{calculate_var(af, a_init, 0.95)/a_init:.1%}",
            "pill_type": "neg",
            "sub": "Caída en P5",
            "accent": QUANT_COLORS["rose"],
        },
    ])

    if asset_class == "Forex":
        st.info(
            f"📏 **Especificación FX ({sel})**: 1 pip = `{pip_size(sel)}` · "
            f"Un movimiento de 100 pips equivale a `{100 * pip_size(sel):.4f}` en cotización. "
            f"Apalancamiento activo: `{fx_lev}x`."
        )
    elif asset_class in ("Metales", "Futuros") and sel in FUTURES_SPECS:
        s = FUTURES_SPECS[sel]
        st.info(
            f"📦 **Contrato Futuro ({sel})**: {s['label']} · Multiplicador `{s['multiplier']}x` · "
            f"Margen referencial `${s['margin']:,}` · Coste de roll aplicado: `{mt_roll} bps × {mt_rolls_year}/año`."
        )

    c_a1, c_a2 = st.columns([1.3, 1.0])
    with c_a1:
        g_asset = go.Figure()
        for i in range(min(100, len(ap))):
            g_asset.add_trace(go.Scatter(
                y=ap[i], mode="lines",
                line=dict(color="rgba(56, 189, 248, 0.14)", width=0.7),
                showlegend=False, hoverinfo="skip",
            ))
        g_asset.add_trace(go.Scatter(
            y=np.percentile(ap, 95, axis=0), mode="lines", name="P95",
            line=dict(color="#10B981", width=1.6, dash="dot"),
        ))
        g_asset.add_trace(go.Scatter(
            y=np.median(ap, axis=0), mode="lines", name="Mediana",
            line=dict(color="#38BDF8", width=2.6),
        ))
        g_asset.add_trace(go.Scatter(
            y=np.percentile(ap, 5, axis=0), mode="lines", name="P5",
            line=dict(color="#F43F5E", width=1.6, dash="dot"),
        ))
        apply_quant_layout(g_asset, title=f"Abánico y Trayectorias Simuladas · {sel}",
                           height=350, xaxis_title="Días", yaxis_title="Precio")
        st.plotly_chart(g_asset, use_container_width=True)

    with c_a2:
        render_section_header(f"Guía Táctica · {asset_class}", "Recomendaciones de modelado cuantitativo")
        st.markdown(CLASS_TIPS[asset_class])
        st.markdown(
            "> **Nota Metodológica**: El motor Monte Carlo proyecta distribuciones de probabilidad "
            "condicionadas al régimen de volatilidad, colas y estructura de correlación seleccionada. "
            "Úsalo para dimensionar posiciones, límites de pérdida (VaR/CVaR) y asignación óptima."
        )
