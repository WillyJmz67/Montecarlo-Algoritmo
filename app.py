# app.py — MC Desk Local · Interfaz completa multiactivo (Streamlit)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from engine.models import MonteCarloEngine, shrink_correlation
from engine.metrics import (
    calculate_var, calculate_cvar, max_drawdown,
    percentile_table, sharpe_sortino, cagr,
)
from engine.volatility import VolatilityModel
from engine.portfolio import (
    portfolio_paths_from_asset_paths, apply_generic_costs, apply_contributions,
)
from engine.forex import pip_size, apply_forex_costs
from engine.metals import FUTURES_SPECS, apply_futures_roll
from engine.universe import ASSET_CLASSES, CLASS_TIPS
from engine.data_loader import fetch_yahoo, load_csv

st.set_page_config(page_title="MC Desk Local · Monte Carlo Multiactivo", layout="wide")

# ============ SIDEBAR ============
st.sidebar.title("🎲 MC Desk Local")
st.sidebar.caption("Monte Carlo · Acciones · Índices · Forex · Metales · Futuros")

asset_class = st.sidebar.selectbox("1️⃣ ¿Qué quieres ver?", list(ASSET_CLASSES.keys()), index=0)
cfg_class = ASSET_CLASSES[asset_class]
st.sidebar.info(f"{cfg_class['desc']}\n\n💡 {cfg_class['info']}")

# --- Universo ---
st.sidebar.subheader("2️⃣ Universo de activos")
preset = st.sidebar.multiselect(
    "Tickers sugeridos",
    cfg_class["tickers"],
    default=[t for t in cfg_class["defaults"] if t in cfg_class["tickers"]],
)
custom = st.sidebar.text_input("Añadir tickers (coma)", placeholder="Ej: AAPL, EURUSD=X, GC=F")
extra = [t.strip().upper() for t in custom.split(",") if t.strip()] if custom else []
tickers = sorted(set(preset + extra))
if not tickers:
    st.sidebar.warning("Elige al menos 1 ticker.")
    st.stop()

fuente = st.sidebar.radio("Fuente de datos", ["Yahoo Finance", "Subir CSV"], horizontal=True)
up_prices = None
if fuente == "Subir CSV":
    up = st.sidebar.file_uploader("CSV (1ª col = fecha, resto = precios)", type=["csv"])
    if up is not None:
        try:
            up_prices = load_csv(up)
            tickers = [c for c in up_prices.columns if c in tickers] or list(up_prices.columns)
            st.sidebar.success(f"CSV cargado: {len(up_prices)} filas · {len(tickers)} activos.")
        except Exception as e:
            st.sidebar.error(f"CSV inválido: {e}")

col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("Desde", value=pd.to_datetime("2015-01-01"))
end_date = col_d2.date_input("Hasta", value=pd.Timestamp.today())
auto_adj = st.sidebar.checkbox("Precios ajustados (dividendos/splits)", value=True) if asset_class == "Acciones / ETFs" else True

# --- Pesos ---
st.sidebar.subheader("3️⃣ Cartera (pesos %)")
weights = []
for t in tickers:
    w = st.sidebar.slider(t, 0, 100, int(100 / len(tickers)) if len(tickers) else 0, key=f"w_{t}_{asset_class}")
    weights.append(w)
weights = np.array(weights, dtype=float)
if weights.sum() <= 0:
    weights = np.ones(len(tickers)) / len(tickers)
else:
    weights = weights / weights.sum()
st.sidebar.caption(" · ".join(f"{t}: {w:.0%}" for t, w in zip(tickers, weights)))

rebalance = st.sidebar.selectbox("Rebalanceo", ["none", "monthly", "quarterly", "weekly", "daily"],
                                 format_func=lambda x: {"none": "Sin rebalanceo (buy & hold)", "monthly": "Mensual",
                                                        "quarterly": "Trimestral", "weekly": "Semanal", "daily": "Diario"}[x])

# --- Modelo ---
st.sidebar.subheader("4️⃣ Modelo Monte Carlo")
model = st.sidebar.selectbox("Motor", ["gbm", "student_t", "historical", "block_bootstrap"],
                             format_func=lambda x: {"gbm": "GBM paramétrico (normal)",
                                                    "student_t": "Student-t (colas pesadas)",
                                                    "historical": "Bootstrap histórico",
                                                    "block_bootstrap": "Bootstrap por bloques"}[x])
n_paths = st.sidebar.number_input("Simulaciones", 500, 100_000, 10_000, step=1_000)
horizon = st.sidebar.number_input("Horizonte (días hábiles)", 10, 1260, 252, step=21)
seed = st.sidebar.number_input("Semilla (reproducible)", 0, 99999, 42)
antithetic = st.sidebar.checkbox("Antithetic variates (menos varianza)", value=True) if model in ("gbm", "student_t") else False
df_t = st.sidebar.slider("Grados libertad t (menor = colas más gordas)", 3, 15, 5) if model == "student_t" else 5
block_size = st.sidebar.slider("Tamaño bloque (días)", 5, 60, 20) if model == "block_bootstrap" else 20

drift_opt = st.sidebar.selectbox("Drift μ", ["historico", "cero", "personalizado"],
                                 help="Histórico = media pasada. Cero = solo riesgo (recomendado en forex).")
drift_custom = st.sidebar.number_input("Drift anual personalizado %", -20.0, 30.0, 5.0, step=0.5) / 100 if drift_opt == "personalizado" else None
if asset_class == "Forex" and drift_opt == "historico":
    st.sidebar.caption("⚠️ En forex el drift histórico suele ser ruido: considera 'cero'.")

vol_opt = st.sidebar.selectbox("Volatilidad σ", ["historica", "ewma", "garch"],
                               help="EWMA pondera lo reciente. GARCH modela clusters de volatilidad.")
ewma_span = st.sidebar.slider("Ventana EWMA", 20, 180, 60) if vol_opt == "ewma" else 60
corr_opt = st.sidebar.radio("Correlación", ["historica", "shrinkage (Ledoit-Wolf)"], horizontal=False)

# --- Ajustes por clase ---
st.sidebar.subheader(f"5️⃣ Ajustes · {asset_class}")
commission_bps = st.sidebar.number_input("Comisión (bps/op)", 0.0, 100.0, 5.0)
slippage_bps = st.sidebar.number_input("Slippage (bps/op)", 0.0, 100.0, 2.0)
monthly_contrib = st.sidebar.number_input("Aportación mensual ($)", 0.0, 100000.0, 0.0, step=100.0)
risk_free = st.sidebar.number_input("Tasa libre riesgo % (Sharpe)", 0.0, 10.0, 0.0) / 100
target_ret = st.sidebar.number_input("Objetivo rentabilidad %", -50.0, 300.0, 10.0) / 100

fx_spread = fx_swap = fx_lev = None
mt_roll = mt_rolls_year = None
if asset_class == "Forex":
    fx_spread = st.sidebar.number_input("Spread FX (bps)", 0.0, 50.0, 1.0)
    fx_swap = st.sidebar.number_input("Swap diario (bps, + = coste)", -10.0, 10.0, 0.0)
    fx_lev = st.sidebar.number_input("Apalancamiento (1 = contado)", 1.0, 500.0, 1.0)
    with st.sidebar.expander("📏 Pips por par"):
        for t in tickers:
            st.write(f"{t}: pip = {pip_size(t)}")
if asset_class in ("Metales", "Futuros"):
    mt_roll = st.sidebar.number_input("Coste roll por vencimiento (bps)", 0.0, 100.0, 2.0)
    mt_rolls_year = st.sidebar.number_input("Rolls / año", 0.0, 12.0, 6.0)
    with st.sidebar.expander("📦 Contratos"):
        for t in tickers:
            s = FUTURES_SPECS.get(t)
            st.write(f"{t}: {s['label'] + ' · mult ' + str(s['multiplier']) if s else 'spot/acción (mult 1)'}")
if asset_class == "Acciones / ETFs":
    expense_bps = st.sidebar.number_input("Expense ratio ETF (bps/año)", 0.0, 200.0, 5.0)
else:
    expense_bps = 0.0

run = st.sidebar.button("🚀 Ejecutar simulación", type="primary", use_container_width=True)

# ============ DATOS ============
st.title(f"🎲 Monte Carlo · {asset_class}")
st.caption(f"{cfg_class['desc']}  ·  Motor: {model}  ·  {len(tickers)} activos  ·  Horizonte: {horizon}d  ·  Sims: {n_paths:,}")

@st.cache_data(show_spinner=False)
def get_prices(tickers_key, start, end, auto_adj, fuente_nombre):
    if fuente_nombre == "Subir CSV":
        return None
    return fetch_yahoo(list(tickers_key), str(start), str(end), auto_adjust=auto_adj)

prices = up_prices if (fuente == "Subir CSV" and up_prices is not None) else get_prices(tuple(tickers), str(start_date), str(end_date), auto_adj, fuente)
if prices is None or (isinstance(prices, pd.DataFrame) and prices.empty):
    st.error("Sin datos. Revisa tickers, fechas, conexión o sube un CSV.")
    st.stop()
# Alinear columnas pedidas
avail = [t for t in tickers if t in (prices.columns if isinstance(prices, pd.DataFrame) else [])]
missing = [t for t in tickers if t not in avail]
if missing:
    st.warning(f"Sin datos para: {', '.join(missing)} (se excluyen). Yahoo usa p.ej. `^GSPC`, `EURUSD=X`, `GC=F`, `ES=F`.")
if not avail:
    st.stop()
prices = prices[avail].dropna(how="all").ffill().dropna()
tickers, weights = avail, np.array([weights[tickers.index(t)] for t in avail], dtype=float)
weights = weights / weights.sum()
S0 = prices.iloc[-1].values.astype(float)
returns = prices.pct_change().dropna()
if len(returns) < 60:
    st.error("Muy poco historial (<60 días). Amplía el rango de fechas.")
    st.stop()

# Parametros mu / sigma
if drift_opt == "cero":
    mu_ann = np.zeros(len(tickers))
elif drift_opt == "personalizado":
    mu_ann = np.full(len(tickers), float(drift_custom))
else:
    mu_ann = (returns.mean() * 252).values

if vol_opt == "historica":
    sigma_ann = (returns.std() * np.sqrt(252)).values
    vol_note = "Volatilidad histórica simple."
elif vol_opt == "ewma":
    sigma_ann = VolatilityModel.ewma(returns, span=ewma_span).values
    vol_note = f"EWMA span={ewma_span} (más peso a lo reciente)."
else:
    sigma_ann, garch_info = [], {}
    for c in returns.columns:
        try:
            r = VolatilityModel.garch_last(returns[c])
            sigma_ann.append(r["vol_annual"])
            garch_info[c] = r
        except Exception as e:
            sigma_ann.append(float(returns[c].std() * np.sqrt(252)))
            garch_info[c] = {"vol_annual": float(returns[c].std() * np.sqrt(252)), "fallback": True, "error": str(e)}
    sigma_ann = np.array(sigma_ann)
    vol_note = "GARCH(1,1)-t por activo (última vol condicional)."

corr = shrink_correlation(returns) if corr_opt.startswith("shrinkage") else returns.corr().values

# ============ TABS SUPERIORES: datos siempre visibles ============
tab_data, tab_sim, tab_asset, tab_risk, tab_help = st.tabs(
    ["📊 Datos & diagnóstico", "🔮 Simulación cartera", "🎯 Activo individual", "⚠️ Riesgo & decisiones", "❓ Ayuda por clase"])

with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activos", len(tickers))
    c2.metric("Días historia", f"{len(prices):,}")
    c3.metric("Último dato", str(prices.index[-1].date()) if len(prices) else "-")
    c4.metric("Vol media anual", f"{np.mean(sigma_ann):.1%}")
    st.line_chart(prices / prices.iloc[0] * 100, height=260)
    st.caption("Precios rebaseados a 100 · permite comparar trayectorias históricas.")
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Parámetros estimados (anualizados)")
        st.dataframe(pd.DataFrame({"Precio": S0, "μ anual": mu_ann, "σ anual": sigma_ann,
                                   "Peso": weights}, index=tickers).style.format({"Precio": "{:.2f}", "μ anual": "{:.2%}", "σ anual": "{:.2%}", "Peso": "{:.1%}"}))
        st.caption(vol_note)
    with d2:
        st.subheader("Correlación histórica")
        fig_c = px.imshow(corr, x=tickers, y=tickers, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f")
        fig_c.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_c, use_container_width=True)
    with st.expander("💡 Qué mirar aquí"):
        st.markdown("- **σ alta** = más dispersión futura. **Correlación ≈1** = poca diversificación.\n"
                    "- En **forex**, μ debería rondar 0; si ves μ ±5% anual es casi seguro ruido.\n"
                    "- En **metales/futuros**, revisa si el histórico incluye un shock (2020, 2022): condiciona todo.")

with tab_help:
    st.subheader(f"Guía · {asset_class}")
    st.markdown(CLASS_TIPS[asset_class])
    st.markdown(f"> Monte Carlo **no predice**: genera escenarios compatibles con tus supuestos (μ, σ, correlación, costes). "
                "Úsalo para dimensionar riesgo, no como señal de compra/venta.")

# ============ SIMULACION ============
if not run and "res" not in st.session_state:
    st.info("👈 Configura a la izquierda y pulsa **🚀 Ejecutar simulación**.")
    st.stop()
if run:
    eng = MonteCarloEngine(seed=int(seed))
    with st.spinner(f"Simulando {n_paths:,} trayectorias × {horizon}d × {len(tickers)} activos…"):
        if model == "gbm":
            paths = eng.simulate_gbm(S0, mu_ann, sigma_ann, corr, int(horizon), int(n_paths), antithetic=antithetic)
        elif model == "student_t":
            paths = eng.simulate_student_t(S0, mu_ann, sigma_ann, corr, int(horizon), int(n_paths), df=float(df_t), antithetic=antithetic)
        elif model == "block_bootstrap":
            paths = eng.simulate_block_bootstrap(S0, returns, int(horizon), int(n_paths), block_size=int(block_size))
        else:
            paths = eng.simulate_historical_bootstrap(S0, returns, int(horizon), int(n_paths))
        # Ajustes especificos
        if asset_class in ("Metales", "Futuros") and mt_roll:
            paths = apply_futures_roll(paths, roll_bps=float(mt_roll), rolls_per_year=float(mt_rolls_year), days=int(horizon))
        port = portfolio_paths_from_asset_paths(paths, weights, rebalance=rebalance)
        # Escalar a valor monetario: base 100 → usa S0 ponderado como referencia
        ref_price = float(np.sum(S0 * weights))
        port_val = port * ref_price
        # Costes
        n_rb = 0 if rebalance == "none" else max(1, int(horizon / {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63}[rebalance]))
        port_val = apply_generic_costs(port_val, commission_bps, slippage_bps, n_rb)
        if expense_bps:
            port_val = port_val * (1 - expense_bps / 10_000.0) ** (horizon / 252)
        if asset_class == "Forex" and fx_spread is not None:
            port_val = apply_forex_costs(port_val, spread_bps=fx_spread, swap_bps_dia=fx_swap or 0.0, days=int(horizon), leverage=float(fx_lev or 1.0))
        if monthly_contrib:
            port_val = apply_contributions(port_val, monthly_contrib, ref_price)
    st.session_state["res"] = dict(paths=paths, port=port_val, ref=ref_price,
                                   tickers=list(tickers), weights=np.array(weights),
                                   mu=mu_ann.copy(), sigma=sigma_ann.copy(), corr=np.array(corr))

R = st.session_state["res"]
paths, port, ref = R["paths"], R["port"], R["ref"]
final = port[:, -1]
init = float(port[:, 0].mean())
years = max(horizon / 252, 1e-6)
p_loss = float(np.mean(final < init))
p_target = float(np.mean(final >= init * (1 + target_ret)))
med = float(np.median(final)); mean = float(final.mean())
v95 = calculate_var(final, init, 0.95); c95 = calculate_cvar(final, init, 0.95)
v99 = calculate_var(final, init, 0.99); c99 = calculate_cvar(final, init, 0.99)
mdd = max_drawdown(port)
sharpe, sortino = sharpe_sortino(port, risk_free)
q, qv, qr = percentile_table(final, init)

with tab_sim:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Valor inicial", f"${init:,.2f}")
    k2.metric("Mediana final", f"${med:,.2f}", f"{(med/init-1):+.1%}")
    k3.metric("Media final", f"${mean:,.2f}")
    k4.metric("P(pérdida)", f"{p_loss:.1%}")
    k5.metric(f"P(≥ +{target_ret:.0%})", f"{p_target:.1%}")
    k6.metric("CAGR mediana", f"{cagr(init, med, years):.1%}")
    st.caption(f"Base monetaria = Σ S0·w = ${ref:,.2f} · Rebalanceo: {rebalance} · Costes aplicados: comisión {commission_bps}bps + slippage {slippage_bps}bps"
               + (f" · spread FX {fx_spread}bps · swap {fx_swap}bps/d · lev {fx_lev}x" if asset_class == "Forex" else "")
               + (f" · roll {mt_roll}bps × {mt_rolls_year}/año" if asset_class in ("Metales", "Futuros") else ""))

    # Trayectorias + abanico
    g1, g2 = st.columns([3, 2])
    with g1:
        fig = go.Figure()
        for i in range(min(150, len(port))):
            fig.add_trace(go.Scatter(y=port[i], mode="lines", line=dict(width=0.6), opacity=0.25, showlegend=False, hoverinfo="skip"))
        # Mediana
        fig.add_trace(go.Scatter(y=np.median(port, axis=0), mode="lines", name="Mediana", line=dict(width=2.5)))
        fig.update_layout(title=f"Trayectorias cartera ({asset_class}) — muestra 150", xaxis_title="Días", yaxis_title="$", height=380,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        bands = {p: np.percentile(port, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)}
        x = np.arange(port.shape[1])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=bands[95], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig2.add_trace(go.Scatter(x=x, y=bands[5], mode="lines", line=dict(width=0), fill="tonexty", name="P5–P95", opacity=0.35))
        fig2.add_trace(go.Scatter(x=x, y=bands[90], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig2.add_trace(go.Scatter(x=x, y=bands[10], mode="lines", line=dict(width=0), fill="tonexty", name="P10–P90", opacity=0.45))
        fig2.add_trace(go.Scatter(x=x, y=bands[75], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig2.add_trace(go.Scatter(x=x, y=bands[25], mode="lines", line=dict(width=0), fill="tonexty", name="P25–P75", opacity=0.55))
        fig2.add_trace(go.Scatter(x=x, y=bands[50], mode="lines", name="Mediana", line=dict(width=2)))
        fig2.update_layout(title="Abanico de percentiles", xaxis_title="Días", yaxis_title="$", height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    h1, h2 = st.columns(2)
    with h1:
        fig3 = go.Figure(go.Histogram(x=final, nbinsx=80, name="Valor final"))
        fig3.add_vline(x=init, line_dash="dash", line_color="red", annotation_text="Inicial")
        fig3.add_vline(x=med, line_dash="dot", line_color="green", annotation_text="Mediana")
        fig3.update_layout(title="Distribución del valor final", xaxis_title="$", height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    with h2:
        fig4 = go.Figure(go.Histogram(x=mdd * 100, nbinsx=60, name="MaxDD"))
        fig4.update_layout(title="Max Drawdown por simulación (%)", xaxis_title="%", height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig4, use_container_width=True)
    st.download_button("📥 Descargar valores finales (CSV)",
                       pd.DataFrame({"final": final, "max_dd_pct": mdd * 100}).to_csv(index=False),
                       "mc_resultados.csv", "text/csv")

with tab_asset:
    sel = st.selectbox("Activo a inspeccionar", R["tickers"], index=0)
    j = R["tickers"].index(sel)
    ap = paths[:, :, j]
    af = ap[:, -1]
    a_init = float(ap[:, 0].mean())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{sel} actual", f"{a_init:,.2f}" + (" pips" if False else ""))
    c2.metric("Mediana final", f"{np.median(af):,.2f}", f"{(np.median(af)/a_init-1):+.1%}")
    c3.metric("P(pérdida)", f"{np.mean(af < a_init):.1%}")
    c4.metric(f"VaR 95%", f"{calculate_var(af, a_init, .95):,.2f}")
    if asset_class == "Forex":
        st.info(f"pip = {pip_size(sel)} · Un movimiento de 100 pips ≈ {100 * pip_size(sel):.4f} en precio. "
                f"Con apalancamiento {fx_lev}x, el P&L se multiplica pero también el drawdown.")
    if asset_class in ("Metales", "Futuros") and sel in FUTURES_SPECS:
        s = FUTURES_SPECS[sel]
        st.info(f"📦 {s['label']} · multiplicador {s['multiplier']} · margen aprox ${s['margin']:,} · roll {mt_roll}bps × {mt_rolls_year}/año aplicado.")
    if asset_class == "Indices" and sel.startswith("^"):
        st.warning("Los tickers `^...` son índices no invertibles: úsalos como referencia y opera su ETF (SPY/QQQ/DIA) para la cartera real.")
    g = go.Figure()
    for i in range(min(100, len(ap))):
        g.add_trace(go.Scatter(y=ap[i], mode="lines", line=dict(width=0.6), opacity=0.25, showlegend=False, hoverinfo="skip"))
    g.add_trace(go.Scatter(y=np.median(ap, axis=0), mode="lines", name="Mediana", line=dict(width=2.5)))
    g.update_layout(title=f"Trayectorias · {sel}", xaxis_title="Días", height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(g, use_container_width=True)
    st.dataframe(pd.DataFrame({"Histórico": prices[sel].tail(5).values}, index=prices.index[-5:].date))

with tab_risk:
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("VaR 95%", f"${v95:,.2f}", help="Con 95% de confianza, la pérdida no supera este valor bajo tus supuestos.")
    r2.metric("CVaR 95%", f"${c95:,.2f}", help="Pérdida media en el 5% peor de los casos.")
    r3.metric("VaR 99%", f"${v99:,.2f}")
    r4.metric("CVaR 99%", f"${c99:,.2f}")
    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Tabla de percentiles")
        st.dataframe(pd.DataFrame({"Percentil": [f"P{q_i}" for q_i in q], "Valor $": qv,
                                   "Retorno": qr / 100}).style.format({"Valor $": "${:,.2f}", "Retorno": "{:.1%}"}),
                     use_container_width=True)
        st.caption(f"Sharpe {sharpe:.2f} · Sortino {sortino:.2f} (rf {risk_free:.1%}) · MaxDD mediano {np.median(mdd):.1%}, peor 1% {np.percentile(mdd, 1):.1%}")
    with t2:
        st.subheader("🧠 Lectura para decidir")
        if p_loss > 0.5:
            st.error(f"🔴 P(pérdida) {p_loss:.0%}: el escenario central es negativo con estos supuestos. Revisa drift, costes y apalancamiento.")
        elif v95 > init * 0.2:
            st.warning(f"🟡 Riesgo de cola alto: puedes perder >20% en el 5% peor ({v95:,.0f}$). Reduce tamaño, apalancamiento o aumenta diversificación.")
        else:
            st.success(f"🟢 Riesgo contenido bajo tus supuestos: P(pérdida) {p_loss:.0%}, VaR95 ${v95:,.0f}. Aun así el 1% peor pierde ${v99:,.0f}.")
        st.markdown(f"- **Si tu objetivo es +{target_ret:.0%}**: lo alcanzas en el **{p_target:.0%}** de escenarios.\n"
                    f"- **Drawdown típico**: mediana {np.median(mdd):.1%}. Pregúntate: ¿aguanto el peor 5% ({np.percentile(mdd,5):.1%}) sin vender?\n"
                    "- **No es predicción**: si cambias σ, drift o correlación, estas cifras cambian. Haz sensibilidad (repite con σ ±20%).")
        st.markdown("> Consejo por clase: " + {"Acciones / ETFs": "compara 100% acciones vs 60/40 con AGG para ver el coste de la seguridad.",
            "Indices": "un índice solo concentra; mezcla regiones (USA/Europa/Japón) para bajar correlación.",
            "Forex": "baja el apalancamiento a 1–5x si el VaR te incomoda; el swap mata en horizontes largos.",
            "Metales": "el oro diversifica en crisis pero el futuro con roll rinde menos que el spot en contango.",
            "Futuros": "dimensiona por margen: nº contratos = capital_riesgo / (margen × lev)."}[asset_class])
