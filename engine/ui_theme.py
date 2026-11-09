# engine/ui_theme.py
"""Sistema de diseño visual institucional (Quant Dark Terminal) y temas Plotly para MC Desk Local."""
import streamlit as st
import plotly.graph_objects as go

QUANT_COLORS = {
    "cyan": "#38BDF8",
    "indigo": "#818CF8",
    "emerald": "#10B981",
    "amber": "#F59E0B",
    "rose": "#F43F5E",
    "violet": "#A855F7",
    "slate": "#94A3B8",
    "bg_card": "#111827",
    "bg_main": "#0B0F17",
    "border": "rgba(148, 163, 184, 0.15)",
}

ASSET_PALETTE = [
    "#38BDF8", "#818CF8", "#10B981", "#F59E0B", "#F43F5E",
    "#A855F7", "#2DD4BF", "#FB923C", "#60A5FA", "#E879F9",
]


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Reducir espacio superior genérico de Streamlit */
        .block-container {
            padding-top: 1.6rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1480px !important;
        }

        /* Sidebar institucional */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0D1322 0%, #090D16 100%) !important;
            border-right: 1px solid rgba(148, 163, 184, 0.12) !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem !important;
        }

        /* Brand Header en Sidebar */
        .sb-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 14px;
            margin-bottom: 14px;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(129, 140, 248, 0.06) 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 12px;
        }
        .sb-brand-icon {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 16px;
            color: #FFFFFF;
            box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3);
            font-family: 'JetBrains Mono', monospace;
        }
        .sb-brand-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }
        .sb-brand-sub {
            font-size: 0.72rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 2px;
        }

        /* Hero Header Principal */
        .quant-hero {
            background: radial-gradient(circle at top right, rgba(56, 189, 248, 0.14), transparent 45%),
                        radial-gradient(circle at bottom left, rgba(129, 140, 248, 0.10), transparent 45%),
                        linear-gradient(180deg, rgba(17, 24, 39, 0.92) 0%, rgba(11, 15, 23, 0.95) 100%);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 16px;
            padding: 20px 26px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        }
        .quant-hero-left h1 {
            margin: 0 !important;
            padding: 0 !important;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            color: #F8FAFC !important;
            letter-spacing: -0.025em !important;
        }
        .quant-hero-desc {
            color: #94A3B8;
            font-size: 0.88rem;
            margin-top: 5px;
        }
        .quant-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .q-badge {
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.22);
            color: #E2E8F0;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .q-badge.cyan {
            background: rgba(56, 189, 248, 0.12);
            border-color: rgba(56, 189, 248, 0.35);
            color: #38BDF8;
        }
        .q-badge.indigo {
            background: rgba(129, 140, 248, 0.12);
            border-color: rgba(129, 140, 248, 0.35);
            color: #A5B4FC;
        }
        .q-badge.emerald {
            background: rgba(16, 185, 129, 0.12);
            border-color: rgba(16, 185, 129, 0.35);
            color: #34D399;
        }

        /* Grid de Tarjetas KPI */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
            gap: 14px;
            margin-bottom: 18px;
        }
        .kpi-card {
            background: linear-gradient(180deg, rgba(22, 31, 48, 0.88) 0%, rgba(15, 22, 34, 0.92) 100%);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 13px;
            padding: 15px 17px;
            position: relative;
            overflow: hidden;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .kpi-card:hover {
            border-color: rgba(56, 189, 248, 0.4);
            transform: translateY(-2px);
        }
        .kpi-card::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: var(--accent, #38BDF8);
            opacity: 0.85;
        }
        .kpi-label {
            font-size: 0.73rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94A3B8;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.38rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.2;
        }
        .kpi-sub {
            margin-top: 7px;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            display: flex;
            align-items: center;
            gap: 6px;
            color: #94A3B8;
        }
        .kpi-pill {
            padding: 2px 7px;
            border-radius: 5px;
            font-weight: 600;
            font-size: 0.71rem;
        }
        .kpi-pill.pos { background: rgba(16, 185, 129, 0.16); color: #34D399; }
        .kpi-pill.neg { background: rgba(244, 63, 94, 0.16); color: #FB7185; }
        .kpi-pill.warn { background: rgba(245, 158, 11, 0.16); color: #FBBF24; }
        .kpi-pill.neu { background: rgba(148, 163, 184, 0.15); color: #CBD5E1; }

        /* Section Cards / Paneles */
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 14px 0 10px 0;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        }
        .section-title {
            font-size: 1.02rem;
            font-weight: 600;
            color: #F1F5F9;
            letter-spacing: -0.01em;
        }
        .section-caption {
            font-size: 0.78rem;
            color: #64748B;
        }

        /* Semáforo de Diagnóstico Ejecutivo */
        .verdict-box {
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 16px;
            border: 1px solid;
        }
        .verdict-box.emerald {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.11) 0%, rgba(15, 23, 42, 0.9) 100%);
            border-color: rgba(16, 185, 129, 0.35);
        }
        .verdict-box.amber {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.11) 0%, rgba(15, 23, 42, 0.9) 100%);
            border-color: rgba(245, 158, 11, 0.35);
        }
        .verdict-box.rose {
            background: linear-gradient(135deg, rgba(244, 63, 94, 0.12) 0%, rgba(15, 23, 42, 0.9) 100%);
            border-color: rgba(244, 63, 94, 0.4);
        }
        .verdict-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 6px;
        }
        .verdict-desc {
            font-size: 0.86rem;
            color: #CBD5E1;
            line-height: 1.5;
        }

        /* Estilizado de Pestañas (st.tabs) */
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            font-size: 0.86rem !important;
            font-weight: 600 !important;
            padding: 10px 18px !important;
            border-radius: 8px 8px 0 0 !important;
            color: #94A3B8 !important;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            color: #38BDF8 !important;
            background: rgba(56, 189, 248, 0.08) !important;
        }

        /* Botón primario estilizado */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%) !important;
            border: 1px solid rgba(56, 189, 248, 0.4) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em !important;
            border-radius: 10px !important;
            padding: 0.6rem 1rem !important;
            box-shadow: 0 4px 16px rgba(2, 132, 199, 0.35) !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 22px rgba(56, 189, 248, 0.5) !important;
            transform: translateY(-1px) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="sb-brand">
            <div class="sb-brand-icon">MC</div>
            <div>
                <div class="sb-brand-title">MC Desk · Quant</div>
                <div class="sb-brand-sub">Multi-Asset Risk Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_header(asset_class: str, desc: str, model_label: str, vol_label: str,
                       n_assets: int, horizon: int, n_paths: int, capital: float) -> None:
    st.markdown(
        f"""
        <div class="quant-hero">
            <div class="quant-hero-left">
                <h1>Terminal Monte Carlo · {asset_class}</h1>
                <div class="quant-hero-desc">{desc}</div>
            </div>
            <div class="quant-badges">
                <span class="q-badge cyan">⚙️ {model_label}</span>
                <span class="q-badge indigo">📈 Vol: {vol_label}</span>
                <span class="q-badge">{n_assets} Activos</span>
                <span class="q-badge">⏱️ {horizon}d ({horizon/252:.1f}a)</span>
                <span class="q-badge">{n_paths:,} Trayectorias</span>
                <span class="q-badge emerald">💰 Capital Base: ${capital:,.0f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_grid(cards: list[dict]) -> None:
    """Renderiza una fila de tarjetas KPI institucionales.
    Cada dict: {label, value, sub, pill_text (opcional), pill_type ('pos'|'neg'|'warn'|'neu'), accent (color hex)}
    """
    html_parts = ['<div class="kpi-grid">']
    for c in cards:
        accent = c.get("accent", QUANT_COLORS["cyan"])
        pill_html = ""
        if c.get("pill_text"):
            p_type = c.get("pill_type", "neu")
            pill_html = f'<span class="kpi-pill {p_type}">{c["pill_text"]}</span>'
        sub_text = c.get("sub", "")
        html_parts.append(
            f'<div class="kpi-card" style="--accent: {accent};">'
            f'<div class="kpi-label">{c["label"]}</div>'
            f'<div class="kpi-value">{c["value"]}</div>'
            f'<div class="kpi-sub">{pill_html}<span>{sub_text}</span></div>'
            f'</div>'
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_section_header(title: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-title">{title}</div>
            <div class="section-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_quant_layout(fig: go.Figure, title: str = "", height: int = 380,
                       xaxis_title: str = "", yaxis_title: str = "",
                       show_legend: bool = True) -> go.Figure:
    """Aplica estilo institucional oscuro uniforme a cualquier gráfico Plotly."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Inter, sans-serif", size=14, color="#E2E8F0"),
            x=0.01,
            y=0.97,
        ) if title else None,
        height=height,
        paper_bgcolor="rgba(17, 24, 39, 0.55)",
        plot_bgcolor="rgba(11, 15, 23, 0.45)",
        font=dict(family="Inter, sans-serif", size=12, color="#94A3B8"),
        margin=dict(l=14, r=14, t=44 if title else 20, b=14),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0F172A",
            bordercolor="rgba(56, 189, 248, 0.4)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#F8FAFC"),
        ),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#CBD5E1"),
        ),
        xaxis=dict(
            title=xaxis_title,
            gridcolor="rgba(148, 163, 184, 0.08)",
            zerolinecolor="rgba(148, 163, 184, 0.15)",
            tickfont=dict(family="JetBrains Mono, monospace", size=10),
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor="rgba(148, 163, 184, 0.08)",
            zerolinecolor="rgba(148, 163, 184, 0.15)",
            tickfont=dict(family="JetBrains Mono, monospace", size=10),
        ),
    )
    return fig
