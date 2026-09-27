# MC Desk Local

**Monte Carlo Multiactivo Local** — Aplicación de escritorio auto-hospedada para simulación Monte Carlo de acciones, ETFs, divisas (Forex) y metales/commodities. Diseñada para inspeccionar riesgo, correlaciones y escenarios futuros con total precisión y personalización.

---

## 📌 Resumen

MC Desk Local permite:

- **Simular trayectorias de precios** para múltiples activos simultáneamente con correlación preservada.
- **Probar diferentes modelos**: GBM paramétrico, Student-t (colas pesadas), bootstrap histórico y block bootstrap.
- **Ajustar por clase de activo**: comisiones, spread, swap diario, costes de roll en futuros, apalancamiento.
- **Construir y comparar carteras** con pesos fijos, rebalanceo periódico y costes de transacción.
- **Visualizar riesgo** con VaR, CVaR, drawdowns y percentiles.
- **Usar datos reales**: Yahoo Finance, CSV locales o proveedores gratuitos (Stooq, Dukascopy).
- **Ejecutarse completamente localmente** — tus datos nunca abandonan tu máquina.

> **Fase actual**: MVP funcional con interfaz Streamlit. Motor Monte Carlo multiactivo validado (Fase 1). Próximamente: GARCH, quasi-Monte Carlo, persistencia de proyectos y frontend profesional (Fase 3).

---

## 🚀 Características

| Característica | Descripción |
| :--- | :--- |
| **Motor Monte Carlo** | GBM, Student-t, bootstrap histórico, block bootstrap. Reducción de varianza con variates antitéticos. |
| **Clases de activo** | Acciones/ETF, Índices, Forex (con spread/swap/lev), Metales/Futuros (roll, multiplicador, margen). |
| **Modelos de volatilidad** | Histórica, EWMA, GARCH(1,1) t-student. |
| **Correlación** | Matriz histórica directa o shrinkage Ledoit-Wolf. |
| **Costes y fricciones** | Comisión, slippage, spread, expense ratio, swap diario, roll yield. |
| **Rebalanceo** | Ninguno, diario, semanal, mensual, trimestral. |
| **Aportaciones/retiros** | Monto mensual fijo sobre la cartera. |
| **Importador de datos** | Yahoo Finance (`yfinance`), CSV local (Stooq, HistData, export Duckascopy). |
| **Visualizaciones** | Trayectorias, histogramas de valor final, abanicos de percentiles, drawdowns, heatmap correlación. |
| **Reproducibilidad** | Semilla configurable, resultados idénticos con misma semilla. |
| **Exportación** | CSV con valores finales, max drawdown, percentiles. |
| **Docker** | `docker compose up` levanta todo el stack (backend FastAPI + frontend + Redis). |

---

## 📦 Instalación

### Requisitos previos

- Python 3.11+ (recomendado en entorno virtual).
- `git` (opcional, para clonar).
- Docker & Docker Compose (opcional, para stack completo).

### 1. Clonar y entrar

```powershell
git clone https://github.com/tu-usuario/mc-desk-local.git
cd mc-desk-local
```

### 2. Entorno virtual (recomendado)

```powershell
# Crear y activar
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Verificar instalación

```powershell
python -c "import streamlit, yfinance, numpy, pandas, plotly, arch, sklearn; print('Todas las dependencias OK')"
```

### 4. Ejecutar la app

```powershell
python -m streamlit run app.py
```

Abre en tu navegador: `http://localhost:8501`

### 5. (Opcional) Stack completo con Docker

```powershell
docker compose up --build
```

- **Backend API**: `http://localhost:8000` (docs en `/docs`)
- **Frontend**: `http://localhost:3000`
- **Redis**: `localhost:6379`

---

## 🎮 Uso rápido (primeros pasos)

1. **Elegir clase de activo**: En el sidebar, selecciona `Acciones / ETFs`, `Forex`, `Metales` o `Futuros`.
2. **Seleccionar tickers**: Usa los sugeridos o escribe símbolos personalizados (ver tabla de mapeo abajo).
3. **Ajustar pesos**: Arrastra el slider para cada activo; la suma auto-normaliza a 100%.
4. **Configurar el modelo**:
   - *Modelo*: `Student-t` recommended (mejores colas).
   - *Volatilidad*: `EWMA` si quieres más peso a lo reciente.
   - *Drift*: `cero` para forex; `histórico` para acciones.
5. **Pulsar "🚀 Ejecutar simulación"**.
6. **Inspeccionar resultados**:
   - Pestaña `📊 Datos & diagnóstico` → ver precios, μ, σ, correlación.
   - Pestaña `🔮 Simulación cartera` → trayectorias, abanico de riesgo.
   - Pestaña `⚠️ Riesgo & decisiones` → VaR, CVaR, probabilidad de pérdida, semáforo de lectura.

### Tabla de mapeo de símbolos

| Activo | Yahoo/Fin | Stooq | Ejemplo app |
| :--- | :--- | :--- | :--- |
| **AAPL** | `AAPL` | `aapl.us` | `AAPL` (Yahoo funciona por defecto) |
| **SPY** | `SPY` | `spy.us` | `SPY` |
| **^GSPC** | `^GSPC` | `^spx` | Usar `SPY` o `QQQ` en su lugar |
| **EURUSD=X** | `EURUSD=X` | `eurusd` | `EURUSD=X` |
| **USDJPY=X** | `USDJPY=X` | `usdjpy` | `USDJPY=X` |
| **GC=F** | `GC=F` | `gc.f` | `GC=F` |
| **SI=F** | `SI=F` | `si.f` | `SI=F` |
| **XAUUSD=X** | `XAUUSD=X` | `xauusd` | `XAUUSD=X` |

---

## 🗂 Estructura del proyecto

```
mc-desk-local/
├── app.py                  # Interfaz Streamlit (punto de entrada)
├── requirements.txt        # Dependencias Python
├── config.yaml             # Configuración por defecto
├── data/                   # CSVs o Parquet locales (opcional)
│   └── precios_actualizados.csv  (generado por update_csv.py)
│
├── engine/                 # ✅ Núcleo matemático (independiente de UI)
│   ├── __init__.py
│   ├── models.py           # GBM, Student-t, bootstrap, block bootstrap, Cholesky
│   ├── metrics.py          # VaR, CVaR, MaxDD, Sharpe/Sortino, percentiles
│   ├── volatility.py       # Histórica, EWMA, GARCH(1,1)
│   ├── forex.py            # Spread, swap, apalancamiento, conversión de pips
│   ├── metals.py           # Roll yield, multiplicador, margen de futuros
│   ├── portfolio.py        # Construcción de cartera, rebalanceo, costes, aportaciones
│   ├── universe.py         # Presets de tickers por clase + guías por activo
│   └── data_loader.py      # Yahoo Finance + CSV + cache Parquet
│
├── docker-compose.yml      # Levanta backend + frontend + Redis (Fase 3)
├── Dockerfile.backend
├── Dockerfile.frontend
└── README.md              # Este archivo
```

---

## 🛠 Personalización avanzada

### Añadir tus propios tickers

1. Prepara un CSV con formato `Date,col1,col2,…` (1ª columna = fecha, resto = precios de cierre).
2. En la app elige `Subir CSV` y sube el archivo.
3. Los tickers se tomarán automáticamente de los nombres de columna.

### Modificar la configuración por defecto

Edita `config.yaml`:

```yaml
simulation:
  n_paths: 20000      # más sims = más precisión, más tiempo
  horizon_days: 504   # 2 años
  model: student_t    # gbm | student_t | historical | block_bootstrap
  seed: 1234

portfolio:
  base_currency: USD
  weights:
    SPY: 0.40
    QQQ: 0.20
    GLD: 0.15
    EURUSD=X: 0.15
    GC=F: 0.10
  rebalance: monthly

risk:
  var_confidence: 0.95
  target_return: 0.10   # objetivo anual % para calcular P(≥ objetivo)
```

### Agregar un nuevo modelo Monte Carlo

1. Agrega el método a `engine/models.py` (siguiendo el patrón `simulate_*`).
2. Agrega un `elif model == "..."` en `app.py` dentro del bloque `if run:`.
3. La UI ya expone el selector; solo hay que conectar los parámetros.

---

## 📊 Casos de uso típicos

### 1. "¿Cuánto puedo perder en un año con mi cartera 60/40?"

```text
Clase: Acciones / ETFs
Tickers: SPY, QQQ, AGG (añade AGG manualmente o usa CSV)
Pesos: 60% SPY, 20% QQQ, 20% AGG
Modelo: Student-t (df=5)
Costes: comisión 5bps, slippage 2bps
Rebalanceo: mensual
```

Revisa: `VaR 95%`, `CVaR 95%`, `P(pérdida)`, `Max DD mediano`.

### 2. "¿Protege el oro en crisis?"

```text
Clase: Metales
Tickers: GC=F, XAUUSD=X
Pesos: 10% cada uno
Modelo: Student-t (colas gordas = protección en crashes)
```

Fíjate que en la pestaña `🎯 Activo individual` el drawdown es menor que en acciones puras cuando el mercado colapsa.

### 3. "¿Cómo afecta el apalancamiento en Forex?"

```text
Clase: Forex
Tickers: EURUSD=X, GBPUSD=X
Pesos: 50% cada uno
Apalancamiento: 1 (contado) vs 10 (ejemplo teórico)
Spread: 1bps, swap: 0
```

Mueve el slider de apalancamiento y observa cómo el histograma de "Distribución final" se estira y los drawdowns aumentan proporcionalmente.

### 4. "¿Importancia del rebalanceo?"

Crea dos simulaciones idénticas, solo cambia `Rebalanceo: none` vs `monthly`. En la pestaña de trayectorias verás cómo el rebalanceo "endereza" la curva al vender activos que subieron y comprar los que bajaron.

---

## 🔔 Roadmap (próximas versiones)

| Hito | Fecha estimada | Entregable |
| :--- | :--- | :--- |
| **Fase 2 - Precisión** | Ahora | GARCH, Student-t, block bootstrap, antithetic, costes FX/metales |
| **Fase 3 - SaaS Local** | 4-6 semanas | FastAPI + Next.js, Docker Compose, persistencia DuckDB + Redis |
| **Fase 4 - Análisis Avanzado** | Post-launch | Copulas, Markov Switching, optimización de riesgo, backtesting |

---

## 📄 Licencia

Este proyecto es de uso personal/educativo. No constituya recomendación de inversión. El Monte Carlo es una herramienta estadística — sus resultados dependen de los supuestos del modelo y no predicen el futuro.

---

## 🤝 Contribuciones

1. Fork el proyecto.
2. Crea una rama `feature/nombre-feature`.
3. Asegúrate de que `engine/` siga siendo independiente de la UI.
4. Abre un Pull Request con descripción clara de los cambios y pruebas.

---

## 📬 Contacto

¿Problemas, sugerencias o colaboración?

- Abre un Issue en GitHub.
- Revisa `config.yaml` y `engine/universe.py` antes de pedir soporte.
- Para datos históricos: consulta [Stooq](https://stooq.com/db/h/) o usa `update_csv.py` con Yahoo Finance.

---

*Última actualización: 27 Sep 2026 — MVP Streamlit V1 con motor multiactivo completo.*