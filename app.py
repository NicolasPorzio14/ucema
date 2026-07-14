# -*- coding: utf-8 -*-
"""
Dashboard de Cartera — Grupo 8 · Administración de Carteras de Inversión (EFI)
================================================================================
Panel de monitoreo para la cartera IPS del cliente PyME (financiamiento de
expansión con préstamo de $150M ARS, cobertura cambiaria sin acceso a dólar
oficial, capital de trabajo administrado en tres tramos).

Qué hace este panel:
  1) CARTERA HOY        → composición (Objetivo / Tramo / Instrumento), TIR y
                           Duration ponderadas de la cartera, tabla de holdings.
  2) EVOLUCIÓN HISTÓRICA → cómo vino evolucionando la TIR, la Duration y el
                           índice de valorización de la cartera ("para atrás").
  3) SEMANA & PROYECCIÓN → comparación hoy vs. hace 7 días, y una proyección
                           hacia adelante por DEVENGAMIENTO a tasa constante
                           (no es una predicción de precios), cruzada con el
                           cronograma de desembolsos del préstamo y el DSCR.
  4) RIESGO & LÍMITES    → chequeo automático de los límites del IPS: banda de
                           duration objetivo, concentración máxima por emisor
                           corporativo (ON individual) y sublímite Dollar-Linked
                           sobre el total de la cartera.
  5) METODOLOGÍA         → fuentes y supuestos.

Fuente de mercado: Alphacast, dataset 41886 (ONs / Bonos / Soberanos — el mismo
dataset del panel PRO de Renta Fija). Los instrumentos de money-market (caución,
FCI, cuenta remunerada) no cotizan ahí: se modelan con una TNA manual editable.

La cartera de 8 holdings es la composición EXACTA reconciliada de los dos
gráficos de torta del IPS (cartera consolidada + detalle de Objetivo 1): no es
una estimación ni requiere completar instrumentos. El único dato que no sale
de Alphacast es la TNA de la caución/money market, que se carga con un único
número en la parte superior de la app.

Requisitos: streamlit, pandas, numpy, plotly, alphacast, openpyxl
"""

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

try:
    from alphacast import Alphacast
    ALPHACAST_AVAILABLE = True
except Exception:
    ALPHACAST_AVAILABLE = False


# =====================================================================
# Configuración general y tema visual
# =====================================================================

DATASET_ID = 41886  # ONs / Bonos / Soberanos — mismo dataset que el panel PRO

COLORS = {
    "primary": "#1f2a44",
    "accent": "#c8963e",
    "obj1": "#2563eb",
    "obj2": "#ea580c",
    "tramo1": "#0891b2",
    "tramo2": "#7c3aed",
    "tramo3": "#059669",
    "cash": "#94a3b8",
    "cer": "#7c3aed",
    "fija": "#0891b2",
    "dl": "#ea580c",
    "hd": "#2563eb",
    "dual": "#a16207",
    "otro": "#64748b",
    "ok": "#16a34a",
    "warn": "#dc2626",
    "grid": "rgba(120,130,150,0.18)",
}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Helvetica, Arial", size=13, color="#1e293b"),
    title_font=dict(size=16, color=COLORS["primary"]),
    margin=dict(l=50, r=20, t=60, b=45),
    hoverlabel=dict(bgcolor="#1e293b", font=dict(size=12, color="white"), bordercolor="#1e293b"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(255,255,255,0.6)"),
    plot_bgcolor="white",
)


def style_axes(fig: go.Figure, xtitle: str, ytitle: str) -> go.Figure:
    fig.update_xaxes(title=xtitle, showgrid=True, gridcolor=COLORS["grid"],
                      zeroline=False, showline=True, linecolor="#94a3b8", ticks="outside")
    fig.update_yaxes(title=ytitle, showgrid=True, gridcolor=COLORS["grid"],
                      zeroline=False, showline=True, linecolor="#94a3b8", ticks="outside")
    return fig


def watermark(fig: go.Figure, fecha=None, fuente="Alphacast") -> go.Figure:
    txt = f"Fuente: {fuente}"
    if fecha is not None:
        txt += f" · Datos al {pd.Timestamp(fecha).strftime('%d/%m/%Y')}"
    fig.add_annotation(text=txt, xref="paper", yref="paper", x=0, y=-0.18,
                        showarrow=False, font=dict(size=10, color="#94a3b8"), align="left")
    return fig


# =====================================================================
# Clasificación de instrumentos (misma convención que el panel PRO)
# =====================================================================

COUPON_TO_CLASE = {
    "ARS inflation-linked rate": "CER",
    "ARS fixed rate": "Fija",
    "Dollar-linked rate": "DL",
    "Dual (CER Dollar-linked rate)": "Dual",
    "Dual (Fixed or TAMAR rate)": "Dual",
    "Dual (CER or TAMAR rate)": "Dual",
    "ARS floating rate": "Badlar/Pase",
}

CLASE_COLOR = {"CER": COLORS["cer"], "Fija": COLORS["fija"], "DL": COLORS["dl"],
               "Dual": COLORS["dual"], "HD": COLORS["hd"], "Cash": COLORS["cash"],
               "Otro": COLORS["otro"]}


# =====================================================================
# Cartera por defecto — composición EXACTA del IPS (Grupo 8, comité julio 2026)
# =====================================================================
# Reconciliada a partir de los DOS gráficos de "Composición" del IPS (ver el
# docstring de default_portfolio() para el detalle numérico). Los 8 tickers y
# pesos son la cartera completa — no faltan instrumentos por cargar.

def default_portfolio() -> pd.DataFrame:
    """Composición EXACTA de la cartera del Grupo 8, reconciliada a partir de los
    DOS gráficos de torta del IPS (no es una estimación):

      · Slide "Cartera Consolidada ≈ $175.000.000": D31M7 79,1% · D30S6 6,9% ·
        TZXD6 7,0% · TLCQO 2,1% · Cauciones 1,4% (suma visible 96,5%).
      · Slide "Objetivo 1 (Capital de trabajo)": Cauciones 10% · S31L6 10% ·
        TZXD6 50% · TLCQO 15% · LOC5O 7,5% · AO27 7,5% (100% de Objetivo 1).

    D31M7 y D30S6 (Dollar-Linked) son la cobertura cambiaria del préstamo:
    79,1% + 6,9% = 86,0% del total — coincide EXACTO con el "Sublímite 86%
    Dollar-Linked" de la diapositiva de riesgos, y con $150M/$175M ≈ 86%.
    Esos dos instrumentos son, por lo tanto, el 100% de Objetivo 2 → Objetivo 1
    es el 14,0% restante del total. Aplicando ese 14,0% a los pesos del segundo
    gráfico se obtiene el 3,5% que faltaba en el primero (S31L6 1,4% + LOC5O
    1,05% + AO27 1,05% = 3,5%), y los tres instrumentos que sí aparecían en
    ambos gráficos coinciden centavo a centavo (Cauciones 1,4%, TZXD6 7,0%,
    TLCQO 2,1%). Las 8 filas de abajo son, entonces, la cartera COMPLETA —
    no hace falta agregar ni completar ningún instrumento.
    """
    rows = [
        # ---- Objetivo 2 · Cobertura FX del préstamo (86,0% del total) ----
        dict(Ticker="D31M7", Descripcion="Dollar-Linked — cobertura FX del préstamo (venc. ~mar-2027)",
             Objetivo="2 · Cobertura FX Préstamo", Tramo="Cobertura FX (Estructural)",
             Peso_pct=79.1, Es_Cash=False, TNA_Manual_pct=np.nan,
             Segmento_Manual="Sovereign", Clase_Manual="DL"),
        dict(Ticker="D30S6", Descripcion="Dollar-Linked — cobertura FX del préstamo (venc. ~sep-2026)",
             Objetivo="2 · Cobertura FX Préstamo", Tramo="Cobertura FX (Estructural)",
             Peso_pct=6.9, Es_Cash=False, TNA_Manual_pct=np.nan,
             Segmento_Manual="Sovereign", Clase_Manual="DL"),
        # ---- Objetivo 1 · Capital de trabajo (14,0% del total) ----
        dict(Ticker="CAUCION", Descripcion="Caución 1 día / FCI Money Market / Cta. Remunerada",
             Objetivo="1 · Capital de Trabajo", Tramo="1 · Operativo (≤1 mes)",
             Peso_pct=1.4, Es_Cash=True, TNA_Manual_pct=30.0,
             Segmento_Manual="Cash", Clase_Manual="Cash"),
        dict(Ticker="S31L6", Descripcion="LECAP corta",
             Objetivo="1 · Capital de Trabajo", Tramo="1 · Operativo (≤1 mes)",
             Peso_pct=1.4, Es_Cash=False, TNA_Manual_pct=np.nan,
             Segmento_Manual="Sovereign", Clase_Manual="Fija"),
        dict(Ticker="TZXD6", Descripcion="Bono del Tesoro Nacional — CER",
             Objetivo="1 · Capital de Trabajo", Tramo="2 · Táctico (1-12 meses)",
             Peso_pct=7.0, Es_Cash=False, TNA_Manual_pct=np.nan,
             Segmento_Manual="Sovereign", Clase_Manual="CER"),
        dict(Ticker="TLCQO", Descripcion="ON corporativa",
             Objetivo="1 · Capital de Trabajo", Tramo="3 · Estructural (>1 año)",
             Peso_pct=2.1, Es_Cash=False, TNA_Manual_pct=np.nan,
             Segmento_Manual="Corporate", Clase_Manual="HD"),
        dict(Ticker="LOC5O", Descripcion="ON Loma Negra",
             Objetivo="1 · Capital de Trabajo", Tramo="3 · Estructural (>1 año)",
             Peso_pct=1.05, Es_Cash=False, TNA_Manual_pct=np.nan,
             Segmento_Manual="Corporate", Clase_Manual="HD"),
        dict(Ticker="AO27", Descripcion="Bono Soberano",
             Objetivo="1 · Capital de Trabajo", Tramo="3 · Estructural (>1 año)",
             Peso_pct=1.05, Es_Cash=False, TNA_Manual_pct=np.nan,
             Segmento_Manual="Sovereign", Clase_Manual="Otro"),
    ]
    return pd.DataFrame(rows)


PORTFOLIO_COLS = ["Ticker", "Descripcion", "Objetivo", "Tramo", "Peso_pct",
                   "Es_Cash", "TNA_Manual_pct", "Segmento_Manual", "Clase_Manual"]


# =====================================================================
# Descarga y normalización (Alphacast) — mismo criterio que el panel PRO
# =====================================================================

@st.cache_data(show_spinner=False, ttl=15 * 60)
def download_dataset(api_key: str, dataset_id: int) -> pd.DataFrame:
    alphacast = Alphacast(api_key)
    csv_bytes = alphacast.datasets.dataset(int(dataset_id)).download_data(format="csv")
    return pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")))


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el dataset 41886: TIR en %, MD en años, conserva segmento y
    estructura de cupón para clasificar y auditar cada holding."""
    df = df.copy()
    ren = {"symbol": "Ticker", "irr": "TIR", "modified duration": "MD",
           "convexity": "Convexidad", "parity": "Paridad", "residual value": "Valor Residual",
           "market segment": "Segmento", "coupon structure": "CouponStructure",
           "issue currency": "IssueCcy", "trading currency": "TradingCcy",
           "volume": "Volumen", "issuer": "Emisor"}
    df.rename(columns={k: v for k, v in ren.items() if k in df.columns}, inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "TIR" in df.columns:
        df["TIR"] = pd.to_numeric(df["TIR"], errors="coerce") * 100
    if "MD" in df.columns:
        df["MD"] = pd.to_numeric(df["MD"], errors="coerce")
    if "Volumen" in df.columns:
        df["Volumen"] = pd.to_numeric(df["Volumen"], errors="coerce")
    if "Paridad" in df.columns:
        df["Paridad"] = pd.to_numeric(df["Paridad"], errors="coerce")

    wanted = ["Date", "Ticker", "Emisor", "Segmento", "CouponStructure",
              "IssueCcy", "TradingCcy", "TIR", "MD", "Convexidad", "Paridad", "Volumen"]
    df = df[[c for c in wanted if c in df.columns]].copy()
    df["Clase"] = df.get("CouponStructure", pd.Series(dtype=str)).map(COUPON_TO_CLASE).fillna("Otro")
    return df


def snapshot_asof(df_norm: pd.DataFrame, tickers: list, as_of: pd.Timestamp) -> pd.DataFrame:
    """Última fila disponible por ticker con Date <= as_of (maneja instrumentos
    ilíquidos que no cotizan todos los días — no fuerza la fecha exacta)."""
    if df_norm.empty:
        return pd.DataFrame()
    d = df_norm[df_norm["Ticker"].isin(tickers) & (df_norm["Date"] <= pd.Timestamp(as_of))]
    if d.empty:
        return pd.DataFrame()
    idx = d.groupby("Ticker")["Date"].idxmax()
    return d.loc[idx].reset_index(drop=True)


# =====================================================================
# Modo demo — datos sintéticos para ensayar el panel sin API key
# =====================================================================

@st.cache_data(show_spinner=False)
def synthetic_dataset(tickers: list, seed: int = 8, days: int = 420) -> pd.DataFrame:
    """Genera una serie de tiempo plausible (random walk acotado) por ticker,
    SOLO para poder recorrer la interfaz sin conexión a Alphacast. Nunca se usa
    si hay una API key cargada — está pensado para ensayo de la clase."""
    rng = np.random.default_rng(seed)
    hoy = pd.Timestamp(datetime.now().date())
    fechas = pd.bdate_range(end=hoy, periods=days)
    base_tir = {"D31M7": 6.0, "D30S6": 4.0, "S31L6": 32.0, "TZXD6": 24.0,
                "TLCQO": 9.0, "LOC5O": 9.5, "AO27": 12.0}
    base_md = {"D31M7": 0.65, "D30S6": 0.20, "S31L6": 0.10, "TZXD6": 0.55,
               "TLCQO": 1.0, "LOC5O": 1.1, "AO27": 1.7}
    clase_map = {"D31M7": "DL", "D30S6": "DL", "S31L6": "Fija", "TZXD6": "CER",
                 "TLCQO": "HD", "LOC5O": "HD", "AO27": "Otro"}
    rows = []
    for tk in tickers:
        if tk == "CAUCION":
            continue
        tir0 = base_tir.get(tk, 15.0)
        md0 = base_md.get(tk, 1.0)
        tir_walk = tir0 + np.cumsum(rng.normal(0, 0.06, len(fechas)))
        md_walk = np.clip(md0 + np.cumsum(rng.normal(0, 0.004, len(fechas))), 0.05, None)
        paridad_ret = rng.normal(0.0002, 0.003, len(fechas))
        paridad = 100 * np.cumprod(1 + paridad_ret)
        for i, f in enumerate(fechas):
            rows.append(dict(Date=f, Ticker=tk, Emisor=tk, Segmento="Corporate" if tk in
                              ("LOC5O", "TLCQO") else "Sovereign", CouponStructure=None,
                              TradingCcy="ARS", TIR=round(float(tir_walk[i]), 2),
                              MD=round(float(md_walk[i]), 2), Convexidad=np.nan,
                              Paridad=round(float(paridad[i]), 2), Volumen=np.nan,
                              Clase=clase_map.get(tk, "Otro")))
    return pd.DataFrame(rows)


# =====================================================================
# Motor de cartera: snapshot ponderado, KPIs, serie histórica, proyección
# =====================================================================

def build_holdings_snapshot(port_df: pd.DataFrame, df_norm: pd.DataFrame,
                             as_of: pd.Timestamp) -> pd.DataFrame:
    """Combina la cartera (pesos) con el dato de mercado más reciente disponible
    a la fecha 'as_of'. Para filas Es_Cash usa la TNA manual (MD=0). Para
    tickers que no aparecen en el dataset, deja TIR/MD en NaN y lo marca."""
    port = port_df.copy()
    bond_tickers = port.loc[~port["Es_Cash"], "Ticker"].tolist()
    mkt = snapshot_asof(df_norm, bond_tickers, as_of) if bond_tickers else pd.DataFrame()

    out_rows = []
    for _, r in port.iterrows():
        row = r.to_dict()
        if r["Es_Cash"]:
            row["TIR"] = float(r.get("TNA_Manual_pct", np.nan))
            row["MD"] = 0.0
            row["Paridad"] = 100.0
            row["Fecha_Dato"] = pd.Timestamp(as_of)
            row["Encontrado"] = True
            row["Segmento"] = r.get("Segmento_Manual", "Cash")
            row["Clase"] = r.get("Clase_Manual", "Cash")
        else:
            m = mkt[mkt["Ticker"] == r["Ticker"]] if not mkt.empty else pd.DataFrame()
            if not m.empty:
                mm = m.iloc[0]
                row["TIR"] = mm.get("TIR", np.nan)
                row["MD"] = mm.get("MD", np.nan)
                row["Paridad"] = mm.get("Paridad", np.nan)
                row["Fecha_Dato"] = mm.get("Date", pd.NaT)
                row["Encontrado"] = True
                row["Segmento"] = mm.get("Segmento", r.get("Segmento_Manual", "Otro"))
                row["Clase"] = mm.get("Clase", r.get("Clase_Manual", "Otro"))
            else:
                row["TIR"] = np.nan
                row["MD"] = np.nan
                row["Paridad"] = np.nan
                row["Fecha_Dato"] = pd.NaT
                row["Encontrado"] = False
                row["Segmento"] = r.get("Segmento_Manual", "Otro")
                row["Clase"] = r.get("Clase_Manual", "Otro")
        out_rows.append(row)

    d = pd.DataFrame(out_rows)
    d["Peso_frac"] = pd.to_numeric(d["Peso_pct"], errors="coerce") / 100.0
    d["Contrib_TIR"] = d["Peso_frac"] * d["TIR"]
    d["Contrib_MD"] = d["Peso_frac"] * d["MD"]
    return d


def portfolio_kpis(snap: pd.DataFrame) -> dict:
    validos = snap.dropna(subset=["TIR", "MD"])
    peso_valido = validos["Peso_frac"].sum()
    tir_cartera = validos["Contrib_TIR"].sum() / peso_valido if peso_valido > 0 else np.nan
    md_cartera = validos["Contrib_MD"].sum() / peso_valido if peso_valido > 0 else np.nan
    return dict(tir_cartera=tir_cartera, md_cartera=md_cartera,
                peso_cubierto=peso_valido * 100, peso_total=snap["Peso_frac"].sum() * 100,
                n_no_encontrados=int((~snap["Encontrado"]).sum()))


def historical_series(port_df: pd.DataFrame, df_norm: pd.DataFrame,
                       start: pd.Timestamp, end: pd.Timestamp) -> tuple:
    """Serie diaria de TIR y MD ponderados de la cartera + índice de
    valorización (base 100) en [start, end]. Devuelve (df_serie, tickers_sin_dato).
    Los bonos ilíquidos se llevan hacia adelante (ffill) entre ruedas sin cotizar;
    un ticker sin NINGÚN dato en el dataset queda con retorno plano (0%) y se
    reporta aparte para no ensuciar el índice silenciosamente."""
    port = port_df.copy()
    bonos = port.loc[~port["Es_Cash"], "Ticker"].tolist()
    pesos = port.set_index("Ticker")["Peso_pct"] / 100.0

    fechas = pd.bdate_range(start=start, end=end)
    hist = df_norm[(df_norm["Ticker"].isin(bonos)) & (df_norm["Date"] >= start - pd.Timedelta(days=30))
                   & (df_norm["Date"] <= end)]

    sin_dato = [t for t in bonos if t not in set(hist["Ticker"].unique())]

    piv_tir = hist.pivot_table(index="Date", columns="Ticker", values="TIR", aggfunc="first")
    piv_md = hist.pivot_table(index="Date", columns="Ticker", values="MD", aggfunc="first")
    piv_par = hist.pivot_table(index="Date", columns="Ticker", values="Paridad", aggfunc="first")

    piv_tir = piv_tir.reindex(fechas).ffill()
    piv_md = piv_md.reindex(fechas).ffill()
    piv_par = piv_par.reindex(fechas).ffill()

    for t in sin_dato:
        piv_tir[t] = np.nan
        piv_md[t] = np.nan
        piv_par[t] = np.nan

    cash_tna = port.loc[port["Es_Cash"], "TNA_Manual_pct"]
    cash_tna_val = float(cash_tna.iloc[0]) if not cash_tna.empty and pd.notna(cash_tna.iloc[0]) else 0.0
    cash_daily_ret = (1 + cash_tna_val / 100.0) ** (1 / 365.0) - 1

    ret_bonos = piv_par.pct_change().fillna(0.0)
    w_bonos = pesos.reindex(ret_bonos.columns).fillna(0.0)
    port_ret = (ret_bonos.mul(w_bonos, axis=1)).sum(axis=1)

    w_cash = float(pesos.get("CAUCION", 0.0)) if "CAUCION" in pesos.index else \
        float(port.loc[port["Es_Cash"], "Peso_pct"].sum() / 100.0)
    port_ret = port_ret + w_cash * cash_daily_ret

    idx = 100 * (1 + port_ret).cumprod()
    idx.iloc[0] = 100.0

    w_valid = pesos.reindex(piv_tir.columns).fillna(0.0)
    tir_cartera_t = (piv_tir.mul(w_valid, axis=1)).sum(axis=1, min_count=1)
    md_cartera_t = (piv_md.mul(w_valid, axis=1)).sum(axis=1, min_count=1)
    if "CAUCION" in pesos.index or port["Es_Cash"].any():
        tir_cartera_t = tir_cartera_t + w_cash * cash_tna_val
        md_cartera_t = md_cartera_t + 0.0  # cash aporta MD=0

    out = pd.DataFrame({"Date": fechas, "TIR_Cartera": tir_cartera_t.values,
                         "MD_Cartera": md_cartera_t.values, "Indice_Valorizacion": idx.values})
    return out, sin_dato


def forward_projection(valor_actual: float, tir_cartera_pct: float, horizonte_dias: int,
                        as_of: pd.Timestamp) -> pd.DataFrame:
    """Proyecta el valor de la cartera hacia adelante asumiendo DEVENGAMIENTO a
    tasa constante (la TIR ponderada actual), sin variación de precios. Es una
    referencia de 'piso esperado si no pasa nada', no una predicción de mercado."""
    dias = np.arange(0, horizonte_dias + 1)
    fechas = [pd.Timestamp(as_of) + timedelta(days=int(d)) for d in dias]
    valores = valor_actual * (1 + tir_cartera_pct / 100.0) ** (dias / 365.0)
    return pd.DataFrame({"Date": fechas, "Valor_Proyectado": valores})


def build_excel_snapshot(tabla_hoy: pd.DataFrame, kpis: dict, serie_hist: pd.DataFrame = None) -> bytes:
    """Informe simple en Excel (foto del momento): resumen de KPIs, tabla de
    holdings y serie histórica si está disponible. Sin fórmulas ni gráficos
    nativos — pensado como respaldo de datos, no como modelo vivo."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        resumen = pd.DataFrame({
            "Métrica": ["TIR ponderada de la cartera (%)", "Duration (MD) ponderada (años)",
                        "Peso cubierto con datos (%)", "Peso total cargado (%)"],
            "Valor": [round(kpis.get("tir_cartera", np.nan), 2), round(kpis.get("md_cartera", np.nan), 2),
                      round(kpis.get("peso_cubierto", np.nan), 1), round(kpis.get("peso_total", np.nan), 1)],
        })
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        tabla_hoy.to_excel(writer, sheet_name="Holdings", index=False)
        if serie_hist is not None and not serie_hist.empty:
            serie_hist.to_excel(writer, sheet_name="Serie Historica", index=False)
    return buf.getvalue()


# =====================================================================
# App
# =====================================================================

st.set_page_config(page_title="Cartera Grupo 8 — Dashboard IPS", layout="wide", page_icon="📊")

st.markdown(
    f"""
    <div style="padding:16px 22px;border-radius:12px;
         background:linear-gradient(90deg,{COLORS['primary']},#31456e);color:white;">
      <span style="font-size:1.5rem;font-weight:700;">📊 Cartera Grupo 8 — Panel de Monitoreo IPS</span><br>
      <span style="opacity:.85;">Financiamiento de expansión PyME · Capital de trabajo + cobertura cambiaria ·
      Comité de Inversiones, Administración de Carteras de Inversión (EFI)</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Herramienta de análisis y monitoreo para uso académico. No constituye recomendación de inversión.")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    modo_demo = st.toggle("Usar datos de ejemplo (sin API key)", value=not ALPHACAST_AVAILABLE,
                           help="Genera una serie sintética para ensayar el panel. Desactivalo para "
                                "usar datos reales de Alphacast en la clase.")
    api_key = ""
    if not modo_demo:
        if not ALPHACAST_AVAILABLE:
            st.error("El paquete `alphacast` no está instalado en este entorno. Instalalo con "
                      "`pip install alphacast` o activá el modo de ejemplo.")
        try:
            default_key = st.secrets["ALPHACAST_API_KEY"]
        except Exception:
            default_key = ""
        api_key = st.text_input("Alphacast API Key", value=default_key, type="password")
        dataset_id = st.number_input("Dataset (ONs/Bonos/Soberanos)", value=int(DATASET_ID), step=1)
        if st.button("🔄 Refrescar datos (limpiar caché)"):
            download_dataset.clear()
            st.rerun()
    else:
        dataset_id = DATASET_ID
        st.info("Modo de ejemplo activo: los valores son sintéticos, solo para recorrer la interfaz.")

    st.divider()
    st.subheader("💰 Cartera")
    valor_total = st.number_input("Valor nominal total de la cartera (ARS)",
                                   value=175_000_000.0, step=1_000_000.0, format="%.0f")

    st.divider()
    st.subheader("📐 Límites del IPS (editables)")
    md_target_lo, md_target_hi = st.slider("Banda objetivo de Duration (años)", 0.0, 2.0, (0.50, 0.80), 0.01)
    limite_emisor_corp = st.number_input("Límite máx. por ON corporativa individual (%)", value=3.15, step=0.05)
    sublimite_dl_tramo_largo = st.number_input("Sublímite Dollar-Linked sobre el total (%)", value=86.0, step=1.0)

    st.divider()
    st.subheader("🗓️ Cronograma del préstamo")
    fecha_desembolso = st.date_input("Fecha de desembolso del préstamo", value=datetime(2026, 4, 1))

# ---------------------------------------------------------------------
# Cartera — composición fija (según el IPS), con un único input real
# ---------------------------------------------------------------------
if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = default_portfolio()

st.subheader("📋 Composición de la cartera (según el IPS del Grupo 8)")
st.caption("Los 8 instrumentos y sus pesos surgen de reconciliar los dos gráficos de torta del IPS "
           "(cartera consolidada + detalle de Objetivo 1) — no hace falta agregar ni completar nada. "
           "El único dato que no sale de Alphacast es la tasa de la caución/money market: se carga abajo.")

base_port = st.session_state["portfolio"].copy()
tna_caucion = st.number_input(
    "Tasa de Caución / FCI Money Market — TNA (%)", min_value=0.0, max_value=200.0,
    value=float(base_port.loc[base_port["Ticker"] == "CAUCION", "TNA_Manual_pct"].iloc[0]), step=0.5,
    help="Único instrumento que no cotiza en Alphacast: la caución/FCI money market. El resto de la "
         "tabla se busca automáticamente por Ticker en el dataset 41886.",
)
base_port.loc[base_port["Ticker"] == "CAUCION", "TNA_Manual_pct"] = tna_caucion
st.session_state["portfolio"] = base_port

tabla_composicion = base_port[["Ticker", "Descripcion", "Objetivo", "Tramo", "Peso_pct"]].rename(
    columns={"Peso_pct": "Peso (%)"})
st.dataframe(tabla_composicion, use_container_width=True, hide_index=True, height=320)
st.caption(f"Suma de pesos: **{base_port['Peso_pct'].sum():.1f}%** · Objetivo 2 (cobertura FX) = "
           f"**{base_port.loc[base_port['Objetivo'].str.startswith('2'), 'Peso_pct'].sum():.1f}%** · "
           f"Objetivo 1 (capital de trabajo) = "
           f"**{base_port.loc[base_port['Objetivo'].str.startswith('1'), 'Peso_pct'].sum():.1f}%**")

with st.expander("⚙️ Ajustes avanzados (opcional — solo si el IPS real cambió tickers o pesos)"):
    st.caption("Esto NO hace falta para la clase. Se deja por si hay que corregir algo puntual sin "
                "tocar el código: no se pueden agregar ni borrar instrumentos, solo editar valores.")
    edited = st.data_editor(
        base_port, num_rows="fixed", use_container_width=True, hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "Descripcion": st.column_config.TextColumn("Descripción", width="large"),
            "Objetivo": st.column_config.SelectboxColumn("Objetivo",
                         options=["1 · Capital de Trabajo", "2 · Cobertura FX Préstamo"]),
            "Tramo": st.column_config.TextColumn("Tramo"),
            "Peso_pct": st.column_config.NumberColumn("Peso (%)", min_value=0.0, max_value=100.0, step=0.05),
            "Es_Cash": st.column_config.CheckboxColumn("¿Cash/Money Market?"),
            "TNA_Manual_pct": st.column_config.NumberColumn("TNA manual (% si es Cash)", step=0.5),
            "Segmento_Manual": st.column_config.SelectboxColumn("Segmento (fallback)",
                                options=["Sovereign", "Corporate", "Cash", "Otro"]),
            "Clase_Manual": st.column_config.SelectboxColumn("Clase (fallback)",
                             options=["CER", "Fija", "DL", "Dual", "HD", "Cash", "Otro"]),
        },
        key="portfolio_editor",
    )
    st.session_state["portfolio"] = edited.copy()

port_df = st.session_state["portfolio"].copy()
port_df["Peso_pct"] = pd.to_numeric(port_df["Peso_pct"], errors="coerce").fillna(0.0)

suma_pesos = port_df["Peso_pct"].sum()
if abs(suma_pesos - 100.0) > 0.5:
    st.warning(f"Los pesos suman {suma_pesos:.1f}%, no 100% — se modificó algo en 'Ajustes avanzados'. "
               f"Las métricas de abajo se calculan sobre el peso que sí está cargado.")

# ---------------------------------------------------------------------
# Descarga de datos (real o sintética)
# ---------------------------------------------------------------------
tickers_bono = port_df.loc[~port_df["Es_Cash"], "Ticker"].tolist()

df_norm = pd.DataFrame()
fecha_datos = None
data_ok = False

if modo_demo:
    df_norm = synthetic_dataset(tickers_bono)
    if not df_norm.empty:
        fecha_datos = df_norm["Date"].max()
        data_ok = True
else:
    if not api_key.strip():
        st.warning("Ingresá tu Alphacast API Key en la barra lateral (o activá el modo de ejemplo) "
                    "para descargar los datos de mercado.")
    elif not ALPHACAST_AVAILABLE:
        st.error("No se puede consultar Alphacast: falta el paquete `alphacast` en este entorno.")
    else:
        try:
            with st.spinner("Descargando dataset de Alphacast..."):
                raw = download_dataset(api_key.strip(), int(dataset_id))
            df_norm = normalize_dataset(raw)
            fecha_datos = df_norm["Date"].max() if "Date" in df_norm.columns and not df_norm.empty else None
            data_ok = not df_norm.empty
        except Exception as e:
            st.error(f"No se pudo descargar/procesar el dataset: {e}")

if not data_ok:
    st.stop()

as_of_hoy = pd.Timestamp(fecha_datos)
snap_hoy = build_holdings_snapshot(port_df, df_norm, as_of_hoy)
kpis_hoy = portfolio_kpis(snap_hoy)

tab_hoy, tab_hist, tab_proy, tab_riesgo, tab_metodo = st.tabs(
    ["🏠 Cartera Hoy", "📈 Evolución Histórica", "🗓️ Semana & Proyección",
     "⚖️ Riesgo & Límites IPS", "ℹ️ Metodología"]
)

# ---------------------------------------------------------------------
# TAB 1 — Cartera Hoy
# ---------------------------------------------------------------------
with tab_hoy:
    no_encontrados = snap_hoy.loc[~snap_hoy["Encontrado"], "Ticker"].tolist()
    if no_encontrados:
        st.info(f"Sin dato de mercado para: **{', '.join(no_encontrados)}**. No se incluyen en la TIR/Duration "
                f"ponderada hasta que tengan cotización en el dataset (o edites su fila como Cash con TNA manual).")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("TIR ponderada de la cartera", f"{kpis_hoy['tir_cartera']:.2f}%" if pd.notna(kpis_hoy['tir_cartera']) else "—")
    k2.metric("Duration (MD) ponderada", f"{kpis_hoy['md_cartera']:.2f} años" if pd.notna(kpis_hoy['md_cartera']) else "—")
    k3.metric("Valor nominal total", f"${valor_total:,.0f}".replace(",", "."))
    k4.metric("Peso cubierto con datos", f"{kpis_hoy['peso_cubierto']:.1f}%")
    st.caption(f"Snapshot al {as_of_hoy.strftime('%d/%m/%Y')}" + (" (datos de ejemplo)" if modo_demo else ""))

    st.divider()
    p1, p2, p3 = st.columns(3)

    with p1:
        g = snap_hoy.groupby("Objetivo")["Peso_pct"].sum().reset_index()
        fig = px.pie(g, names="Objetivo", values="Peso_pct", hole=0.45,
                     color="Objetivo", color_discrete_map={
                         "1 · Capital de Trabajo": COLORS["obj1"], "2 · Cobertura FX Préstamo": COLORS["obj2"]})
        fig.update_traces(textinfo="label+percent", textposition="outside")
        fig.update_layout(title="Por Objetivo", showlegend=False, height=380, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        g = snap_hoy.groupby("Tramo")["Peso_pct"].sum().reset_index()
        fig = px.pie(g, names="Tramo", values="Peso_pct", hole=0.45)
        fig.update_traces(textinfo="label+percent", textposition="outside",
                           marker=dict(colors=px.colors.qualitative.Safe))
        fig.update_layout(title="Por Tramo", showlegend=False, height=380, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with p3:
        g = snap_hoy.groupby("Ticker")["Peso_pct"].sum().reset_index()
        fig = px.pie(g, names="Ticker", values="Peso_pct", hole=0.45)
        fig.update_traces(textinfo="label+percent", textposition="outside",
                           marker=dict(colors=px.colors.qualitative.Set2))
        fig.update_layout(title="Por Instrumento", showlegend=False, height=380, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detalle de holdings")
    cols_show = ["Ticker", "Descripcion", "Objetivo", "Tramo", "Peso_pct", "Clase", "Segmento",
                 "TIR", "MD", "Paridad", "Fecha_Dato", "Encontrado"]
    tabla = snap_hoy[[c for c in cols_show if c in snap_hoy.columns]].copy()
    tabla["Fecha_Dato"] = pd.to_datetime(tabla["Fecha_Dato"], errors="coerce").dt.strftime("%d/%m/%Y")
    st.dataframe(tabla.sort_values("Peso_pct", ascending=False), use_container_width=True, height=280)

    csv = tabla.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar tabla (CSV)", data=csv,
                        file_name=f"cartera_grupo8_{as_of_hoy.strftime('%Y%m%d')}.csv", mime="text/csv")

# ---------------------------------------------------------------------
# TAB 2 — Evolución Histórica
# ---------------------------------------------------------------------
with tab_hist:
    st.markdown("### 📈 ¿Cómo venimos? — Evolución de la cartera")
    ventana = st.select_slider("Ventana de análisis", options=["30 días", "90 días", "180 días", "365 días"],
                                value="180 días")
    dias_map = {"30 días": 30, "90 días": 90, "180 días": 180, "365 días": 365}
    start_hist = as_of_hoy - pd.Timedelta(days=dias_map[ventana])

    serie, sin_dato = historical_series(port_df, df_norm, start_hist, as_of_hoy)
    if sin_dato:
        st.caption(f"⚠️ Sin serie histórica en el dataset para: **{', '.join(sin_dato)}** — se asume "
                    f"retorno plano (0%) para esos tickers en el índice de valorización; no afecta la "
                    f"lectura de TIR/Duration porque esas filas quedan en NaN y se excluyen del promedio.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=serie["Date"], y=serie["TIR_Cartera"], mode="lines",
                                  line=dict(width=2.4, color=COLORS["primary"]), name="TIR cartera"))
        fig.update_layout(title="TIR ponderada de la cartera en el tiempo", height=380, **PLOTLY_LAYOUT)
        style_axes(fig, "Fecha", "TIR (%)")
        watermark(fig, as_of_hoy, "Alphacast" if not modo_demo else "Ejemplo")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=serie["Date"], y=serie["MD_Cartera"], mode="lines",
                                  line=dict(width=2.4, color=COLORS["accent"]), name="MD cartera"))
        fig.add_hrect(y0=md_target_lo, y1=md_target_hi, fillcolor=COLORS["ok"], opacity=0.08, line_width=0)
        fig.update_layout(title="Duration (MD) ponderada en el tiempo · banda objetivo sombreada",
                           height=380, **PLOTLY_LAYOUT)
        style_axes(fig, "Fecha", "Modified Duration (años)")
        watermark(fig, as_of_hoy, "Alphacast" if not modo_demo else "Ejemplo")
        st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie["Date"], y=serie["Indice_Valorizacion"], mode="lines",
                              line=dict(width=2.6, color=COLORS["obj1"]), fill="tozeroy",
                              fillcolor="rgba(37,99,235,0.08)", name="Índice (base 100)"))
    fig.add_hline(y=100, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(title=f"Índice de valorización de la cartera (base 100 al inicio de la ventana · {ventana})",
                       height=380, **PLOTLY_LAYOUT)
    style_axes(fig, "Fecha", "Índice")
    watermark(fig, as_of_hoy, "Alphacast" if not modo_demo else "Ejemplo")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("El índice combina el retorno por precio (Paridad) de cada bono y el devengamiento de la "
               "TNA manual del tramo cash, ponderados por el peso de cada holding. Es una aproximación "
               "de retorno total sin reinversión de cupón ni costos de transacción.")

# ---------------------------------------------------------------------
# TAB 3 — Semana & Proyección
# ---------------------------------------------------------------------
with tab_proy:
    st.markdown("### 🗓️ Semana actual vs. semana pasada")
    as_of_semana_pasada = as_of_hoy - pd.Timedelta(days=7)
    snap_pasado = build_holdings_snapshot(port_df, df_norm, as_of_semana_pasada)
    kpis_pasado = portfolio_kpis(snap_pasado)

    d_tir = kpis_hoy["tir_cartera"] - kpis_pasado["tir_cartera"] if pd.notna(kpis_hoy["tir_cartera"]) and \
        pd.notna(kpis_pasado["tir_cartera"]) else np.nan
    d_md = kpis_hoy["md_cartera"] - kpis_pasado["md_cartera"] if pd.notna(kpis_hoy["md_cartera"]) and \
        pd.notna(kpis_pasado["md_cartera"]) else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("TIR cartera — hoy", f"{kpis_hoy['tir_cartera']:.2f}%", f"{d_tir:+.2f} pp vs. semana pasada"
              if pd.notna(d_tir) else None)
    c2.metric("MD cartera — hoy", f"{kpis_hoy['md_cartera']:.2f} años", f"{d_md:+.2f} vs. semana pasada"
              if pd.notna(d_md) else None)
    c3.metric("Fecha de comparación", as_of_semana_pasada.strftime("%d/%m/%Y"))
    st.caption(f"'Hace una semana' toma la última cotización disponible en o antes del "
               f"{as_of_semana_pasada.strftime('%d/%m/%Y')} para cada instrumento (por si algún bono no operó ese día).")

    st.divider()
    st.markdown("### 🔮 Proyección hacia adelante (devengamiento a tasa constante)")
    st.caption("Proyecta el valor de la cartera asumiendo que se mantiene la TIR ponderada actual sin "
               "cambios de precio — es el **piso esperado por devengamiento**, no una predicción de "
               "mercado. Sirve para chequear si el crecimiento esperado alcanza para cubrir los próximos hitos de pago.")

    horizonte = st.slider("Horizonte de proyección (días)", 30, 365, 180, 15)
    proy = forward_projection(valor_total, kpis_hoy["tir_cartera"], horizonte, as_of_hoy)

    fecha_desembolso_ts = pd.Timestamp(fecha_desembolso)
    hitos = {"Mes 3 (20% obra civil)": fecha_desembolso_ts + pd.Timedelta(days=90),
             "Mes 6 (30% obra civil)": fecha_desembolso_ts + pd.Timedelta(days=180),
             "Mes 9 (50% + maquinaria)": fecha_desembolso_ts + pd.Timedelta(days=270)}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=proy["Date"], y=proy["Valor_Proyectado"], mode="lines",
                              line=dict(width=2.6, color=COLORS["obj1"]), name="Valor proyectado"))
    for nombre, fecha_hito in hitos.items():
        if proy["Date"].min() <= fecha_hito <= proy["Date"].max():
            fig.add_vline(x=fecha_hito, line_dash="dash", line_color=COLORS["obj2"],
                          annotation_text=nombre, annotation_font_size=10, annotation_font_color=COLORS["obj2"])
    fig.update_layout(title=f"Cartera proyectada a {horizonte} días (TIR constante {kpis_hoy['tir_cartera']:.2f}%)",
                       height=420, **PLOTLY_LAYOUT)
    style_axes(fig, "Fecha", "Valor (ARS)")
    watermark(fig, as_of_hoy, "Proyección propia")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 💳 Capacidad de pago — DSCR trimestral (dato del IPS)")
    st.caption("Reproduce la tabla del IPS (Servicio de Deuda — Sistema Francés) como referencia de contexto; "
               "no se recalcula acá, es informativa.")
    dscr = pd.DataFrame({
        "Trimestre": ["T2-26", "T3-26", "T4-26", "T1-27", "T2-27", "T3-27", "T4-27",
                       "T1-28", "T2-28", "T3-28", "T4-28", "T1-29"],
        "DSCR": [1.75, 2.12, 2.30, 2.50, 4.65, 3.84, 4.24, 4.70, 5.18, 5.55, 5.91, 6.32],
    })
    fig = px.bar(dscr, x="Trimestre", y="DSCR", color_discrete_sequence=[COLORS["primary"]])
    fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS["warn"], annotation_text="DSCR = 1x (cobertura mínima)")
    fig.update_layout(title="DSCR trimestral por cuota del préstamo", height=380, **PLOTLY_LAYOUT, showlegend=False)
    style_axes(fig, "Trimestre", "DSCR (x)")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 4 — Riesgo & Límites IPS
# ---------------------------------------------------------------------
with tab_riesgo:
    st.markdown("### ⚖️ Chequeo de límites del IPS")

    md_ok = pd.notna(kpis_hoy["md_cartera"]) and md_target_lo <= kpis_hoy["md_cartera"] <= md_target_hi
    st.markdown(f"""
    **1) Duration dentro de la banda objetivo ({md_target_lo:.2f} – {md_target_hi:.2f} años):**
    {"✅ Cumple" if md_ok else "🔴 Fuera de banda"} — actual **{kpis_hoy['md_cartera']:.2f} años**
    """ if pd.notna(kpis_hoy["md_cartera"]) else "**1) Duration:** sin datos suficientes.")

    st.divider()
    st.markdown(f"**2) Concentración máxima por ON corporativa individual (límite {limite_emisor_corp:.2f}%):**")
    corp = snap_hoy[snap_hoy["Segmento"] == "Corporate"].copy()
    if not corp.empty:
        corp["Cumple"] = corp["Peso_pct"] <= limite_emisor_corp
        corp_show = corp[["Ticker", "Descripcion", "Peso_pct", "Cumple"]].sort_values("Peso_pct", ascending=False)
        st.dataframe(corp_show, use_container_width=True, height=180)
        incumplen = corp_show.loc[~corp_show["Cumple"], "Ticker"].tolist()
        if incumplen:
            st.error(f"🔴 Exceden el límite individual: **{', '.join(incumplen)}**. Rebalancear o "
                      f"diversificar en más emisores.")
        else:
            st.success("✅ Ninguna ON corporativa individual excede el límite.")
    else:
        st.info("No hay holdings marcados como Segmento = Corporate en la cartera actual.")

    st.divider()
    st.markdown(f"**3) Sublímite Dollar-Linked sobre el total de la cartera (referencia IPS: {sublimite_dl_tramo_largo:.0f}%):**")
    peso_total_cartera = snap_hoy["Peso_pct"].sum()
    peso_dl = snap_hoy.loc[snap_hoy["Clase"] == "DL", "Peso_pct"].sum()
    if peso_total_cartera > 0:
        pct_dl = peso_dl / peso_total_cartera * 100
        st.metric("% Dollar-Linked sobre el total de la cartera", f"{pct_dl:.1f}%",
                  f"{pct_dl - sublimite_dl_tramo_largo:+.1f} pp vs. referencia")
        st.caption("Todo instrumento clasificado DL (D31M7 y D30S6 por defecto) es la cobertura cambiaria "
                    "del préstamo — por diseño del IPS, este ratio debería dar ≈86% de la cartera total.")
    else:
        st.info("No hay holdings cargados para calcular este ratio.")

    st.divider()
    st.markdown("""
    **4) Otros lineamientos del IPS (chequeo cualitativo):**
    - ✅ Cero exposición a renta variable, cripto o derivados especulativos (por construcción de esta cartera de renta fija).
    - ✅ Sin compra de dólar oficial/MEP — la cobertura FX se instrumenta vía instrumentos Dollar-Linked.
    - 🔁 Rebalanceo mensual recomendado por desvío de tramo y liquidez mínima del Tramo Operativo.
    """)

# ---------------------------------------------------------------------
# TAB 5 — Metodología
# ---------------------------------------------------------------------
with tab_metodo:
    st.markdown("""
    ### ℹ️ Metodología, fuentes y supuestos

    **Fuente de mercado:** Alphacast, dataset **41886** (ONs / Bonos / Soberanos) — el mismo dataset
    usado en el panel PRO de Renta Fija. Trae TIR (`irr`), Modified Duration, Paridad, segmento de
    mercado (Sovereign/Corporate) y estructura de cupón por ticker y fecha.

    **Instrumentos de money-market (caución, FCI, cuenta remunerada):** no cotizan en Alphacast. Se
    modelan con una **TNA manual editable** en la fila `Es_Cash = True`; su Modified Duration se fija en 0.

    **TIR y Duration de la cartera:** promedio ponderado por peso (`Peso_pct`) de la TIR/MD de cada
    holding, usando la última cotización disponible en o antes de la fecha de análisis (no fuerza que
    todos los instrumentos hayan operado exactamente ese día).

    **Índice de valorización histórico:** retorno diario ponderado, usando la variación de Paridad de
    cada bono más el devengamiento de la TNA del tramo cash. Es una aproximación de retorno total —
    no incluye reinversión de cupón ni costos de transacción.

    **Proyección hacia adelante:** devengamiento a **TIR constante** (sin variación de precios). Es un
    piso de referencia, no una predicción de mercado; sirve para contrastar contra los hitos de pago
    conocidos del préstamo (Mes 3 / 6 / 9).

    **Composición de la cartera:** son los 8 instrumentos exactos del IPS del Grupo 8, reconciliando
    los dos gráficos de torta de la presentación (cartera consolidada $175M + detalle de Objetivo 1):
    **D31M7** (79,1%) y **D30S6** (6,9%) — Dollar-Linked, cobertura FX del préstamo, 86,0% del total —
    y dentro de Objetivo 1 (14,0% del total): **Cauciones** (1,4%), **S31L6** (1,4%), **TZXD6** (7,0%),
    **TLCQO** (2,1%), **LOC5O** (1,05%) y **AO27** (1,05%). No hace falta agregar ni completar ningún
    instrumento — el único dato manual es la TNA de la caución, porque no cotiza en Alphacast.

    **Cómo correr esto en Streamlit Cloud:**
    1. Subí `app.py` y `requirements.txt` a un repositorio de GitHub.
    2. En [share.streamlit.io](https://share.streamlit.io), creá una app apuntando a ese repo/`app.py`.
    3. Cargá tu Alphacast API Key como *secret* (`ALPHACAST_API_KEY`) en la configuración de la app,
       o pegala directamente en la barra lateral al abrir el panel.

    ⚠️ Esta herramienta es de análisis y monitoreo académico. No constituye recomendación de inversión.
    """)

# ---------------------------------------------------------------------
# Exportar informe completo (Excel) — combina lo calculado en las pestañas
# ---------------------------------------------------------------------
st.divider()
st.subheader("📥 Exportar informe completo (Excel)")
st.caption("Descarga un .xlsx con el resumen de KPIs, la tabla de holdings de hoy y la serie histórica "
           "mostrada en la pestaña 'Evolución Histórica'. Es una foto del momento, no un modelo con fórmulas.")
try:
    excel_bytes = build_excel_snapshot(tabla, kpis_hoy, serie if "serie" in dir() else None)
    st.download_button(
        "⬇️ Descargar informe Excel (.xlsx)",
        data=excel_bytes,
        file_name=f"cartera_grupo8_informe_{as_of_hoy.strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
except Exception as e:
    st.warning(f"No se pudo generar el Excel: {e}")
