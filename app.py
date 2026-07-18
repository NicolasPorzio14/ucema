# -*- coding: utf-8 -*-
"""
Dashboard de Cartera — Grupo 8 · Administración de Carteras de Inversión (EFI)
================================================================================
Panel de monitoreo para la cartera IPS del cliente PyME (financiamiento de
expansión con préstamo de $150M ARS, cobertura cambiaria sin acceso a dólar
oficial, capital de trabajo administrado en tres tramos).

Qué hace este panel:
  1) CARTERA HOY      → composición (Objetivo/Tramo/Instrumento), TIR y
                         Duration ponderadas, y el resultado de la cartera
                         desde la compra (nominales fijos vs. precio de hoy).
  2) SIMULACIÓN SEMANAL → simula la compra de los 8 instrumentos el 29/6/2026
                         con el capital asignado, y muestra el valor de la
                         cartera CADA LUNES (a nominales fijos) hasta fin de
                         año: tramo realizado con precio de mercado real, tramo
                         proyectado por devengamiento a la TIR vigente hoy.
  3) BENCHMARKS & RATIOS → compara la cartera contra A3500 (dólar oficial),
                         CER (inflación) y el ETF SHY (bonos del Tesoro de
                         EE.UU. 1-3 años), y calcula Sharpe, Sortino e
                         Information Ratio para la cartera total y para cada
                         Objetivo (1 · Capital de Trabajo / 2 · Cobertura FX).
  4) BRECHA CAMBIARIA  → dólar Oficial/Minorista/MEP/CCL y bandas de flotación,
                         con la brecha (MEP vs. Oficial) en el tiempo. Oficial/
                         Minorista/Bandas vienen de la API pública del BCRA (sin
                         API key); MEP/CCL de Alphacast.
  5) METODOLOGÍA       → fuentes y supuestos.

Fuente de mercado: Alphacast, dataset 41886 (ONs / Bonos / Soberanos — el mismo
dataset del panel PRO de Renta Fija). Los instrumentos de money-market (caución,
FCI, cuenta remunerada) no cotizan ahí: se modelan con una TNA manual editable.

La cartera de 8 holdings es la composición EXACTA reconciliada de los dos
gráficos de torta del IPS (cartera consolidada + detalle de Objetivo 1): no es
una estimación ni requiere completar instrumentos. El único dato que no sale
de Alphacast es la TNA de la caución/money market.

Requisitos: streamlit, pandas, numpy, plotly, alphacast, openpyxl
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
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

# --- Brecha cambiaria: fuentes de datos ---
# MEP y CCL no son series oficiales del BCRA (surgen de operar bonos/acciones en
# distintas plazas), así que salen de Alphacast. Oficial, minorista y las bandas
# de flotación SÍ son series oficiales del BCRA, y su API pública (sin API key)
# las expone directamente — se usa esa en lugar de Alphacast para esas cuatro,
# porque es la fuente primaria y no consume cupo de la cuenta de Alphacast.
FX_DATASET_ID = 5288  # Alphacast: "Markets - Argentina - FX premiums - Daily" (Blue/MEP/CCL/Oficial/Mayorista)
BCRA_API_BASE = "https://api.bcra.gob.ar/estadisticas/v4.0/monetarias"
BCRA_VARS = {"usd_oficial": 5, "usd_minorista": 4, "lower_band": 1187, "upper_band": 1188}
REGIMEN_BANDAS_INICIO = "2025-04-14"  # primera fecha con dato de banda publicada por el BCRA

# --- Benchmarks (A3500 / CER / SHY) ---
# A3500 y CER son series oficiales del BCRA (API pública, sin key): A3500 es la
# misma variable "usd_oficial" ya definida arriba (idVariable 5); CER es el
# "Coeficiente de Estabilización de Referencia" (idVariable 30). SHY (ETF de
# bonos del Tesoro de EE.UU. de 1-3 años) no lo publica ningún organismo
# argentino: sale de la API pública de cotizaciones de Yahoo Finance.
BCRA_CER_ID = 30
YAHOO_CHART_API = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
SHY_TICKER = "SHY"

COLORS = {
    "primary": "#1f2a44",
    "accent": "#c8963e",
    "obj1": "#2563eb",
    "obj2": "#ea580c",
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


def fmt_ars(x: float) -> str:
    """Formatea un monto en pesos con separador de miles al estilo argentino
    (punto), sin tocar el resto del texto — evita pisar comas gramaticales
    cuando el número se inserta dentro de una oración."""
    if pd.isna(x):
        return "—"
    return f"${x:,.0f}".replace(",", ".")


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
    El resto (14,0%) es Objetivo 1, y sus pesos internos reconstruyen el 3,5%
    que faltaba en el primer gráfico (S31L6 1,4% + LOC5O 1,05% + AO27 1,05%).
    Las 8 filas de abajo son la cartera COMPLETA.
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
        dict(Ticker="TLCQO", Descripcion="ON corporativa (Hard-Dollar)",
             Objetivo="1 · Capital de Trabajo", Tramo="3 · Estructural (>1 año)",
             Peso_pct=2.1, Es_Cash=False, TNA_Manual_pct=np.nan,
             Segmento_Manual="Corporate", Clase_Manual="HD"),
        dict(Ticker="LOC5O", Descripcion="ON Loma Negra (Hard-Dollar)",
             Objetivo="1 · Capital de Trabajo", Tramo="3 · Estructural (>1 año)",
             Peso_pct=1.05, Es_Cash=False, TNA_Manual_pct=np.nan,
             Segmento_Manual="Corporate", Clase_Manual="HD"),
        dict(Ticker="AO27", Descripcion="Bono Soberano",
             Objetivo="1 · Capital de Trabajo", Tramo="3 · Estructural (>1 año)",
             Peso_pct=1.05, Es_Cash=False, TNA_Manual_pct=np.nan,
             Segmento_Manual="Sovereign", Clase_Manual="Otro"),
    ]
    return pd.DataFrame(rows)


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
    SOLO para poder recorrer la interfaz sin conexión a Alphacast."""
    rng = np.random.default_rng(seed)
    hoy = pd.Timestamp(datetime.now().date())
    fechas = pd.bdate_range(end=hoy, periods=days)
    base_tir = {"D31M7": 6.0, "D30S6": 4.0, "S31L6": 32.0, "TZXD6": 24.0,
                "TLCQO": 4.5, "LOC5O": 5.0, "AO27": 12.0}
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
# Brecha cambiaria — dólar oficial/minorista/bandas (BCRA, API pública sin
# key) + MEP/CCL (Alphacast, dataset de FX premiums)
# =====================================================================

@st.cache_data(show_spinner=False, ttl=30 * 60)
def fetch_bcra_variable(id_variable: int, desde: str, hasta: str) -> pd.DataFrame:
    """Serie diaria de una variable del BCRA (API pública v4.0, sin autenticación).
    Devuelve columnas Date/Valor; DataFrame vacío si la API falla o no hay datos
    (nunca inventa un valor — un error acá se ve como un hueco en el gráfico)."""
    try:
        r = requests.get(f"{BCRA_API_BASE}/{id_variable}", params={"desde": desde, "hasta": hasta}, timeout=20)
        r.raise_for_status()
        data = r.json()
        det = data.get("results", [{}])[0].get("detalle", [])
        if not det:
            return pd.DataFrame(columns=["Date", "Valor"])
        df = pd.DataFrame(det)
        df["Date"] = pd.to_datetime(df["fecha"])
        df["Valor"] = pd.to_numeric(df["valor"], errors="coerce")
        return df[["Date", "Valor"]].sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["Date", "Valor"])


@st.cache_data(show_spinner=False, ttl=30 * 60)
def fetch_bcra_fx_bundle(desde: str, hasta: str) -> tuple:
    """Combina usd_oficial, usd_minorista, lower_band y upper_band del BCRA en
    un solo DataFrame por fecha. Devuelve (df, variables_sin_dato)."""
    series = {}
    faltantes = []
    for nombre, id_var in BCRA_VARS.items():
        d = fetch_bcra_variable(id_var, desde, hasta)
        if d.empty:
            faltantes.append(nombre)
        else:
            series[nombre] = d.set_index("Date")["Valor"]
    if not series:
        return pd.DataFrame(), list(BCRA_VARS.keys())
    out = pd.concat(series, axis=1).reset_index().rename(columns={"index": "Date"})
    return out.sort_values("Date").reset_index(drop=True), faltantes


def normalize_fx_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el dataset de Alphacast de FX premiums para quedarse con MEP y
    CCL. Los nombres de columna varían de dataset a dataset, así que se buscan
    por palabra clave en vez de hardcodear un nombre exacto — si no encuentra
    alguna, la deja en NaN y lo reporta (nunca inventa el dato)."""
    df = df.copy()
    date_col = next((c for c in df.columns if str(c).strip().lower() in ("date", "fecha")), df.columns[0])
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    def find_col(keywords, exclude=()):
        for c in df.columns:
            cl = str(c).strip().lower()
            if any(k in cl for k in keywords) and not any(k in cl for k in exclude):
                return c
        return None

    col_mep = find_col(["mep", "bolsa"])
    col_ccl = find_col(["ccl", "contadoconliqui", "contado con liqui", "contado_con_liqui"])

    out = pd.DataFrame({"Date": df["Date"]})
    out["usd_mep"] = pd.to_numeric(df[col_mep], errors="coerce") if col_mep else np.nan
    out["usd_ccl"] = pd.to_numeric(df[col_ccl], errors="coerce") if col_ccl else np.nan
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    out.attrs["col_mep"] = col_mep
    out.attrs["col_ccl"] = col_ccl
    return out


@st.cache_data(show_spinner=False)
def synthetic_fx_bundle(desde: str, hasta: str, seed: int = 8) -> pd.DataFrame:
    """Serie sintética de oficial/minorista/MEP/CCL/bandas SOLO para el modo de
    ejemplo — reproduce el orden de magnitud y la lógica de bandas (piso -1%/mes,
    techo +1%/mes desde $1000-$1400 el 14/4/2025) pero no es dato real."""
    rng = np.random.default_rng(seed)
    fechas = pd.bdate_range(start=desde, end=hasta)
    base = pd.Timestamp(REGIMEN_BANDAS_INICIO)
    dias = np.array([(f - base).days for f in fechas], dtype=float)
    meses = np.clip(dias / 30.44, 0, None)

    oficial0 = 1200.0
    oficial = oficial0 + np.cumsum(rng.normal(1.0, 4.0, len(fechas)))
    minorista = oficial * 1.015
    mep = oficial * (1 + np.clip(0.02 + np.cumsum(rng.normal(0, 0.0009, len(fechas))), 0.0, 0.20))
    ccl = mep * (1 + np.abs(rng.normal(0.004, 0.004, len(fechas))))
    upper_band = 1400.0 * (1.01 ** meses)
    lower_band = 1000.0 * (0.99 ** meses)

    return pd.DataFrame({"Date": fechas, "usd_oficial": oficial, "usd_minorista": minorista,
                          "usd_mep": mep, "usd_ccl": ccl, "upper_band": upper_band, "lower_band": lower_band})


def compute_brecha(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la columna Brecha = (MEP - Oficial) / Oficial, idéntica a la
    fórmula de la planilla original del usuario."""
    d = df.copy()
    d["Brecha"] = (d["usd_mep"] - d["usd_oficial"]) / d["usd_oficial"]
    return d


# =====================================================================
# Benchmarks (A3500 / CER / SHY) y ratios de riesgo-retorno
# =====================================================================

@st.cache_data(show_spinner=False, ttl=30 * 60)
def fetch_shy_series(desde: str, hasta: str) -> pd.DataFrame:
    """Precio diario ajustado del ETF SHY (bonos del Tesoro de EE.UU. 1-3 años),
    API pública de Yahoo Finance (sin API key). Devuelve Date/Valor en USD."""
    try:
        start = pd.Timestamp(desde) - pd.Timedelta(days=6)
        end = pd.Timestamp(hasta) + pd.Timedelta(days=2)
        url = YAHOO_CHART_API.format(ticker=SHY_TICKER)
        r = requests.get(url, params={"period1": int(start.timestamp()), "period2": int(end.timestamp()),
                                       "interval": "1d"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        ts = res["timestamp"]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
        closes = adj if adj else res["indicators"]["quote"][0]["close"]
        df = pd.DataFrame({"Date": pd.to_datetime(ts, unit="s").normalize(), "Valor": closes})
        return df.dropna().sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["Date", "Valor"])


@st.cache_data(show_spinner=False, ttl=30 * 60)
def fetch_benchmark_bcra(desde: str, hasta: str) -> pd.DataFrame:
    """A3500 (idVariable 5, mismo dato que usd_oficial) y CER (idVariable 30),
    ambos de la API pública del BCRA."""
    a3500 = fetch_bcra_variable(BCRA_VARS["usd_oficial"], desde, hasta).rename(columns={"Valor": "A3500"})
    cer = fetch_bcra_variable(BCRA_CER_ID, desde, hasta).rename(columns={"Valor": "CER"})
    return pd.merge(a3500, cer, on="Date", how="outer").sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def synthetic_benchmark_series(desde: str, hasta: str, seed: int = 9) -> pd.DataFrame:
    """A3500/CER/SHY sintéticos SOLO para el modo de ejemplo."""
    rng = np.random.default_rng(seed)
    fechas = pd.bdate_range(start=desde, end=hasta)
    a3500 = 1450 + np.cumsum(rng.normal(0.6, 3.0, len(fechas)))
    cer = 50.0 * np.cumprod(1 + rng.normal(0.0009, 0.0004, len(fechas)))
    shy = 82.0 * np.cumprod(1 + rng.normal(0.00005, 0.0010, len(fechas)))
    return pd.DataFrame({"Date": fechas, "A3500": a3500, "CER": cer, "SHY": shy})


def _align_to_dates(df_diario: pd.DataFrame, fechas: pd.DatetimeIndex, cols: list) -> pd.DataFrame:
    """Lleva una serie diaria a una grilla de fechas específica (ej. los lunes
    de la simulación) tomando, para cada fecha, el último dato conocido en o
    antes de esa fecha (forward-fill) — no inventa datos entre publicaciones."""
    d = df_diario.set_index("Date").sort_index()
    idx_union = pd.DatetimeIndex(sorted(set(list(d.index)) | set(fechas)))
    d = d.reindex(idx_union).ffill()
    return d.reindex(fechas)[cols].reset_index().rename(columns={"index": "Date"})


def build_benchmark_table(port_df: pd.DataFrame, fechas: pd.DatetimeIndex,
                           df_bcra_bench: pd.DataFrame, df_shy: pd.DataFrame) -> tuple:
    """Arma, sobre la MISMA grilla semanal de la simulación de cartera, los
    retornos e índices (base 100) de A3500, CER, SHY-en-pesos y los benchmarks
    compuestos de Objetivo 1 / Objetivo 2 / Cartera Total. Los pesos del blend
    se toman de la cartera actual (dinámico, no hardcodeado), reproduciendo la
    estructura de benchmarks de la diapositiva "Medición de Desempeño" del IPS:
    Objetivo 1 = TAMAR/CER (Tramo 1/2) + ETF SHY (Tramo 3); Objetivo 2 = A3500."""
    bench = df_bcra_bench.merge(df_shy.rename(columns={"Valor": "SHY"}), on="Date", how="outer").sort_values("Date")
    aligned = _align_to_dates(bench, fechas, [c for c in ["A3500", "CER", "SHY"] if c in bench.columns])
    for c in ["A3500", "CER", "SHY"]:
        if c not in aligned.columns:
            aligned[c] = np.nan

    peso_obj1 = port_df.loc[port_df["Objetivo"].str.startswith("1"), "Peso_pct"].sum()
    peso_obj2 = port_df.loc[port_df["Objetivo"].str.startswith("2"), "Peso_pct"].sum()
    tramo12_tk = ["CAUCION", "S31L6", "TZXD6"]
    tramo3_tk = ["TLCQO", "LOC5O", "AO27"]
    peso_t12 = port_df.loc[port_df["Ticker"].isin(tramo12_tk), "Peso_pct"].sum()
    peso_t3 = port_df.loc[port_df["Ticker"].isin(tramo3_tk), "Peso_pct"].sum()
    w_cer = peso_t12 / peso_obj1 if peso_obj1 > 0 else 0.0
    w_shy = peso_t3 / peso_obj1 if peso_obj1 > 0 else 0.0

    out = aligned.copy()
    out["ret_A3500"] = out["A3500"].pct_change()
    out["ret_CER"] = out["CER"].pct_change()
    out["ret_SHY_usd"] = out["SHY"].pct_change()
    out["ret_SHY_ars"] = (1 + out["ret_SHY_usd"]) * (1 + out["ret_A3500"]) - 1
    out["ret_obj1_bench"] = w_cer * out["ret_CER"] + w_shy * out["ret_SHY_ars"]
    out["ret_obj2_bench"] = out["ret_A3500"]
    out["ret_total_bench"] = (peso_obj1 / 100.0) * out["ret_obj1_bench"] + (peso_obj2 / 100.0) * out["ret_obj2_bench"]

    for nombre, retcol in [("A3500", "ret_A3500"), ("CER", "ret_CER"), ("SHY_ars", "ret_SHY_ars"),
                           ("obj1_bench", "ret_obj1_bench"), ("obj2_bench", "ret_obj2_bench"),
                           ("total_bench", "ret_total_bench")]:
        idx = (1 + out[retcol].fillna(0)).cumprod() * 100
        if len(idx):
            idx.iloc[0] = 100.0
        out[f"idx_{nombre}"] = idx

    pesos = dict(peso_obj1=peso_obj1, peso_obj2=peso_obj2, w_cer_in_obj1=w_cer, w_shy_in_obj1=w_shy)
    return out, pesos


def objetivo_returns_from_semanal(semanal: pd.DataFrame, port_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye el valor y el retorno semanal de Objetivo 1 y Objetivo 2 por
    separado, sumando las columnas Valor_<ticker> que ya calculó la simulación
    semanal — no hace falta volver a simular nada."""
    obj1_tk = port_df.loc[port_df["Objetivo"].str.startswith("1"), "Ticker"].tolist()
    obj2_tk = port_df.loc[port_df["Objetivo"].str.startswith("2"), "Ticker"].tolist()
    cols_obj1 = [f"Valor_{t}" for t in obj1_tk if f"Valor_{t}" in semanal.columns]
    cols_obj2 = [f"Valor_{t}" for t in obj2_tk if f"Valor_{t}" in semanal.columns]

    out = semanal[["Date", "Estado"]].copy()
    out["Valor_Obj1"] = semanal[cols_obj1].sum(axis=1) if cols_obj1 else np.nan
    out["Valor_Obj2"] = semanal[cols_obj2].sum(axis=1) if cols_obj2 else np.nan
    out["Valor_Total"] = semanal["Valor_Cartera"]
    out["ret_Obj1"] = out["Valor_Obj1"].pct_change()
    out["ret_Obj2"] = out["Valor_Obj2"].pct_change()
    out["ret_Total"] = out["Valor_Total"].pct_change()
    return out


def sharpe_ratio(returns: pd.Series, rf_periodo: float, periods_per_year: int = 52) -> float:
    """Sharpe anualizado sobre retornos periódicos (semanales por defecto)."""
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    excess = r - rf_periodo
    sd = excess.std(ddof=1)
    if pd.isna(sd) or sd < 1e-10:
        return np.nan
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf_periodo: float, periods_per_year: int = 52) -> float:
    """Sortino anualizado: igual que Sharpe pero solo penaliza la volatilidad
    a la baja (retornos por debajo de la tasa libre de riesgo)."""
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    excess = r - rf_periodo
    downside = excess[excess < 0]
    if len(downside) < 2:
        return np.nan
    dd = downside.std(ddof=1)
    if pd.isna(dd) or dd < 1e-10:
        return np.nan
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def information_ratio(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 52) -> float:
    """Information Ratio anualizado: retorno activo (cartera − benchmark)
    sobre el tracking error (desvío del retorno activo)."""
    df = pd.concat([returns, benchmark], axis=1).dropna()
    if len(df) < 2:
        return np.nan
    active = df.iloc[:, 0] - df.iloc[:, 1]
    sd = active.std(ddof=1)
    if pd.isna(sd) or sd < 1e-10:
        return np.nan
    return float(active.mean() / sd * np.sqrt(periods_per_year))


def extended_weekly_returns(port_df: pd.DataFrame, df_norm: pd.DataFrame, tickers: list,
                             fecha_fin: pd.Timestamp, semanas: int) -> tuple:
    """Serie semanal de retornos PONDERADOS POR PESO (no por VN comprado) de un
    subconjunto de la cartera (ej. Objetivo 1), extendida 'semanas' semanas
    hacia atrás desde 'fecha_fin'. Existe SOLO para darle más tamaño de
    muestra al cálculo de Sharpe/Sortino/Information Ratio — no es la
    simulación de tenencia real (esa vive en 'semanal', a nominales fijos
    desde la compra). Cada semana pondera solo entre los tickers que
    efectivamente tienen dato ese día (re-normalizando por el peso
    disponible), en vez de asumir retorno cero en los que falten.
    Devuelve (df[Date, ret], tickers_sin_ningun_dato, fecha_inicio_real)."""
    fecha_fin = pd.Timestamp(fecha_fin)
    fecha_inicio = fecha_fin - pd.Timedelta(weeks=int(semanas))
    fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="W-MON")
    if fecha_fin not in fechas:
        fechas = fechas.append(pd.DatetimeIndex([fecha_fin]))
    fechas = pd.DatetimeIndex(sorted(fechas.unique()))

    sub = port_df[port_df["Ticker"].isin(tickers)].copy()
    peso_total_sub = sub["Peso_pct"].sum()
    sub["Peso_norm"] = sub["Peso_pct"] / peso_total_sub if peso_total_sub > 0 else 0.0

    bonos = sub.loc[~sub["Es_Cash"], "Ticker"].tolist()
    hist = df_norm[df_norm["Ticker"].isin(bonos)]
    piv_par = hist.pivot_table(index="Date", columns="Ticker", values="Paridad", aggfunc="first").sort_index()
    idx_union = pd.DatetimeIndex(sorted(set(list(piv_par.index)) | set(fechas)))
    piv_par = piv_par.reindex(idx_union).ffill().reindex(fechas)
    for t in bonos:
        if t not in piv_par.columns:
            piv_par[t] = np.nan
    faltan = [t for t in bonos if piv_par[t].isna().all()]

    ret_bonos = piv_par[bonos].pct_change()
    w_bonos = sub.set_index("Ticker")["Peso_norm"].reindex(bonos).fillna(0.0)
    peso_disponible = ret_bonos.notna().astype(float).mul(w_bonos, axis=1).sum(axis=1)
    contrib_bonos = ret_bonos.fillna(0.0).mul(w_bonos, axis=1).sum(axis=1)

    cash_rows = sub[sub["Es_Cash"]]
    w_cash = float(cash_rows["Peso_norm"].sum()) if not cash_rows.empty else 0.0
    tna = float(cash_rows["TNA_Manual_pct"].iloc[0]) if not cash_rows.empty else 0.0
    ret_cash_semanal = (1 + tna / 100.0) ** (7 / 365.0) - 1

    peso_total_disponible = peso_disponible + w_cash
    ret_total = (contrib_bonos + w_cash * ret_cash_semanal) / peso_total_disponible.replace(0, np.nan)
    ret_total.iloc[0] = np.nan  # la primera fecha de la serie no tiene retorno (no hay período previo)

    out = pd.DataFrame({"Date": fechas, "ret": ret_total.values,
                         "peso_cubierto_pct": (peso_total_disponible * 100).values})
    return out, faltan, fecha_inicio


# =====================================================================
# Motor de cartera: snapshot ponderado y KPIs (para "Cartera Hoy")
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
                peso_cubierto=peso_valido * 100, peso_total=snap["Peso_frac"].sum() * 100)


# =====================================================================
# Simulación de compra (29/6/2026) y seguimiento semanal a nominales fijos
# =====================================================================

def compute_compra(port_df: pd.DataFrame, df_norm: pd.DataFrame, fecha_compra: pd.Timestamp,
                    monto_total: float) -> pd.DataFrame:
    """Simula la compra de TODA la cartera en 'fecha_compra' con 'monto_total'
    de capital, repartido según Peso_pct. El VN comprado de cada bono surge de
    dividir el monto asignado por su Paridad (precio) en esa fecha exacta. Para
    Cash, el 'VN' es directamente el monto (no hay concepto de paridad — crece
    por TNA, no por precio)."""
    port = port_df.copy()
    fecha_compra = pd.Timestamp(fecha_compra)
    bond_tickers = port.loc[~port["Es_Cash"], "Ticker"].tolist()
    mkt = snapshot_asof(df_norm, bond_tickers, fecha_compra) if bond_tickers else pd.DataFrame()

    montos, precios, vns, fechas_precio = [], [], [], []
    for _, r in port.iterrows():
        monto_i = monto_total * (r["Peso_pct"] / 100.0)
        montos.append(monto_i)
        if r["Es_Cash"]:
            precios.append(100.0)
            vns.append(monto_i)
            fechas_precio.append(fecha_compra)
        else:
            m = mkt[mkt["Ticker"] == r["Ticker"]] if not mkt.empty else pd.DataFrame()
            if not m.empty and pd.notna(m.iloc[0].get("Paridad")):
                p = float(m.iloc[0]["Paridad"])
                precios.append(p)
                vns.append(monto_i / (p / 100.0) if p > 0 else np.nan)
                fechas_precio.append(m.iloc[0]["Date"])
            else:
                precios.append(np.nan)
                vns.append(np.nan)
                fechas_precio.append(pd.NaT)

    port["Monto_Invertido"] = montos
    port["Precio_Compra"] = precios
    port["VN_Comprado"] = vns
    port["Fecha_Precio_Compra"] = fechas_precio
    return port


def simulacion_semanal(port_compra: pd.DataFrame, df_norm: pd.DataFrame, fecha_compra: pd.Timestamp,
                        as_of_hoy: pd.Timestamp, fecha_fin: pd.Timestamp) -> pd.DataFrame:
    """Valor de la cartera CADA LUNES desde 'fecha_compra' hasta 'fecha_fin', a
    NOMINALES FIJOS (los comprados en fecha_compra — sin rebalanceo). Las fechas
    <= as_of_hoy usan precio real de mercado (Paridad, con forward-fill para
    instrumentos ilíquidos); las posteriores se proyectan por devengamiento a
    la TIR vigente en as_of_hoy (constante desde ahí — no es una predicción de
    mercado, es el piso esperado por carry)."""
    fecha_compra = pd.Timestamp(fecha_compra)
    as_of_hoy = pd.Timestamp(as_of_hoy)
    fecha_fin = pd.Timestamp(fecha_fin)

    lunes = list(pd.date_range(start=fecha_compra, end=max(fecha_fin, fecha_compra), freq="W-MON"))
    fechas = sorted(set(lunes + [fecha_compra, as_of_hoy]))
    fechas = [f for f in fechas if f >= fecha_compra]

    bonos = port_compra.loc[~port_compra["Es_Cash"], "Ticker"].tolist()
    hist_bonos = df_norm[df_norm["Ticker"].isin(bonos)] if bonos else pd.DataFrame(columns=["Date", "Ticker"])

    piv_par = hist_bonos.pivot_table(index="Date", columns="Ticker", values="Paridad", aggfunc="first")
    piv_tir = hist_bonos.pivot_table(index="Date", columns="Ticker", values="TIR", aggfunc="first")
    idx_union = pd.DatetimeIndex(sorted(set(list(piv_par.index) + fechas + [as_of_hoy])))
    piv_par = piv_par.reindex(idx_union).ffill()
    piv_tir = piv_tir.reindex(idx_union).ffill()

    filas = []
    for f in fechas:
        valor_f = 0.0
        peso_con_dato = 0.0
        detalle = {}
        for _, r in port_compra.iterrows():
            vn = r["VN_Comprado"]
            if pd.isna(vn):
                continue
            if r["Es_Cash"]:
                dias = (f - fecha_compra).days
                tna = r.get("TNA_Manual_pct", 0.0) or 0.0
                val = vn * (1 + tna / 100.0) ** (dias / 365.0)
                valor_f += val
                peso_con_dato += r["Peso_pct"]
                detalle[r["Ticker"]] = val
            else:
                tk = r["Ticker"]
                if tk not in piv_par.columns:
                    continue
                if f <= as_of_hoy and f in piv_par.index:
                    p = piv_par.loc[f, tk]
                else:
                    p_hoy = piv_par.loc[as_of_hoy, tk] if as_of_hoy in piv_par.index else np.nan
                    tir_hoy = piv_tir.loc[as_of_hoy, tk] if as_of_hoy in piv_tir.index else np.nan
                    if pd.notna(p_hoy) and pd.notna(tir_hoy):
                        dias_proy = (f - as_of_hoy).days
                        p = p_hoy * (1 + tir_hoy / 100.0) ** (dias_proy / 365.0)
                    else:
                        p = np.nan
                if pd.notna(p):
                    val = vn * (p / 100.0)
                    valor_f += val
                    peso_con_dato += r["Peso_pct"]
                    detalle[tk] = val
        fila = dict(Date=f, Valor_Cartera=valor_f, Peso_Cubierto_pct=peso_con_dato)
        fila.update({f"Valor_{k}": v for k, v in detalle.items()})
        filas.append(fila)

    out = pd.DataFrame(filas).sort_values("Date").reset_index(drop=True)
    out["Dias_Periodo"] = out["Date"].diff().dt.days
    out["Rendimiento_Semanal_%"] = out["Valor_Cartera"].pct_change() * 100
    out["Rendimiento_Acumulado_%"] = (out["Valor_Cartera"] / out["Valor_Cartera"].iloc[0] - 1) * 100
    out["Estado"] = np.select(
        [out["Date"] == fecha_compra, out["Date"] == as_of_hoy, out["Date"] > as_of_hoy],
        ["Compra", "Hoy", "Proyectado"], default="Realizado")
    return out


def build_excel_snapshot(tabla_hoy: pd.DataFrame, kpis: dict, port_compra: pd.DataFrame = None,
                          semanal: pd.DataFrame = None) -> bytes:
    """Informe simple en Excel (foto del momento): resumen de KPIs, tabla de
    holdings, detalle de la compra del 29/6 y la simulación semanal. Sin
    fórmulas ni gráficos nativos — es un respaldo de datos, no un modelo vivo."""
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
        if port_compra is not None and not port_compra.empty:
            port_compra.to_excel(writer, sheet_name="Compra 29-06", index=False)
        if semanal is not None and not semanal.empty:
            cols = [c for c in semanal.columns if not c.startswith("Valor_") or c == "Valor_Cartera"]
            semanal[cols].to_excel(writer, sheet_name="Simulacion Semanal", index=False)
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
    st.subheader("💰 Simulación de compra")
    monto_compra = st.number_input("Monto invertido el día de la compra (ARS)",
                                    value=175_000_000.0, step=1_000_000.0, format="%.0f")
    fecha_compra = pd.Timestamp(st.date_input("Fecha de compra", value=datetime(2026, 6, 29)))
    fecha_fin_sim = pd.Timestamp(st.date_input("Ver la cartera semana a semana hasta",
                                                value=datetime(2026, 12, 31)))

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
    st.warning("⚠️ Sin datos de mercado del dataset principal (ONs/Bonos/Soberanos): las pestañas "
               "'Cartera Hoy', 'Simulación Semanal' y 'Benchmarks & Ratios' quedan sin contenido hasta "
               "que cargues la Alphacast API Key o actives el modo de ejemplo. La pestaña "
               "'Brecha Cambiaria' funciona igual, porque usa la API pública del BCRA.")
    as_of_hoy = pd.Timestamp(datetime.now().date())
    snap_hoy = pd.DataFrame()
    kpis_hoy = dict(tir_cartera=np.nan, md_cartera=np.nan, peso_cubierto=0.0, peso_total=0.0)
    port_compra = pd.DataFrame()
    semanal = pd.DataFrame()
    valor_actual_hoy = np.nan
    resultado_pct_hoy = np.nan
else:
    as_of_hoy = pd.Timestamp(fecha_datos)
    snap_hoy = build_holdings_snapshot(port_df, df_norm, as_of_hoy)
    kpis_hoy = portfolio_kpis(snap_hoy)

    # Compra del 29/6 y simulación semanal — se calculan UNA vez y se reusan en
    # todas las pestañas (única fuente de verdad).
    port_compra = compute_compra(port_df, df_norm, fecha_compra, monto_compra)
    semanal = simulacion_semanal(port_compra, df_norm, fecha_compra, as_of_hoy, fecha_fin_sim)

    fila_hoy = semanal[semanal["Estado"] == "Hoy"]
    valor_actual_hoy = float(fila_hoy["Valor_Cartera"].iloc[0]) if not fila_hoy.empty else np.nan
    resultado_pct_hoy = float(fila_hoy["Rendimiento_Acumulado_%"].iloc[0]) if not fila_hoy.empty else np.nan

tab_hoy, tab_sim, tab_bench, tab_brecha, tab_metodo = st.tabs(
    ["🏠 Cartera Hoy", "📊 Simulación Semanal", "📐 Benchmarks & Ratios",
     "💵 Brecha Cambiaria", "ℹ️ Metodología"]
)

# ---------------------------------------------------------------------
# TAB 1 — Cartera Hoy
# ---------------------------------------------------------------------
with tab_hoy:
    if not data_ok:
        st.info("Cargá datos de mercado (Alphacast real o modo de ejemplo) para ver esta pestaña.")
    else:
        no_encontrados = snap_hoy.loc[~snap_hoy["Encontrado"], "Ticker"].tolist()
        if no_encontrados:
            st.info(f"Sin dato de mercado para: **{', '.join(no_encontrados)}**. No se incluyen en la TIR/Duration "
                    f"ponderada ni en la simulación de compra hasta que tengan cotización en el dataset.")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("TIR ponderada de la cartera", f"{kpis_hoy['tir_cartera']:.2f}%" if pd.notna(kpis_hoy['tir_cartera']) else "—")
        k2.metric("Duration (MD) ponderada", f"{kpis_hoy['md_cartera']:.2f} años" if pd.notna(kpis_hoy['md_cartera']) else "—")
        k3.metric("Monto invertido (29/6)", fmt_ars(monto_compra))
        k4.metric("Valor actual (hoy)", fmt_ars(valor_actual_hoy),
                  f"{resultado_pct_hoy:+.2f}% desde la compra" if pd.notna(resultado_pct_hoy) else None)
        st.caption(f"Snapshot al {as_of_hoy.strftime('%d/%m/%Y')}" + (" (datos de ejemplo)" if modo_demo else "") +
                   f" · Compra simulada el {fecha_compra.strftime('%d/%m/%Y')}")

        st.divider()
        p1, p2, p3 = st.columns(3)

        with p1:
            g = snap_hoy.groupby("Objetivo")["Peso_pct"].sum().reset_index()
            fig = px.pie(g, names="Objetivo", values="Peso_pct", hole=0.45,
                         color="Objetivo", color_discrete_map={
                             "1 · Capital de Trabajo": COLORS["obj1"], "2 · Cobertura FX Préstamo": COLORS["obj2"]})
            fig.update_traces(textinfo="label+percent", textposition="outside",
                               textfont=dict(color="white", size=12),
                               outsidetextfont=dict(color="white", size=12))
            fig.update_layout(title="Por Objetivo", showlegend=False, height=380, **PLOTLY_LAYOUT)
            fig.update_layout(title_font=dict(color="white"), font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True, key="pie_objetivo")

        with p2:
            g = snap_hoy.groupby("Tramo")["Peso_pct"].sum().reset_index()
            fig = px.pie(g, names="Tramo", values="Peso_pct", hole=0.45)
            fig.update_traces(textinfo="label+percent", textposition="outside",
                               marker=dict(colors=px.colors.qualitative.Safe),
                               textfont=dict(color="white", size=12),
                               outsidetextfont=dict(color="white", size=12))
            fig.update_layout(title="Por Tramo", showlegend=False, height=380, **PLOTLY_LAYOUT)
            fig.update_layout(title_font=dict(color="white"), font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True, key="pie_tramo")

        with p3:
            g = snap_hoy.groupby("Ticker")["Peso_pct"].sum().reset_index()
            fig = px.pie(g, names="Ticker", values="Peso_pct", hole=0.45)
            fig.update_traces(textinfo="label+percent", textposition="outside",
                               marker=dict(colors=px.colors.qualitative.Set2),
                               textfont=dict(color="white", size=12),
                               outsidetextfont=dict(color="white", size=12))
            fig.update_layout(title="Por Instrumento", showlegend=False, height=380, **PLOTLY_LAYOUT)
            fig.update_layout(title_font=dict(color="white"), font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True, key="pie_instrumento")

        st.divider()
        st.subheader("Detalle de holdings — compra vs. hoy")
        detalle = snap_hoy.merge(
            port_compra[["Ticker", "Monto_Invertido", "Precio_Compra", "VN_Comprado"]], on="Ticker", how="left")
        detalle["Valor_Actual"] = np.where(
            detalle["Es_Cash"],
            detalle["Monto_Invertido"] * (1 + detalle["TNA_Manual_pct"] / 100.0) ** ((as_of_hoy - fecha_compra).days / 365.0),
            detalle["VN_Comprado"] * detalle["Paridad"] / 100.0,
        )
        detalle["Resultado_%"] = (detalle["Valor_Actual"] / detalle["Monto_Invertido"] - 1) * 100
        cols_show = ["Ticker", "Descripcion", "Objetivo", "Peso_pct", "Clase", "TIR", "MD",
                     "Monto_Invertido", "Precio_Compra", "VN_Comprado", "Valor_Actual", "Resultado_%"]
        tabla = detalle[[c for c in cols_show if c in detalle.columns]].copy()
        st.dataframe(tabla.sort_values("Peso_pct", ascending=False), use_container_width=True, height=300,
                     column_config={
                         "Monto_Invertido": st.column_config.NumberColumn("Monto invertido", format="$ %.0f"),
                         "Valor_Actual": st.column_config.NumberColumn("Valor actual", format="$ %.0f"),
                         "Resultado_%": st.column_config.NumberColumn("Resultado (%)", format="%.2f%%"),
                     })

        csv = tabla.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar tabla (CSV)", data=csv,
                            file_name=f"cartera_grupo8_{as_of_hoy.strftime('%Y%m%d')}.csv", mime="text/csv")

    # ---------------------------------------------------------------------
    # TAB 2 — Simulación Semanal
    # ---------------------------------------------------------------------
with tab_sim:
    if not data_ok:
        st.info("Cargá datos de mercado (Alphacast real o modo de ejemplo) para ver esta pestaña.")
    else:
        st.markdown("### 📊 Simulación: compra el 29/6/2026 y seguimiento semanal")
        st.caption("Se compran los 8 instrumentos el día de la compra con el monto asignado, a precio de "
                   "mercado de ese día. De ahí en más los NOMINALES quedan fijos (sin rebalanceo): lo que "
                   "cambia cada lunes es el precio. Tramo sólido = precio real de mercado. Tramo punteado = "
                   "proyección por devengamiento a la TIR vigente hoy (no es una predicción de precios).")

        cols_compra = ["Ticker", "Descripcion", "Peso_pct", "Monto_Invertido", "Precio_Compra", "VN_Comprado"]
        st.dataframe(port_compra[cols_compra].rename(columns={"Peso_pct": "Peso (%)"}),
                     use_container_width=True, hide_index=True, height=300,
                     column_config={
                         "Monto_Invertido": st.column_config.NumberColumn("Monto invertido", format="$ %.0f"),
                         "Precio_Compra": st.column_config.NumberColumn("Precio de compra (Paridad)", format="%.2f"),
                         "VN_Comprado": st.column_config.NumberColumn("VN comprado", format="%.0f"),
                     })
        sin_precio = port_compra.loc[port_compra["VN_Comprado"].isna(), "Ticker"].tolist()
        if sin_precio:
            st.warning(f"Sin precio de mercado en la fecha de compra para: **{', '.join(sin_precio)}** — "
                       f"no se pueden calcular sus nominales y quedan afuera de la simulación.")

        st.divider()
        realizado = semanal[semanal["Estado"].isin(["Compra", "Hoy", "Realizado"])].sort_values("Date")
        proyectado = semanal[semanal["Estado"].isin(["Hoy", "Proyectado"])].sort_values("Date")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=realizado["Date"], y=realizado["Valor_Cartera"], mode="lines+markers",
                                  line=dict(width=2.6, color=COLORS["primary"]), name="Realizado"))
        fig.add_trace(go.Scatter(x=proyectado["Date"], y=proyectado["Valor_Cartera"], mode="lines+markers",
                                  line=dict(width=2.2, color=COLORS["primary"], dash="dot"), name="Proyectado"))
        fig.add_vline(x=as_of_hoy, line_dash="dash", line_color=COLORS["accent"],
                      annotation_text="Hoy", annotation_font_color=COLORS["accent"])

        fecha_desembolso_ts = pd.Timestamp(fecha_desembolso)
        hitos = {"Mes 3 (20% obra civil)": fecha_desembolso_ts + pd.Timedelta(days=90),
                 "Mes 6 (30% obra civil)": fecha_desembolso_ts + pd.Timedelta(days=180),
                 "Mes 9 (50% + maquinaria)": fecha_desembolso_ts + pd.Timedelta(days=270)}
        for nombre, fecha_hito in hitos.items():
            if semanal["Date"].min() <= fecha_hito <= semanal["Date"].max():
                fig.add_vline(x=fecha_hito, line_dash="dot", line_color=COLORS["obj2"],
                              annotation_text=nombre, annotation_font_size=9, annotation_font_color=COLORS["obj2"])

        fig.update_layout(title="Valor de la cartera cada lunes (nominales fijos desde la compra)",
                           height=440, **PLOTLY_LAYOUT)
        style_axes(fig, "Fecha", "Valor de la cartera (ARS)")
        watermark(fig, as_of_hoy, "Alphacast" if not modo_demo else "Ejemplo")
        st.plotly_chart(fig, use_container_width=True, key=f"sim_valor_{fecha_compra}_{fecha_fin_sim}")

        fig2 = px.bar(semanal.iloc[1:], x="Date", y="Rendimiento_Semanal_%",
                      color=semanal.iloc[1:]["Rendimiento_Semanal_%"] >= 0,
                      color_discrete_map={True: COLORS["ok"], False: COLORS["warn"]})
        fig2.add_vline(x=as_of_hoy, line_dash="dash", line_color=COLORS["accent"])
        fig2.update_layout(title="Rendimiento semanal de la cartera (%)", height=340, **PLOTLY_LAYOUT, showlegend=False)
        style_axes(fig2, "Fecha", "Rendimiento semanal (%)")
        st.plotly_chart(fig2, use_container_width=True, key=f"sim_rend_{fecha_compra}_{fecha_fin_sim}")

        st.subheader("Tabla semanal completa")
        tabla_sem = semanal[["Date", "Estado", "Valor_Cartera", "Peso_Cubierto_pct",
                              "Rendimiento_Semanal_%", "Rendimiento_Acumulado_%"]].copy()
        st.dataframe(tabla_sem, use_container_width=True, height=320,
                     column_config={
                         "Valor_Cartera": st.column_config.NumberColumn("Valor de la cartera", format="$ %.0f"),
                         "Peso_Cubierto_pct": st.column_config.NumberColumn("Peso con dato (%)", format="%.1f%%"),
                         "Rendimiento_Semanal_%": st.column_config.NumberColumn("Rend. semanal", format="%.2f%%"),
                         "Rendimiento_Acumulado_%": st.column_config.NumberColumn("Rend. acumulado", format="%.2f%%"),
                     })
        csv_sem = tabla_sem.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar simulación semanal (CSV)", data=csv_sem,
                            file_name=f"simulacion_semanal_grupo8_{as_of_hoy.strftime('%Y%m%d')}.csv", mime="text/csv")

    # ---------------------------------------------------------------------
    # TAB 3 — Benchmarks & Ratios
    # ---------------------------------------------------------------------
with tab_bench:
    if not data_ok:
        st.info("Cargá datos de mercado (Alphacast real o modo de ejemplo) para ver esta pestaña.")
    else:
        st.markdown("### 📐 Comparación con Benchmarks")
        st.caption("A3500 y CER: API pública del BCRA (sin API key). SHY (ETF de bonos del Tesoro de "
                   "EE.UU. 1-3 años, convertido a pesos vía A3500): API pública de Yahoo Finance. Todo "
                   "en base 100 al día de la compra.")

        desde_bm = fecha_compra
        hasta_bm = pd.Timestamp(semanal["Date"].max())

        if modo_demo:
            df_bench_raw = synthetic_benchmark_series(desde_bm.strftime("%Y-%m-%d"), hasta_bm.strftime("%Y-%m-%d"))
            df_shy = df_bench_raw[["Date", "SHY"]].rename(columns={"SHY": "Valor"})
            df_bench_bcra = df_bench_raw[["Date", "A3500", "CER"]]
        else:
            df_bench_bcra = fetch_benchmark_bcra(desde_bm.strftime("%Y-%m-%d"), hasta_bm.strftime("%Y-%m-%d"))
            df_shy = fetch_shy_series(desde_bm.strftime("%Y-%m-%d"), hasta_bm.strftime("%Y-%m-%d"))
            if df_shy.empty:
                st.warning("No se pudo descargar SHY de Yahoo Finance (puede ser un bloqueo temporal del "
                           "servicio). El benchmark de Objetivo 1 / Tramo 3 queda incompleto hasta reintentar.")

        fechas_grid = pd.DatetimeIndex(semanal["Date"])
        bench_tabla, pesos_bench = build_benchmark_table(port_df, fechas_grid, df_bench_bcra, df_shy)

        cartera_idx = 100 * semanal["Valor_Cartera"] / semanal["Valor_Cartera"].iloc[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=semanal["Date"], y=cartera_idx, mode="lines", name="Cartera",
                                  line=dict(width=2.8, color=COLORS["primary"])))
        fig.add_trace(go.Scatter(x=bench_tabla["Date"], y=bench_tabla["idx_A3500"], mode="lines", name="A3500",
                                  line=dict(width=1.8, color=COLORS["obj2"], dash="dash")))
        fig.add_trace(go.Scatter(x=bench_tabla["Date"], y=bench_tabla["idx_CER"], mode="lines", name="CER",
                                  line=dict(width=1.8, color=COLORS["cer"], dash="dash")))
        fig.add_trace(go.Scatter(x=bench_tabla["Date"], y=bench_tabla["idx_SHY_ars"], mode="lines",
                                  name="SHY (en pesos)", line=dict(width=1.8, color=COLORS["accent"], dash="dot")))
        fig.add_vline(x=as_of_hoy, line_dash="dash", line_color="#94a3b8", annotation_text="Hoy")
        fig.update_layout(title="Cartera vs. Benchmarks (índice base 100 desde la compra)",
                           height=440, **PLOTLY_LAYOUT)
        style_axes(fig, "Fecha", "Índice (base 100)")
        watermark(fig, hasta_bm, "BCRA + Yahoo Finance" if not modo_demo else "Ejemplo")
        st.plotly_chart(fig, use_container_width=True, key=f"bench_idx_{desde_bm}_{hasta_bm}")

        st.caption(f"Benchmark compuesto — Objetivo 1: {pesos_bench['w_cer_in_obj1']*100:.0f}% CER + "
                   f"{pesos_bench['w_shy_in_obj1']*100:.0f}% SHY(en pesos) · Objetivo 2: 100% A3500 · "
                   f"Total: {pesos_bench['peso_obj1']:.1f}% × benchmark Obj.1 + {pesos_bench['peso_obj2']:.1f}% "
                   f"× benchmark Obj.2 (misma estructura que 'Medición de Desempeño' del IPS).")

        ultimos = bench_tabla.dropna(subset=["idx_A3500"])
        if not ultimos.empty:
            u = ultimos.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cartera (acumulado)", f"{cartera_idx.iloc[-1] - 100:+.1f}%")
            c2.metric("A3500 (acumulado)", f"{u['idx_A3500'] - 100:+.1f}%")
            c3.metric("CER (acumulado)", f"{u['idx_CER'] - 100:+.1f}%" if pd.notna(u["idx_CER"]) else "—")
            c4.metric("SHY en pesos (acumulado)", f"{u['idx_SHY_ars'] - 100:+.1f}%" if pd.notna(u["idx_SHY_ars"]) else "—")

        st.divider()
        st.markdown("### 📊 Ratios de riesgo-retorno — Objetivo 1 (Capital de Trabajo)")
        st.caption("Sharpe, Sortino e Information Ratio tienen sentido económico en Objetivo 1 (busca "
                   "capturar tasa/carry). En Objetivo 2 y en la Cartera Total **no** se muestran estos "
                   "ratios: son cobertura cambiaria, no gestión activa de retorno — ver más abajo la "
                   "métrica de efectividad de cobertura, que es la que corresponde a ese mandato.")

        semanas_ext = st.slider("Semanas de historia para calcular estos ratios y la cobertura de Objetivo 2",
                                 min_value=8, max_value=104, value=26, step=2,
                                 help="Amplía la muestra hacia atrás usando una serie ponderada por peso "
                                      "(no por VN comprado) de los mismos instrumentos — SOLO para que "
                                      "estos indicadores tengan una base estadística más razonable que "
                                      "las pocas semanas reales desde la compra. Se usa tanto para los "
                                      "ratios de Objetivo 1 como para el Hedge Ratio de Objetivo 2, más abajo.")

        obj1_tickers = port_df.loc[port_df["Objetivo"].str.startswith("1"), "Ticker"].tolist()
        obj2_tickers = port_df.loc[port_df["Objetivo"].str.startswith("2"), "Ticker"].tolist()
        ext_ret, ext_faltan, fecha_inicio_ext = extended_weekly_returns(
            port_df, df_norm, obj1_tickers, as_of_hoy, semanas_ext)
        n_semanas_ext = int(ext_ret["ret"].notna().sum())

        df_bench_ext = fetch_benchmark_bcra(fecha_inicio_ext.strftime("%Y-%m-%d"), as_of_hoy.strftime("%Y-%m-%d")) \
            if not modo_demo else synthetic_benchmark_series(fecha_inicio_ext.strftime("%Y-%m-%d"), as_of_hoy.strftime("%Y-%m-%d"))
        df_shy_ext = fetch_shy_series(fecha_inicio_ext.strftime("%Y-%m-%d"), as_of_hoy.strftime("%Y-%m-%d")) \
            if not modo_demo else df_bench_ext[["Date", "SHY"]].rename(columns={"SHY": "Valor"})
        if modo_demo:
            df_bench_ext = df_bench_ext[["Date", "A3500", "CER"]]
        bench_ext, _ = build_benchmark_table(port_df, pd.DatetimeIndex(ext_ret["Date"]), df_bench_ext, df_shy_ext)

        if n_semanas_ext < 8:
            st.warning(f"⚠️ Solo **{n_semanas_ext} semana(s)** con dato utilizable en la ventana elegida "
                       f"({fecha_inicio_ext.strftime('%d/%m/%Y')} en adelante). Con tan poca muestra, estos "
                       f"ratios siguen siendo poco confiables — probá ampliar la ventana si el dataset lo permite.")
        if ext_faltan:
            st.caption(f"Sin ninguna historia en el dataset para: **{', '.join(ext_faltan)}** dentro de la "
                       f"ventana elegida — quedan afuera del cálculo (no se les asume retorno cero).")

        merged_ext = ext_ret.merge(bench_ext, on="Date", how="left")

        # Sharpe/Sortino SIN tasa libre de riesgo (retorno/desvío puro). Restar la TNA
        # de la caución (una tasa nominal alta, ~30%) contra retornos de precio (Paridad)
        # de instrumentos de menor volatilidad exagera el "excess return" negativo y
        # distorsiona el número — por eso se usa rf=0 acá.
        sh = sharpe_ratio(merged_ext["ret"], 0.0)
        so = sortino_ratio(merged_ext["ret"], 0.0)
        ir = information_ratio(merged_ext["ret"], merged_ext["ret_obj1_bench"])

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Sharpe (sin tasa libre de riesgo)", f"{sh:.2f}" if pd.notna(sh) else "—")
        r2.metric("Sortino (sin tasa libre de riesgo)", f"{so:.2f}" if pd.notna(so) else "—")
        r3.metric("Information Ratio", f"{ir:.2f}" if pd.notna(ir) else "—")
        r4.metric("N° semanas usadas", f"{n_semanas_ext}")
        st.caption("Sharpe = retorno medio semanal ÷ desvío semanal (anualizado ×√52) — **sin** restar la "
                   "TNA de la caución. Sortino: igual, pero el desvío solo considera semanas con retorno "
                   "negativo. Se omite la tasa libre de riesgo porque estos retornos vienen de la Paridad "
                   "(precio de mercado) y no siempre son directamente comparables contra una tasa nominal "
                   "de money-market — restarla puede exagerar artificialmente un Sharpe negativo. "
                   f"Benchmark del Information Ratio: el compuesto de Objetivo 1 (CER + SHY). "
                   f"Ventana usada: {fecha_inicio_ext.strftime('%d/%m/%Y')} → {as_of_hoy.strftime('%d/%m/%Y')}.")

        st.divider()
        st.markdown("### 🛡️ Efectividad de Cobertura — Objetivo 2 y Cartera Total")
        st.caption("Objetivo 2 y la Cartera Total (86% Dollar-Linked) no buscan retorno ajustado por "
                   "riesgo: buscan CALZAR una obligación en dólares. Por eso se miden con métricas de "
                   "cobertura, no con Sharpe/Sortino/Information Ratio.")

        st.markdown("**Objetivo 2 · Calidad del hedge (dollar-offset)**")
        ext_ret_obj2, ext_faltan_obj2, _ = extended_weekly_returns(
            port_df, df_norm, obj2_tickers, as_of_hoy, semanas_ext)
        merged_obj2 = ext_ret_obj2.merge(bench_ext[["Date", "A3500"]], on="Date", how="left")
        merged_obj2["ret_A3500"] = merged_obj2["A3500"].pct_change()
        n_semanas_hedge = int(merged_obj2["ret"].notna().sum())

        # IMPORTANTE: acumular ambas series (Objetivo 2 y A3500) sobre el MISMO
        # sub-período con dato real de Objetivo 2 — nunca completar con 0% los
        # huecos de un lado y no del otro. Rellenar con 0 solo la cartera (porque
        # D31M7/D30S6 son instrumentos jóvenes y no tienen historia en todo el
        # rango pedido) mientras el A3500 sí acumula devaluación en esas mismas
        # semanas "vacías" infla artificialmente la devaluación del denominador
        # y hunde el Hedge Ratio, aunque la cobertura esté funcionando bien.
        validos_obj2 = merged_obj2.dropna(subset=["ret"])
        if len(validos_obj2) >= 8:
            fecha_ini_valida = validos_obj2["Date"].iloc[0]
            fecha_fin_valida = validos_obj2["Date"].iloc[-1]
            cum_obj2 = (1 + validos_obj2["ret"]).cumprod()
            cum_a3500 = (1 + validos_obj2["ret_A3500"].fillna(0)).cumprod()
            ret_obj2_cum = cum_obj2.iloc[-1] - 1
            ret_a3500_cum = cum_a3500.iloc[-1] - 1
            hedge_ratio = (ret_obj2_cum / ret_a3500_cum * 100) if abs(ret_a3500_cum) > 1e-6 else np.nan
            corr_obj2_a3500 = validos_obj2["ret"].corr(validos_obj2["ret_A3500"])
            h1, h2, h3 = st.columns(3)
            h1.metric("Hedge Ratio (dollar-offset)", f"{hedge_ratio:.0f}%" if pd.notna(hedge_ratio) else "—",
                      "100% = calzó 1:1 con la devaluación")
            h2.metric("Correlación semanal vs. A3500", f"{corr_obj2_a3500:.2f}" if pd.notna(corr_obj2_a3500) else "—")
            h3.metric("N° semanas usadas", f"{n_semanas_hedge}")
            st.caption(f"Hedge Ratio = retorno acumulado de Objetivo 2 ÷ devaluación acumulada de A3500, "
                       f"ambos calculados sobre el **mismo período con dato real**: "
                       f"{fecha_ini_valida.strftime('%d/%m/%Y')} → {fecha_fin_valida.strftime('%d/%m/%Y')} "
                       f"({n_semanas_hedge} semanas — puede ser más corto que la ventana elegida arriba si "
                       f"D31M7/D30S6 no tienen historia en todo ese rango, algo esperable en instrumentos "
                       f"jóvenes). >100%: la cobertura rindió por encima de la pura devaluación (spread "
                       f"propio de los DL). <100%: quedó por detrás. La correlación debería ser fuertemente "
                       f"positiva si el hedge funciona bien.")
        else:
            st.warning(f"⚠️ Solo **{n_semanas_hedge} semana(s)** con dato utilizable en la ventana elegida — "
                       f"muy poco para que el Hedge Ratio o la correlación signifiquen algo. Probá ampliar "
                       f"la ventana arriba (D31M7/D30S6 son instrumentos jóvenes: puede que el dataset "
                       f"tampoco tenga mucha más historia disponible).")
        if ext_faltan_obj2:
            st.caption(f"Sin ninguna historia en el dataset para: **{', '.join(ext_faltan_obj2)}** dentro "
                       f"de la ventana elegida.")

        st.markdown("**Cartera Total · Cobertura de las necesidades en USD del cronograma del préstamo**")
        fecha_desembolso_ts = pd.Timestamp(fecha_desembolso)
        hitos_usd = [("Mes 3 (obra civil)", 90, 7300), ("Mes 6 (obra civil)", 180, 10500),
                     ("Mes 9 (obra civil + maquinaria)", 270, 16700 + 50000)]
        filas_usd = []
        for nombre, dias, monto_usd in hitos_usd:
            fecha_hito = fecha_desembolso_ts + pd.Timedelta(days=dias)
            if fecha_hito < semanal["Date"].min() or fecha_hito > semanal["Date"].max():
                filas_usd.append(dict(Hito=nombre, Fecha=fecha_hito, USD_Necesario=monto_usd,
                                       Valor_Obj2_ARS=np.nan, FX=np.nan, Valor_Obj2_USD=np.nan, Cobertura_pct=np.nan))
                continue
            idx_cercano = (semanal["Date"] - fecha_hito).abs().idxmin()
            fila_sem = semanal.loc[idx_cercano]
            fila_bench = bench_tabla.loc[(bench_tabla["Date"] - fecha_hito).abs().idxmin()]
            valor_obj2_ars = fila_sem.get("Valor_D31M7", 0.0) + fila_sem.get("Valor_D30S6", 0.0)
            fx = fila_bench.get("A3500", np.nan)
            valor_obj2_usd = valor_obj2_ars / fx if pd.notna(fx) and fx > 0 else np.nan
            cobertura = (valor_obj2_usd / monto_usd * 100) if pd.notna(valor_obj2_usd) else np.nan
            filas_usd.append(dict(Hito=nombre, Fecha=fila_sem["Date"], USD_Necesario=monto_usd,
                                   Valor_Obj2_ARS=valor_obj2_ars, FX=fx, Valor_Obj2_USD=valor_obj2_usd,
                                   Cobertura_pct=cobertura))
        tabla_usd = pd.DataFrame(filas_usd)
        st.dataframe(tabla_usd, use_container_width=True, hide_index=True,
                     column_config={
                         "Fecha": st.column_config.DateColumn("Fecha (más cercana)", format="DD/MM/YYYY"),
                         "USD_Necesario": st.column_config.NumberColumn("USD necesario", format="US$ %.0f"),
                         "Valor_Obj2_ARS": st.column_config.NumberColumn("Valor Objetivo 2 (ARS)", format="$ %.0f"),
                         "FX": st.column_config.NumberColumn("A3500 en la fecha", format="%.0f"),
                         "Valor_Obj2_USD": st.column_config.NumberColumn("Valor Objetivo 2 (USD)", format="US$ %.0f"),
                         "Cobertura_pct": st.column_config.NumberColumn("Cobertura", format="%.0f%%"),
                     })
        st.caption("Convierte el valor de Objetivo 2 (D31M7+D30S6, realizado o proyectado según la fecha) "
                   "a USD con el A3500 de esa misma fecha, y lo compara contra el monto en USD que exige "
                   "cada hito del cronograma de desembolsos (según la fecha de desembolso del préstamo, "
                   "en la barra lateral). ≥100% = la cobertura alcanza para ese hito.")

    # ---------------------------------------------------------------------
    # TAB 5 — Brecha Cambiaria
    # ---------------------------------------------------------------------
with tab_brecha:
    st.markdown("### 💵 Dólar y Brecha Cambiaria")
    st.caption("Oficial, minorista y bandas de flotación: API pública del BCRA (no requiere API key). "

               "MEP y CCL: dataset de Alphacast — mismo mecanismo de API key que el resto del panel.")

    cb1, cb2 = st.columns(2)
    desde_brecha = pd.Timestamp(cb1.date_input("Desde", value=pd.Timestamp(REGIMEN_BANDAS_INICIO),
                                                key="desde_brecha"))
    hasta_brecha = pd.Timestamp(cb2.date_input("Hasta", value=as_of_hoy, key="hasta_brecha"))

    with st.expander("⚙️ Fuente de MEP/CCL (Alphacast) — opcional"):
        fx_dataset_id = st.number_input("Dataset Alphacast (FX premiums)", value=int(FX_DATASET_ID), step=1)
        st.caption("Si Alphacast reorganiza este dataset y el auto-detect de columnas falla, "
                   "cambiá el ID acá. El resto de la app no se ve afectado.")

    df_bcra, faltantes_bcra = fetch_bcra_fx_bundle(desde_brecha.strftime("%Y-%m-%d"), hasta_brecha.strftime("%Y-%m-%d"))
    if faltantes_bcra:
        st.warning(f"La API del BCRA no devolvió datos para: **{', '.join(faltantes_bcra)}**. "
                   f"Puede ser un corte temporal del servicio — probá 'Refrescar' más tarde.")

    df_mep_ccl = pd.DataFrame()
    fuente_mep_ccl = ""
    if modo_demo:
        df_mep_ccl = synthetic_fx_bundle(desde_brecha.strftime("%Y-%m-%d"), hasta_brecha.strftime("%Y-%m-%d"))[
            ["Date", "usd_mep", "usd_ccl"]]
        fuente_mep_ccl = "Ejemplo (sintético)"
    elif api_key.strip() and ALPHACAST_AVAILABLE:
        try:
            with st.spinner("Descargando MEP/CCL de Alphacast..."):
                raw_fx = download_dataset(api_key.strip(), int(fx_dataset_id))
            df_mep_ccl = normalize_fx_dataset(raw_fx)
            df_mep_ccl = df_mep_ccl[(df_mep_ccl["Date"] >= desde_brecha) & (df_mep_ccl["Date"] <= hasta_brecha)]
            fuente_mep_ccl = "Alphacast"
            if df_mep_ccl["usd_mep"].isna().all() or df_mep_ccl["usd_ccl"].isna().all():
                st.warning(f"No se pudo identificar automáticamente la columna de MEP y/o CCL en el "
                           f"dataset {fx_dataset_id}. Columnas encontradas: "
                           f"MEP → `{df_mep_ccl.attrs.get('col_mep')}`, CCL → `{df_mep_ccl.attrs.get('col_ccl')}`. "
                           f"Revisá el dataset en Alphacast o probá otro ID en 'Fuente de MEP/CCL'.")
        except Exception as e:
            st.error(f"No se pudo descargar MEP/CCL de Alphacast (dataset {fx_dataset_id}): {e}")
    else:
        st.info("Sin API key de Alphacast: no se puede traer MEP/CCL. Oficial/minorista/bandas sí se "
                "muestran igual, porque salen de la API pública del BCRA.")

    if df_bcra.empty and df_mep_ccl.empty:
        st.info("Sin datos disponibles todavía para graficar la brecha.")
    else:
        df_fx = df_bcra.copy() if not df_bcra.empty else pd.DataFrame({"Date": df_mep_ccl["Date"]})
        if not df_mep_ccl.empty:
            df_fx = df_fx.merge(df_mep_ccl, on="Date", how="outer")
        df_fx = df_fx.sort_values("Date").reset_index(drop=True)
        for c in ["usd_oficial", "usd_minorista", "usd_mep", "usd_ccl", "upper_band", "lower_band"]:
            if c not in df_fx.columns:
                df_fx[c] = np.nan
        df_fx = compute_brecha(df_fx)

        ultima = df_fx.dropna(subset=["usd_oficial"]).iloc[-1] if df_fx["usd_oficial"].notna().any() else None
        k1, k2, k3, k4, k5 = st.columns(5)
        if ultima is not None:
            k1.metric("Dólar Oficial", fmt_ars(ultima["usd_oficial"]))
            k2.metric("Dólar MEP", fmt_ars(ultima["usd_mep"]))
            k3.metric("Dólar CCL", fmt_ars(ultima["usd_ccl"]))
            k4.metric("Brecha (MEP vs. Oficial)", f"{ultima['Brecha']*100:.1f}%" if pd.notna(ultima["Brecha"]) else "—")
            if pd.notna(ultima.get("upper_band")) and pd.notna(ultima.get("lower_band")):
                pos_banda = (ultima["usd_oficial"] - ultima["lower_band"]) / (ultima["upper_band"] - ultima["lower_band"]) * 100
                k5.metric("Posición en la banda", f"{pos_banda:.0f}%", "0%=piso · 100%=techo")
            fecha_ultima = pd.Timestamp(df_fx.loc[df_fx["usd_oficial"].notna(), "Date"].iloc[-1])
            st.caption(f"Último dato: {fecha_ultima.strftime('%d/%m/%Y')} · Fuente oficial/minorista/bandas: BCRA · "
                       f"Fuente MEP/CCL: {fuente_mep_ccl or '—'}")

        fig = go.Figure()
        if df_fx["upper_band"].notna().any():
            fig.add_trace(go.Scatter(x=df_fx["Date"], y=df_fx["upper_band"], mode="lines", name="Banda superior",
                                      line=dict(width=1, color=COLORS["warn"], dash="dot")))
            fig.add_trace(go.Scatter(x=df_fx["Date"], y=df_fx["lower_band"], mode="lines", name="Banda inferior",
                                      line=dict(width=1, color=COLORS["ok"], dash="dot"),
                                      fill="tonexty", fillcolor="rgba(148,163,184,0.10)"))
        fig.add_trace(go.Scatter(x=df_fx["Date"], y=df_fx["usd_oficial"], mode="lines", name="Oficial",
                                  line=dict(width=2.2, color=COLORS["primary"])))
        if df_fx["usd_mep"].notna().any():
            fig.add_trace(go.Scatter(x=df_fx["Date"], y=df_fx["usd_mep"], mode="lines", name="MEP",
                                      line=dict(width=2.2, color=COLORS["obj2"])))
        if df_fx["usd_ccl"].notna().any():
            fig.add_trace(go.Scatter(x=df_fx["Date"], y=df_fx["usd_ccl"], mode="lines", name="CCL",
                                      line=dict(width=1.6, color=COLORS["accent"], dash="dash")))
        fig.update_layout(title="Comportamiento del dólar (Oficial, MEP, CCL y bandas de flotación)",
                           height=440, **PLOTLY_LAYOUT)
        style_axes(fig, "Fecha", "$ por USD")
        watermark(fig, hasta_brecha, "BCRA + Alphacast" if not modo_demo else "Ejemplo")
        st.plotly_chart(fig, use_container_width=True, key=f"fx_dolar_{desde_brecha}_{hasta_brecha}")

        if df_fx["Brecha"].notna().any():
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_fx["Date"], y=df_fx["Brecha"] * 100, mode="lines", fill="tozeroy",
                                       line=dict(width=1.6, color=COLORS["obj2"]),
                                       fillcolor="rgba(234,88,12,0.18)", name="Brecha"))
            fig2.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
            fig2.update_layout(title="Brecha cambiaria — (MEP − Oficial) / Oficial", height=340,
                               **PLOTLY_LAYOUT, showlegend=False)
            style_axes(fig2, "Fecha", "Brecha (%)")
            st.plotly_chart(fig2, use_container_width=True, key=f"fx_brecha_{desde_brecha}_{hasta_brecha}")

        st.subheader("Tabla completa")
        cols_fx = ["Date", "usd_mep", "usd_ccl", "usd_oficial", "usd_minorista", "upper_band", "lower_band", "Brecha"]
        tabla_fx = df_fx[[c for c in cols_fx if c in df_fx.columns]].copy()
        tabla_fx["Brecha_%"] = tabla_fx["Brecha"] * 100
        tabla_fx = tabla_fx.drop(columns=["Brecha"])
        st.dataframe(tabla_fx, use_container_width=True, height=320,
                     column_config={"Brecha_%": st.column_config.NumberColumn("Brecha (%)", format="%.2f%%")})
        csv_fx = tabla_fx.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar brecha cambiaria (CSV)", data=csv_fx,
                            file_name=f"brecha_cambiaria_{hasta_brecha.strftime('%Y%m%d')}.csv", mime="text/csv")

# ---------------------------------------------------------------------
# TAB 6 — Metodología
# ---------------------------------------------------------------------
with tab_metodo:
    st.markdown("""
    ### ℹ️ Metodología, fuentes y supuestos

    **Fuente de mercado:** Alphacast, dataset **41886** (ONs / Bonos / Soberanos) — el mismo dataset
    usado en el panel PRO de Renta Fija. Trae TIR (`irr`), Modified Duration, Paridad, segmento de
    mercado (Sovereign/Corporate) y estructura de cupón por ticker y fecha.

    **Simulación de compra (29/6/2026):** el monto asignado a cada instrumento (Peso_pct × monto total)
    se convierte a **nominales (VN)** dividiendo por la Paridad de esa fecha exacta. Para la caución, el
    "VN" es directamente el monto invertido (no hay concepto de paridad).

    **Simulación semanal:** a partir de la compra, los nominales quedan **fijos** (sin rebalanceo). Cada
    lunes se recalcula el valor de la cartera como VN × Paridad/100 (bonos) o devengamiento por TNA
    (caución). Los lunes con fecha ≤ hoy usan Paridad real de Alphacast; los posteriores se proyectan
    devengando la TIR vigente hoy de cada instrumento — es el piso esperado por carry, no una predicción
    de precios de mercado.

    **Benchmarks (pestaña Benchmarks & Ratios):** A3500 y CER salen de la API pública del BCRA (idVariables
    5 y 30). SHY (ETF de bonos del Tesoro de EE.UU. 1-3 años) sale de la API pública de Yahoo Finance, y se
    convierte a pesos componiendo su retorno en USD con la devaluación de A3500. El benchmark de Objetivo 1
    combina CER y SHY(en pesos) ponderados por el peso real de Tramo 1/2 y Tramo 3 dentro de Objetivo 1; el
    de Objetivo 2 es 100% A3500; el de la Cartera Total pondera ambos por el peso de cada Objetivo — misma
    estructura que la diapositiva "Medición de Desempeño" del IPS (TAMAR/CER para Tramo 1/2, ETF SHY para
    Tramo 3, A3500 para la cobertura cambiaria).

    **Ratios de riesgo-retorno:** Sharpe y Sortino usan como tasa libre de riesgo la TNA de la caución;
    Sortino solo penaliza la volatilidad a la baja. El Information Ratio compara el retorno activo (cartera
    − su benchmark compuesto) contra el tracking error. Los tres se anualizan multiplicando por √52 y se
    calculan únicamente sobre semanas ya REALIZADAS (nunca sobre las proyectadas) — con pocas semanas de
    historia real, son poco representativos y la app lo advierte explícitamente.

    **Brecha cambiaria:** Oficial (`Tipo de cambio mayorista de referencia`, idVariable 5), Minorista
    (idVariable 4) y las bandas de flotación (`Régimen de bandas cambiarias — Límite inferior/superior`,
    idVariables 1187/1188) salen de la **API pública del BCRA** (`api.bcra.gob.ar/estadisticas/v4.0`),
    que no requiere API key y está verificada contra los valores históricos reales. MEP y CCL no son
    series que el BCRA publique (surgen de operar bonos/acciones en distintas plazas): salen de
    Alphacast, dataset "Markets - Argentina - FX premiums - Daily" (ID configurable en la pestaña).
    La Brecha se calcula igual que en la planilla original: `(MEP − Oficial) / Oficial`.

    **Composición de la cartera:** los 8 instrumentos exactos del IPS del Grupo 8, reconciliando los dos
    gráficos de torta de la presentación (cartera consolidada $175M + detalle de Objetivo 1): **D31M7**
    (79,1%) y **D30S6** (6,9%) — Dollar-Linked, cobertura FX, 86,0% del total — y dentro de Objetivo 1
    (14,0%): **Cauciones** (1,4%), **S31L6** (1,4%), **TZXD6** (7,0%), **TLCQO** (2,1%), **LOC5O** (1,05%)
    y **AO27** (1,05%). El único dato manual es la TNA de la caución.

    **Cómo correr esto en Streamlit Cloud:**
    1. Subí `app.py` y `requirements.txt` a un repositorio de GitHub.
    2. En [share.streamlit.io](https://share.streamlit.io), creá una app apuntando a ese repo/`app.py`.
    3. Cargá tu Alphacast API Key como *secret* (`ALPHACAST_API_KEY`) en la configuración de la app,
       o pegala directamente en la barra lateral al abrir el panel.

    ⚠️ Esta herramienta es de análisis y monitoreo académico. No constituye recomendación de inversión.
    """)

# ---------------------------------------------------------------------
# Exportar informe completo (Excel)
# ---------------------------------------------------------------------
st.divider()
st.subheader("📥 Exportar informe completo (Excel)")
if not data_ok:
    st.info("Disponible cuando haya datos de mercado cargados (Alphacast o modo de ejemplo).")
else:
    st.caption("Descarga un .xlsx con el resumen de KPIs, la tabla de holdings de hoy, el detalle de la "
               "compra del 29/6 y la simulación semanal completa. Es una foto del momento, no un modelo con fórmulas.")
    try:
        excel_bytes = build_excel_snapshot(tabla, kpis_hoy, port_compra, semanal)
        st.download_button(
            "⬇️ Descargar informe Excel (.xlsx)",
            data=excel_bytes,
            file_name=f"cartera_grupo8_informe_{as_of_hoy.strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.warning(f"No se pudo generar el Excel: {e}")
