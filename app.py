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
  3) RIESGO & LÍMITES  → chequeo automático de los límites del IPS: banda de
                         duration, concentración máxima por emisor corporativo
                         y sublímite Dollar-Linked sobre el total.
  4) INSIGHTS          → performance vs. los benchmarks del propio IPS (TAMAR/
                         CER, ETF SHY, A3500) y propuestas de mejora atadas a
                         los 4 objetivos del mandato (preservación de capital,
                         liquidez, protección inflación, cobertura cambiaria).
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

# Benchmarks del propio IPS (diapositiva "Medición de Desempeño"), usados en
# la pestaña de Insights — no se inventan, son los que define el mandato.
BENCHMARK_TRAMO12_LO, BENCHMARK_TRAMO12_HI = 18.0, 22.0   # TAMAR / CER
BENCHMARK_TRAMO3_LO, BENCHMARK_TRAMO3_HI = 3.6, 3.9        # ETF SHY (USD)
BENCHMARK_A3500 = 20.0                                     # Devaluación esperada

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
# Motor de cartera: snapshot ponderado y KPIs (para "Cartera Hoy" y "Riesgo")
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


def tir_ponderada(snap: pd.DataFrame, tickers: list) -> float:
    """TIR ponderada de un subconjunto de tickers (para comparar contra los
    benchmarks del IPS por tramo)."""
    d = snap[snap["Ticker"].isin(tickers)].dropna(subset=["TIR"])
    peso = d["Peso_pct"].sum()
    return float((d["TIR"] * d["Peso_pct"]).sum() / peso) if peso > 0 else np.nan


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

    st.divider()
    st.subheader("📐 Límites del IPS (editables)")
    md_target_lo, md_target_hi = st.slider("Banda objetivo de Duration (años)", 0.0, 2.0, (0.50, 0.80), 0.01)
    limite_emisor_corp = st.number_input("Límite máx. por ON corporativa individual (%)", value=3.15, step=0.05)
    sublimite_dl = st.number_input("Sublímite Dollar-Linked sobre el total (%)", value=86.0, step=1.0)

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

# Compra del 29/6 y simulación semanal — se calculan UNA vez y se reusan en
# todas las pestañas (única fuente de verdad).
port_compra = compute_compra(port_df, df_norm, fecha_compra, monto_compra)
semanal = simulacion_semanal(port_compra, df_norm, fecha_compra, as_of_hoy, fecha_fin_sim)

fila_hoy = semanal[semanal["Estado"] == "Hoy"]
valor_actual_hoy = float(fila_hoy["Valor_Cartera"].iloc[0]) if not fila_hoy.empty else np.nan
resultado_pct_hoy = float(fila_hoy["Rendimiento_Acumulado_%"].iloc[0]) if not fila_hoy.empty else np.nan

tab_hoy, tab_sim, tab_riesgo, tab_insights, tab_metodo = st.tabs(
    ["🏠 Cartera Hoy", "📊 Simulación Semanal", "⚖️ Riesgo & Límites IPS",
     "💡 Insights & Propuestas", "ℹ️ Metodología"]
)

# ---------------------------------------------------------------------
# TAB 1 — Cartera Hoy
# ---------------------------------------------------------------------
with tab_hoy:
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
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        g = snap_hoy.groupby("Tramo")["Peso_pct"].sum().reset_index()
        fig = px.pie(g, names="Tramo", values="Peso_pct", hole=0.45)
        fig.update_traces(textinfo="label+percent", textposition="outside",
                           marker=dict(colors=px.colors.qualitative.Safe),
                           textfont=dict(color="white", size=12),
                           outsidetextfont=dict(color="white", size=12))
        fig.update_layout(title="Por Tramo", showlegend=False, height=380, **PLOTLY_LAYOUT)
        fig.update_layout(title_font=dict(color="white"), font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with p3:
        g = snap_hoy.groupby("Ticker")["Peso_pct"].sum().reset_index()
        fig = px.pie(g, names="Ticker", values="Peso_pct", hole=0.45)
        fig.update_traces(textinfo="label+percent", textposition="outside",
                           marker=dict(colors=px.colors.qualitative.Set2),
                           textfont=dict(color="white", size=12),
                           outsidetextfont=dict(color="white", size=12))
        fig.update_layout(title="Por Instrumento", showlegend=False, height=380, **PLOTLY_LAYOUT)
        fig.update_layout(title_font=dict(color="white"), font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(semanal.iloc[1:], x="Date", y="Rendimiento_Semanal_%",
                  color=semanal.iloc[1:]["Rendimiento_Semanal_%"] >= 0,
                  color_discrete_map={True: COLORS["ok"], False: COLORS["warn"]})
    fig2.add_vline(x=as_of_hoy, line_dash="dash", line_color=COLORS["accent"])
    fig2.update_layout(title="Rendimiento semanal de la cartera (%)", height=340, **PLOTLY_LAYOUT, showlegend=False)
    style_axes(fig2, "Fecha", "Rendimiento semanal (%)")
    st.plotly_chart(fig2, use_container_width=True)

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
# TAB 3 — Riesgo & Límites IPS
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
        st.dataframe(corp_show, use_container_width=True, height=160)
        incumplen = corp_show.loc[~corp_show["Cumple"], "Ticker"].tolist()
        if incumplen:
            st.error(f"🔴 Exceden el límite individual: **{', '.join(incumplen)}**. Rebalancear o "
                      f"diversificar en más emisores.")
        else:
            st.success("✅ Ninguna ON corporativa individual excede el límite.")
    else:
        st.info("No hay holdings marcados como Segmento = Corporate en la cartera actual.")

    st.divider()
    st.markdown(f"**3) Sublímite Dollar-Linked sobre el total de la cartera (referencia IPS: {sublimite_dl:.0f}%):**")
    peso_total_cartera = snap_hoy["Peso_pct"].sum()
    peso_dl = snap_hoy.loc[snap_hoy["Clase"] == "DL", "Peso_pct"].sum()
    if peso_total_cartera > 0:
        pct_dl = peso_dl / peso_total_cartera * 100
        st.metric("% Dollar-Linked sobre el total de la cartera", f"{pct_dl:.1f}%",
                  f"{pct_dl - sublimite_dl:+.1f} pp vs. referencia")
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
# TAB 4 — Insights & Propuestas de Mejora
# ---------------------------------------------------------------------
with tab_insights:
    st.markdown("### 💡 Performance vs. benchmarks del IPS")
    st.caption("Los benchmarks son los que define el propio IPS (diapositiva 'Medición de Desempeño'): "
               "no se inventan referencias externas.")

    tramo12_tickers = ["CAUCION", "S31L6", "TZXD6"]
    tramo3_tickers = ["TLCQO", "LOC5O", "AO27"]
    obj2_tickers = ["D31M7", "D30S6"]

    tir_t12 = tir_ponderada(snap_hoy, tramo12_tickers)
    tir_t3 = tir_ponderada(snap_hoy, tramo3_tickers)
    tir_obj2 = tir_ponderada(snap_hoy, obj2_tickers)

    b1, b2, b3 = st.columns(3)
    with b1:
        ok12 = pd.notna(tir_t12) and BENCHMARK_TRAMO12_LO <= tir_t12 <= BENCHMARK_TRAMO12_HI
        st.metric("Capital de Trabajo (Tramo 1/2) — TAMAR/CER 18-22%",
                  f"{tir_t12:.1f}%" if pd.notna(tir_t12) else "—",
                  "✅ dentro de banda" if ok12 else "⚠️ fuera de banda")
    with b2:
        ok3 = pd.notna(tir_t3) and BENCHMARK_TRAMO3_LO <= tir_t3 <= BENCHMARK_TRAMO3_HI
        st.metric("Capital de Trabajo (Tramo 3) — ETF SHY ≈3,6-3,9%",
                  f"{tir_t3:.1f}%" if pd.notna(tir_t3) else "—",
                  "✅ dentro de banda" if ok3 else "⚠️ fuera de banda")
    with b3:
        st.metric("Cobertura FX (Objetivo 2) — A3500 ≈20% (devaluación)",
                  f"{tir_obj2:.1f}% spread propio" if pd.notna(tir_obj2) else "—",
                  "El retorno esperado en pesos ≈ devaluación (A3500) + este spread")

    st.divider()
    st.markdown("### 🎯 Estado por objetivo del mandato")

    resultado_txt = f"{resultado_pct_hoy:+.2f}%" if pd.notna(resultado_pct_hoy) else "—"
    cap_ok = pd.notna(resultado_pct_hoy) and resultado_pct_hoy >= 0
    st.markdown(f"""
1. **Preservación de capital** (cero tolerancia a pérdida nominal): resultado acumulado desde la
   compra = **{resultado_txt}**. {"✅ En línea con el mandato." if cap_ok else "⚠️ Resultado nominal negativo — revisar si es un movimiento transitorio de precio (bonos que pagan a la par al vencimiento) o una señal de rebalanceo."}
2. **Liquidez** (disponibilidad calzada a pagos): Tramo Operativo (Caución + S31L6) pesa
   **{snap_hoy.loc[snap_hoy["Ticker"].isin(["CAUCION","S31L6"]), "Peso_pct"].sum():.1f}%** de la cartera.
3. **Protección inflación**: el tramo CER (TZXD6) pesa **{snap_hoy.loc[snap_hoy["Ticker"]=="TZXD6","Peso_pct"].sum():.1f}%**
   de la cartera — es la única cobertura directa contra IPC; el resto de Objetivo 1 (Cauciones/S31L6/Tramo 3) está
   en tasa nominal o dólares, no indexado a inflación.
4. **Cobertura cambiaria** (sin comprar dólar oficial): Dollar-Linked = **{pct_dl:.1f}%** de la cartera vs.
   préstamo de $150M / cartera de {fmt_ars(monto_compra)} ≈ {150e6/monto_compra*100:.1f}% necesario — {"✅ calce adecuado." if abs(pct_dl - 150e6/monto_compra*100) < 5 else "⚠️ revisar calce."}
    """)

    st.divider()
    st.markdown("### 📌 Propuestas de mejora")
    bullets = []

    if not cap_ok:
        bullets.append("**Preservación de capital:** el resultado nominal acumulado es negativo. Si el driver es "
                        "la Paridad de D31M7/D30S6 (que pagan a la par al vencimiento), no requiere acción — pero "
                        "conviene documentarlo para el comité, porque el IPS declara *cero tolerancia* a pérdida nominal.")
    if pd.notna(kpis_hoy["md_cartera"]) and not md_ok:
        bullets.append(f"**Duration fuera de banda** ({kpis_hoy['md_cartera']:.2f} años vs. "
                        f"{md_target_lo:.2f}-{md_target_hi:.2f}): rebalancear entre Tramo 2 (TZXD6) y Tramo 3 "
                        f"para volver a la banda objetivo.")
    if pd.notna(tir_t12) and not ok12:
        direccion = "por debajo" if tir_t12 < BENCHMARK_TRAMO12_LO else "por encima"
        bullets.append(f"**Tramo 1/2 rinde {direccion} del benchmark TAMAR/CER** ({tir_t12:.1f}% vs. 18-22%): "
                        f"evaluar rotar Cauciones/S31L6/TZXD6 hacia instrumentos con mejor tasa dentro del mismo "
                        f"horizonte (≤12 meses), sin resignar liquidez del Tramo Operativo.")
    if pd.notna(tir_t3) and not ok3:
        bullets.append(f"**Tramo 3 (Hard-Dollar) rinde distinto del benchmark SHY** ({tir_t3:.1f}% vs. 3,6-3,9%): "
                        f"con solo 3 papeles y 4,2% del total, hay margen para sumar 1-2 ONs investment-grade "
                        f"adicionales (sin superar el {limite_emisor_corp:.2f}% por emisor) y capturar mejor spread.")
    bullets.append("**Calce de vencimientos:** D31M7 (~mar-2027) y D30S6 (~sep-2026) cubren el horizonte del "
                    "préstamo, pero conviene escalonarlos exactamente contra los hitos de Mes 3/6/9 de desembolso "
                    "de obra civil y maquinaria, para minimizar el riesgo de tener que vender un DL antes de "
                    "vencimiento si el pago cae entre medio de dos vencimientos.")
    bullets.append("**Diversificación del Tramo 3:** hoy son solo 3 instrumentos con pesos individuales muy chicos "
                    "(1,05%-2,1%). Está lejos del límite de concentración (holgura), pero también lejos de "
                    "aprovechar todo el presupuesto de riesgo permitido — se puede sumar diversificación sin "
                    "tocar el perfil de riesgo agregado de la cartera.")

    for b in bullets:
        st.markdown(f"- {b}")

# ---------------------------------------------------------------------
# TAB 5 — Metodología
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

    **Benchmarks (pestaña Insights):** son los del propio IPS, diapositiva "Medición de Desempeño":
    TAMAR/CER 18%-22% para Tramo 1/2, ETF SHY ≈3,6-3,9% para Tramo 3, y A3500 ≈20% de devaluación
    esperada para la cobertura cambiaria de Objetivo 2.

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
