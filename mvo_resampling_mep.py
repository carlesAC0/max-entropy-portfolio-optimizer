"""
==============================================================================
Motor de Optimización de Carteras — MVO + Resampling + MEP (Rao)
==============================================================================
Implementación fiel al Documento Funcional v2.3 / v2.4
Fuentes: Bajo Traver (2025), Michaud (1998), Lo (2002), Rao (1982)

Pipeline:
  Fase 1  — Descarga de precios mensuales desde IBKR TWS API (ib_insync)
  Fase 2  — Limpieza y log-returns  [§1]
  Fase 3  — Estimación base μ, Σ, ρ  [§2]
  Fase 4  — MVO base + grid fijo K puntos  [§3]
  Fase 5  — Resampling block-bootstrap N simulaciones  [§4]
  Fase 6  — Solver MEP·RQE sobre frontera completa  [§5]
             - Calibración δS via Lo (2002)  [§6]
             - Restricción %RC opcional  [§5.5 v2.4]
             - Fallback por punto  [§5.4]
  Fase 7  — Agregación por mediana  [§8]
  Fase 8  — Outputs: pesos, IC p5-p95, RQE, DR, coste Sharpe, fallbacks  [§9]
  Fase 9  — Walk-Forward OOS  [§10.2]
  Fase 10 — Dashboard HTML interactivo  [§12.2]

Paralelización: concurrent.futures.ProcessPoolExecutor
  Las N simulaciones son independientes entre sí → se reparten entre
  todos los núcleos disponibles del i9 (n_workers = cpu_count).
  La lógica del modelo NO cambia: mismo bootstrap, mismo MEP, misma
  agregación por mediana. Solo cambia el orden de ejecución.

Dependencias:
  pip install ib_insync numpy scipy pandas matplotlib openpyxl
  TWS o IB Gateway abierto en local (puerto 7496 live / 7497 paper)

Autor: generado como herramienta de simulación para TFM MFIA
==============================================================================
"""

import sys
import time
import logging
import warnings
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN PRINCIPAL — edita aquí antes de ejecutar
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── IBKR conexión ──────────────────────────────────────────────────────
    "ibkr_host":   "127.0.0.1",
    "ibkr_port":   7496,          # 7496=TWS live | 7497=TWS paper | 4001=Gateway
    "ibkr_client": 10,

    # ── Universo de activos — NÚCLEO del resampling ─────────────────────────
    # NOTA METODOLÓGICA — Proxies de renta fija (§1.1):
    #   MTA  (Amundi Euro Gov 1-3Y)  → IBGS (iShares Euro Govt 1-3yr)
    #   MTD  (Amundi Euro Gov 7-10Y) → IBGM (iShares Euro Govt 7-10yr)
    # Mismo subyacente, distinto emisor (histórico desde ~2011).
    # Estimación con iShares; EJECUCIÓN real con los Amundi.
    #
    # ENFOQUE CORE-SATELLITE (decisión de diseño del gestor, no de los papers):
    # El resampling optimiza solo el NÚCLEO estructural. Se excluyen como
    # SATÉLITES tácticos (gestión discrecional aparte, ~5% conjunto):
    #   AIGA (agro/materiales) — commodity cíclica, volátil y táctica
    #   IEMA (emergentes)      — RV satélite táctica
    # El oro (IGLN) SÍ va al núcleo por su rol diversificador estructural.
    # MTE1 (Amundi 10-15Y) excluido: sin proxy iShares exacto, peso era 0%.
    "universe": {
        "XDWT":  (227264004, "IBIS2",  "EUR"),   # Tecnología
        "XDWS":  (227264011, "IBIS2",  "EUR"),   # Consumer Staples
        "XDWH":  (227263992, "IBIS2",  "EUR"),   # Health Care
        "XDW0":  (227263991, "IBIS2",  "EUR"),   # Energía
        "IGLN":  (86656182,  "LSEETF", "USD"),   # Oro (estructural)
        "EPRA":  (255731263, "SBF",    "EUR"),   # Real Estate
        "IBGS":  (54233539,  "LSEETF", "EUR"),   # proxy de MTA  (bono 1-3Y)
        "IBGM":  (68489986,  "LSEETF", "EUR"),   # proxy de MTD  (bono 7-10Y)
    },
    "proxy_map": {"IBGS": "MTA", "IBGM": "MTD"},   # proxy → activo real

    # ── Datos [§1] ─────────────────────────────────────────────────────────
    "freq":            "monthly",
    "window_months":   180,          # Máximo histórico para diagnóstico real
                                      # (la ventana común la limita el activo más joven)
    "min_obs":         60,
    "rf_annual":       0.035,         # usado solo si rf_mode="constant" o si falla la descarga
    "rf_mode":         "estr",        # "estr" = tipo libre de riesgo €STR vía ETF | "constant"
    "rf_ticker":       (46041702, "IBIS2", "EUR"),   # XEON (Xtrackers II EUR Overnight Rate, €STR)
    "base_currency":   "EUR",        # divisa base de la cartera: convierte series USD→EUR
    "prefer_total_return": True,     # usa ADJUSTED_LAST homogéneo; fallback TRADES homogéneo
    "project_median":  False,        # método base = mediana cruda de Michaud (sin proyectar).
                                      # Probado: proyectar al conjunto factible DEGRADA el OOS
                                      # (SR 0.60→0.53) por sobreajuste → se mantiene desactivado.
                                      # El validador sigue documentando la violación (transparencia).

    # ── MVO base [§3] ──────────────────────────────────────────────────────
    "K":               10,
    "w_min":           0.0,
    "w_max":           0.20,    # cap 20% por activo (Jagannathan & Ma, 2003)

    # ── Resampling [§4] ────────────────────────────────────────────────────
    "N_sim":           10000,         # run definitivo (aval de convergencia); estándar = 1000
    "block_size":      6,
    "random_seed":     42,
    "n_workers":       None,          # None = todos los núcleos del CPU (i9 = 24)

    # ── MEP·RQE [§5] ───────────────────────────────────────────────────────
    "z_alpha":         1.96,
    "sharpe_fallback": 0.5,
    "mep_multistart":  3,

    # ── Restricción %RC [§5.5 v2.4] ────────────────────────────────────────
    # k=2.0 → límite = k/N por activo (N variable según selección).
    # SIEMPRE activo: es parte del modelo (rejilla QEP), no una opción.
    "rc_k":            2.0,

    # ── Superficie QEP + frente de Pareto [Bajo Traver 2025, "Optimal deviation levels"] ──
    # Barrido de tolerancias (δS × cap RC) → superficie de QEPs → Pareto sobre
    # 3 objetivos (RQE, Sharpe, concentración=máxRC). 3.er eje = contribución al
    # riesgo (sustituye a la duración modificada del paper, sin sentido en multiactivo).
    "qep_n_ds":        12,            # nº de niveles de tolerancia de Sharpe (δS)
    "qep_ds_lo_frac":  0.05,          # δS mínimo = frac · δS_Lo ; δS máximo = δS_Lo (Lo 2002)
    "qep_n_rc":        10,            # nº de niveles de cap de contribución al riesgo
    "qep_rc_lo":       1.0,           # rc_k mínimo = riesgo igualado (1/N por activo)
    "qep_rc_hi":       4.0,           # rc_k máximo (incluye la esquina MVO, ~46% concentración)
    # Selección del punto final (MEP_optim) sobre el frente de Pareto:
    #   "max_rqe"  → MÁXIMA diversificación dentro de la banda de Sharpe (MEP_max,
    #               Bajo Traver 2025). Criterio único sin pesos = objetivo del modelo.
    #               Es el más diversificado y el que mejor generaliza OOS (Michaud).
    #   "ideal"    → compromise programming (punto más cercano al ideal).
    #   "lambda"   → media ponderada con qep_lambdas (preferencia explícita).
    #   "sharpe_rqe" → máx Sharpe con RQE ≥ 97% del máximo.
    "qep_select":      "max_rqe",
    "qep_lambdas":     (0.34, 0.33, 0.33),  # solo si qep_select="lambda"

    # ── Walk-Forward OOS [§10.2] ───────────────────────────────────────────
    "oos_train_months":  60,
    "oos_rebal_freq":     3,
    "oos_n_sim":         500,          # subido de 200: estabiliza selección de k OOS
    "oos_k_point":        None,
    "oos_cost_bps":       10,          # coste de transacción por rebalanceo (pb sobre turnover)
    "benchmark_ticker":   "SXR8",      # índice de referencia (S&P 500, iShares Core UCITS EUR/Xetra) para los gráficos OOS; si no está en los datos, se ignora

    # ── Outputs ────────────────────────────────────────────────────────────
    "output_excel":    "resultados_mep.xlsx",
    "output_chart":    "frontera_mep.png",
    "output_oos":      "oos_walkforward.xlsx",
    "output_html":     "dashboard_mep.html",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("MEP")

# ─────────────────────────────────────────────────────────────────────────────
# FASE 1 — DESCARGA IBKR [§1.1]
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices_ibkr(cfg: dict) -> pd.DataFrame:
    try:
        from ib_insync import IB, Stock, Forex, util
    except ImportError:
        raise ImportError("Instala ib_insync: pip install ib_insync")

    ib = IB()
    log.info(f"Conectando a IBKR en {cfg['ibkr_host']}:{cfg['ibkr_port']} "
             f"(clientId={cfg['ibkr_client']})")
    ib.connect(cfg["ibkr_host"], cfg["ibkr_port"], clientId=cfg["ibkr_client"])

    freq     = cfg["freq"]
    years    = max(1, round((cfg["window_months"] + 2) / 12))
    bar_size = {"monthly": "1 month", "weekly": "1 week", "daily": "1 day"}[freq]
    n_chunks = max(1, (years + 1) // 2)

    def _to_period(bars):
        """Barras IBKR → Serie de cierres normalizada a período (mensual)."""
        d = util.df(bars)[["date", "close"]].set_index("date")
        d.index = pd.to_datetime(d.index)
        d = d[~d.index.duplicated(keep="last")]
        if freq == "monthly":
            s = d["close"].copy()
            s.index = s.index.to_period("M")
            return s.groupby(level=0).last()
        return d["close"]

    def _fetch_total_return(contract):
        """
        Retorno total HOMOGÉNEO con ADJUSTED_LAST (incluye dividendos).
        IBKR NO admite ADJUSTED_LAST con barras multi-día ("1 month"/"1 week"),
        solo diarias/intradía → pedimos DIARIO y _to_period lo resamplea a fin
        de mes (último cierre ajustado de cada mes). ADJUSTED_LAST exige
        endDateTime vacío, por eso una sola petición sin troceo.
        Solo aplica a freq mensual; para otras frecuencias se usa TRADES.
        """
        if freq != "monthly":
            return None
        for intento in range(2):
            try:
                b = ib.reqHistoricalData(
                    contract, endDateTime="", durationStr=f"{years} Y",
                    barSizeSetting="1 day", whatToShow="ADJUSTED_LAST",
                    useRTH=True, formatDate=1)
                if b:
                    return b
            except Exception:
                ib.sleep(0.3)
        return None

    def _fetch_chunked_trades(contract):
        """
        Fallback HOMOGÉNEO: TRADES en toda la ventana (misma fuente en todos los
        bloques, sin mezclar con ADJUSTED_LAST). Trocea en bloques de 2 años.
        """
        all_bars, end_dt = [], ""
        for _ in range(n_chunks):
            got = None
            for intento in range(2):
                try:
                    b = ib.reqHistoricalData(
                        contract, endDateTime=end_dt, durationStr="2 Y",
                        barSizeSetting=bar_size, whatToShow="TRADES",
                        useRTH=True, formatDate=1)
                    if b:
                        got = b; break
                except Exception:
                    ib.sleep(0.3)
            if not got:
                break
            all_bars = got + all_bars
            end_dt = pd.to_datetime(got[0].date).strftime("%Y%m%d-%H:%M:%S")
        return all_bars

    prefer_tr = cfg.get("prefer_total_return", True)
    prices, src_used, currencies = {}, {}, {}
    for ticker, (con_id, exchange, currency) in cfg["universe"].items():
        contract = Stock(conId=con_id, exchange="SMART",
                         primaryExchange=exchange, currency=currency)
        bars, src = (None, None)
        if prefer_tr:
            bars = _fetch_total_return(contract)
            src  = "ADJUSTED_LAST" if bars else None
        if not bars:
            bars = _fetch_chunked_trades(contract)
            src  = "TRADES" if bars else None
        if bars:
            prices[ticker]     = _to_period(bars)
            src_used[ticker]   = src
            currencies[ticker] = currency
            log.info(f"  {ticker}: {len(prices[ticker])} per. [{src}, {currency}] "
                     f"({prices[ticker].index.min()} → {prices[ticker].index.max()})")
        else:
            log.error(f"  Sin datos para {ticker}. Se excluirá.")

    # ── Tipo libre de riesgo (€STR vía ETF, p.ej. XEON) ─────────────────────
    # Se descarga como columna 'EUR_RF'; NO entra a la optimización (se separa
    # en prepare_returns/run). Su retorno mensual es el rf_t variable.
    if cfg.get("rf_mode") == "estr" and cfg.get("rf_ticker"):
        rf_cid, rf_exch, rf_cur = cfg["rf_ticker"]
        rf_contract = Stock(conId=rf_cid, exchange="SMART",
                            primaryExchange=rf_exch, currency=rf_cur)
        rf_bars = _fetch_total_return(rf_contract) or _fetch_chunked_trades(rf_contract)
        if rf_bars:
            prices["EUR_RF"] = _to_period(rf_bars)
            log.info(f"  EUR_RF (€STR): {len(prices['EUR_RF'])} per. "
                     f"({prices['EUR_RF'].index.min()} → {prices['EUR_RF'].index.max()})")
        else:
            log.warning("  €STR no disponible; se usará rf constante.")

    if not prices:
        ib.disconnect()
        raise RuntimeError("No se pudieron descargar datos de ningún activo.")

    # Aviso si una serie quedó en TRADES y otra en ADJUSTED_LAST (heterogéneo)
    fuentes = set(src_used.values())
    if len(fuentes) > 1:
        log.warning(f"  Fuentes mixtas de precio entre activos: {src_used} "
                    f"(cada serie es homogénea internamente, pero comparas TR con no-TR)")

    df = pd.DataFrame(prices)

    # ── Conversión a divisa base (FX) ───────────────────────────────────────
    base = cfg.get("base_currency", "EUR")
    non_base = sorted({c for c in currencies.values() if c and c != base})
    for cur in non_base:
        fx = _fetch_fx_to_base(ib, base, cur, years, bar_size, freq, util, Forex)
        if fx is None:
            log.warning(f"  FX {base}/{cur} no disponible: {cur} se deja SIN convertir.")
            continue
        cols = [t for t in df.columns if currencies.get(t) == cur]
        df[cols] = df[cols].multiply(fx.reindex(df.index), axis=0)
        log.info(f"  FX aplicado: {cols} {cur}→{base}")

    ib.disconnect()

    if freq == "monthly":
        df.index = df.index.to_timestamp(how="end").normalize()
    return df


def _fetch_fx_to_base(ib, base, cur, years, bar_size, freq, util, Forex):
    """
    Serie de FX 'unidades de `base` por 1 de `cur`' (para multiplicar precios
    en `cur` y obtenerlos en `base`). Prueba el par directo y el inverso.
    """
    def _series(bars):
        d = util.df(bars)[["date", "close"]].set_index("date")
        d.index = pd.to_datetime(d.index)
        d = d[~d.index.duplicated(keep="last")]
        if freq == "monthly":
            s = d["close"].copy(); s.index = s.index.to_period("M")
            return s.groupby(level=0).last()
        return d["close"]

    # Forex(base+cur) cotiza 'cur por 1 base' → base por cur = 1/serie
    for pair, invert in ((base + cur, True), (cur + base, False)):
        try:
            b = ib.reqHistoricalData(Forex(pair), endDateTime="",
                    durationStr=f"{years} Y", barSizeSetting=bar_size,
                    whatToShow="MIDPOINT", useRTH=False, formatDate=1)
            if b:
                s = _series(b)
                return (1.0 / s) if invert else s
        except Exception:
            ib.sleep(0.3)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# FASE 2 — LIMPIEZA Y LOG-RETURNS [§1.2, §1.3]
# ─────────────────────────────────────────────────────────────────────────────

def prepare_returns(df_prices: pd.DataFrame, cfg: dict):
    log.info("Fase 2: Limpieza y cálculo de log-returns")
    excluded = []

    # ── DIAGNÓSTICO: ventana histórica por activo ──────────────────────────
    # Identifica qué activo recorta la ventana común (intersección §1.2)
    log.info("  ── Diagnóstico de ventana histórica por activo ──")
    diag = []
    for col in df_prices.columns:
        serie = df_prices[col].dropna()
        if len(serie) > 0:
            diag.append((col, serie.index.min(), serie.index.max(), len(serie)))
    # Ordenar por fecha de inicio (el más tardío es el que recorta)
    diag.sort(key=lambda x: x[1], reverse=True)
    log.info(f"  {'Activo':>8}  {'Inicio':>12}  {'Fin':>12}  {'N_meses':>8}")
    for tk, ini, fin, n in diag:
        flag = "  ← RECORTA LA VENTANA" if tk == diag[0][0] else ""
        log.info(f"  {tk:>8}  {str(ini.date()):>12}  {str(fin.date()):>12}  {n:>8}{flag}")
    if len(diag) > 1:
        ventana_potencial = diag[1][3]  # 2º activo más tardío
        log.info(f"  → Si excluyeras {diag[0][0]} (arranca {diag[0][1].date()}), "
                 f"la ventana común subiría hacia ~{ventana_potencial} meses")
    log.info("  ─────────────────────────────────────────────────")

    df_clean = df_prices.dropna(how="any")
    log.info(f"  Filas tras intersección de fechas: {len(df_clean)}")
    df_ret = np.log(df_clean / df_clean.shift(1)).dropna()
    log.info(f"  Observaciones de retornos disponibles: {len(df_ret)}")
    # KO limpio: la ventana común la fija el activo con MENOS histórico (tras la
    # intersección todas las columnas tienen la misma T). Si no llega al mínimo
    # se para AQUÍ con el culpable identificado — antes se excluían todos los
    # activos a la vez y el fallo aparecía aguas abajo con mensajes confusos.
    if len(df_ret) < cfg["min_obs"]:
        culpable = diag[0] if diag else None
        msg = f"KO: ventana común T={len(df_ret)} meses < min_obs={cfg['min_obs']}. "
        if culpable:
            msg += (f"El activo que recorta la ventana es {culpable[0]} "
                    f"(histórico desde {culpable[1].date()}). "
                    f"Quítalo de la selección o reduce min_obs.")
        raise ValueError(msg)
    log.info(f"  Activos válidos: {df_ret.shape[1]} — {list(df_ret.columns)}")
    log.info(f"  T final (retornos): {len(df_ret)}")
    return df_ret, excluded


def split_rf(df_prices: pd.DataFrame, cfg: dict, ann_factor: int):
    """
    Separa la columna 'EUR_RF' (€STR) de los activos.
    Devuelve (df_activos, rf_series_mensual | None, rf_annual_a_usar).
    rf_series es el log-retorno mensual del €STR (tipo libre de riesgo variable);
    rf_annual es su media anualizada (rf escalar para el in-sample, equivalente al
    variable porque el €STR es prácticamente suave mes a mes).
    """
    rf_col = "EUR_RF"
    if rf_col not in df_prices.columns:
        return df_prices, None, cfg.get("rf_annual", 0.0)
    rf_px     = df_prices[[rf_col]].dropna()
    df_assets = df_prices.drop(columns=[rf_col])
    rf_ret    = np.log(rf_px / rf_px.shift(1)).dropna()[rf_col]
    rf_ann    = float(rf_ret.mean()) * ann_factor
    return df_assets, rf_ret, rf_ann

# ─────────────────────────────────────────────────────────────────────────────
# FASE 3 — ESTIMACIÓN BASE [§2]
# ─────────────────────────────────────────────────────────────────────────────

def estimate_base_params(df_ret: pd.DataFrame, ann_factor: int) -> dict:
    log.info("Fase 3: Estimación base de parámetros")
    R = df_ret.values
    mu    = R.mean(axis=0) * ann_factor
    sigma = np.cov(R.T) * ann_factor
    rho   = np.corrcoef(R.T)
    log.info(f"  μ (anualizado): {np.round(mu, 4)}")
    log.info(f"  σ_p mínima posible: {np.round(np.sqrt(np.diag(sigma)), 4)}")
    return {"mu": mu, "sigma": sigma, "rho": rho, "R": R,
            "T": len(R), "N": R.shape[1], "ann": ann_factor,
            "assets": list(df_ret.columns)}

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES MVO Y MEP
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_stats(w, mu, sigma, rf):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ sigma @ w))
    sr  = (ret - rf) / vol if vol > 1e-9 else 0.0
    return ret, vol, sr

def mvo_solve(mu, sigma, ret_target, w_min, w_max):
    N  = len(mu)
    w0 = np.ones(N) / N
    cons = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: w @ mu - ret_target},
    ]
    res = minimize(
        lambda w: w @ sigma @ w, w0, method="SLSQP",
        bounds=[(w_min, w_max)] * N, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    if res.success and abs(res.x.sum() - 1.0) < 1e-4:
        return np.clip(res.x, w_min, w_max)
    return None

def mvo_max_sharpe(mu, sigma, rf, w_min, w_max):
    """Cartera tangente (máximo Sharpe) de la frontera MVO — referencia del QEP."""
    N  = len(mu)
    w0 = np.ones(N) / N
    def neg_sr(w):
        vol = np.sqrt(w @ sigma @ w)
        return -((w @ mu - rf) / vol) if vol > 1e-9 else 0.0
    res = minimize(neg_sr, w0, method="SLSQP",
                   bounds=[(w_min, w_max)] * N,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"ftol": 1e-10, "maxiter": 1000})
    if res.success and abs(res.x.sum() - 1.0) < 1e-4:
        w = np.clip(res.x, w_min, w_max); return w / w.sum()
    return None

def build_distance_matrix(rho):
    """
    Matriz de distancias de Bajo Traver (2025, JAM), Eqs. 5-6:
      1) Gower (1966):        d_ij  = √((1 − ρ_ij)/2)   ∈ [0,1], métrica válida.
      2) López de Prado:      d̃_ij = ‖D_i − D_j‖₂       "distancia de distancias"
         (distancia euclídea entre las columnas i,j de la matriz de Gower).
    Se devuelve D̃, que es la que entra en la RQE (Eq. 7).
    """
    D_gower = np.sqrt(np.clip((1.0 - rho) / 2.0, 0.0, None))          # Eq. 5
    diff    = D_gower[:, :, None] - D_gower[:, None, :]               # diff[n,i,j]=d_ni−d_nj
    D_tilde = np.sqrt(np.einsum("nij,nij->ij", diff, diff))           # Eq. 6
    return D_tilde

def rqe(w, D):
    # Entropía cuadrática de Rao — Bajo Traver (2025), Eq. 7:  H_D = ½ wᵀD̃w
    return float(0.5 * (w @ D @ w))

def risk_contrib_pct(w, sigma):
    port_vol = np.sqrt(w @ sigma @ w)
    if port_vol < 1e-9:
        return np.zeros(len(w))
    mcr = (sigma @ w) / port_vol
    return (w * mcr) / port_vol

def calibrate_delta_s(sr, T, z_alpha):
    return z_alpha * np.sqrt((1.0 + 0.5 * sr**2) / T)

def rc_k_eff(cfg):
    """rc_k efectivo del config. El cap %RC es parte del modelo: siempre activo."""
    return cfg.get("rc_k")

def fmt_rc(rc, dec=2):
    """Formatea un rc_k que puede ser None (cap desactivado)."""
    return "sin cap" if rc is None else f"{rc:.{dec}f}"

def validate_weights(w, mu, sigma, rf, sr_ref, delta_s, w_min, w_max, rc_k, tol=1e-3):
    """
    Comprueba que una cartera AGREGADA (mediana de las simulaciones) respeta
    las restricciones del problema MEP. La mediana componente-a-componente no
    preserva factibilidad automáticamente, por eso se valida a posteriori.
    Devuelve (ok: bool, viols: list[str]).
    """
    viols = []
    if abs(float(w.sum()) - 1.0) > tol:
        viols.append(f"suma={float(w.sum()):.4f}")
    if float(w.max()) > w_max + tol:
        viols.append(f"w_max ({float(w.max())*100:.1f}%>{w_max*100:.0f}%)")
    if float(w.min()) < w_min - tol:
        viols.append(f"w_min ({float(w.min())*100:.2f}%)")
    _, vol, _ = portfolio_stats(w, mu, sigma, rf)
    sr = (w @ mu - rf) / vol if vol > 1e-9 else 0.0
    if not np.isnan(sr_ref) and abs(sr - sr_ref) > delta_s + tol:
        viols.append(f"banda Sharpe (|{sr:.3f}-{sr_ref:.3f}|>{delta_s:.3f})")
    if rc_k is not None:
        prc = risk_contrib_pct(w, sigma)
        if float(prc.max()) > rc_k / len(w) + tol:
            viols.append(f"%RC ({float(prc.max())*100:.1f}%>{rc_k/len(w)*100:.0f}%)")
    return (len(viols) == 0, viols)

def project_to_feasible(w_med, mu, sigma, rf, sr_ref, delta_s, w_min, w_max, rc_k):
    """
    Cartera factible MÁS CERCANA (mínima distancia L2) a la mediana agregada,
    sujeta a las mismas restricciones del MEP: Σw=1, bounds, banda de Sharpe y %RC.

    Criterio NEUTRO respecto al Sharpe: minimiza ‖w − w_med‖², no optimiza Sharpe
    ni RQE. Solo devuelve la mediana al conjunto admisible cuando se ha salido,
    moviéndola lo mínimo. Devuelve None si el solver no converge (se conserva la
    mediana original y se reporta la violación).
    """
    N = len(w_med)
    def obj(w):  return float(np.sum((w - w_med) ** 2))
    def grad(w): return 2.0 * (w - w_med)
    def sr(w):
        v = np.sqrt(w @ sigma @ w)
        return (w @ mu - rf) / v if v > 1e-9 else 0.0

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if not np.isnan(sr_ref):
        cons.append({"type": "ineq", "fun": lambda w: delta_s - (sr_ref - sr(w))})
        cons.append({"type": "ineq", "fun": lambda w: delta_s - (sr(w) - sr_ref)})
    if rc_k is not None:
        lim = rc_k / N
        def c_rc(i):
            return lambda w: lim - risk_contrib_pct(w, sigma)[i]
        for i in range(N):
            cons.append({"type": "ineq", "fun": c_rc(i)})

    res = minimize(obj, w_med, jac=grad, method="SLSQP",
                   bounds=[(w_min, w_max)] * N, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 2000})
    if res.success and abs(res.x.sum() - 1.0) < 1e-3:
        w = np.clip(res.x, w_min, w_max); w /= w.sum()
        return w
    return None

def solve_mep(w_mvo, D, mu, sigma, sr_ref, delta_s, rf, w_min, w_max, rc_k, n_starts,
              rng=None, starts=None):
    N      = len(w_mvo)
    bounds = [(w_min, w_max)] * N
    if rng is None:
        rng = np.random.default_rng()

    def neg_rqe(w):      return -rqe(w, D)
    def grad_neg_rqe(w): return -(D @ w)          # ∇(½wᵀD̃w) = D̃w  (D̃ simétrica)
    def c_sum(w):        return w.sum() - 1.0
    def c_slo(w):
        _, vol, _ = portfolio_stats(w, mu, sigma, rf)
        if vol < 1e-9: return -delta_s
        return delta_s - (sr_ref - (w @ mu - rf) / vol)
    def c_shi(w):
        _, vol, _ = portfolio_stats(w, mu, sigma, rf)
        if vol < 1e-9: return -delta_s
        return delta_s - ((w @ mu - rf) / vol - sr_ref)

    cons = [{"type": "eq", "fun": c_sum},
            {"type": "ineq", "fun": c_slo},
            {"type": "ineq", "fun": c_shi}]

    if rc_k is not None:
        limit = rc_k / N
        def c_rc_factory(i):
            def c_rc_i(w):
                return limit - risk_contrib_pct(w, sigma)[i]
            return c_rc_i
        for i in range(N):
            cons.append({"type": "ineq", "fun": c_rc_factory(i)})

    best_w, best_val = None, -np.inf
    if starts is None:
        starts = [w_mvo.copy()]
        for _ in range(n_starts - 1):
            w_r = rng.dirichlet(np.ones(N))
            w_r = np.clip(w_r, w_min, w_max); w_r /= w_r.sum()
            starts.append(w_r)

    for w0 in starts:
        try:
            res = minimize(neg_rqe, w0, jac=grad_neg_rqe, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"ftol": 1e-10, "maxiter": 2000})
            if res.success and abs(res.x.sum() - 1.0) < 1e-3:
                wc = np.clip(res.x, w_min, w_max); wc /= wc.sum()
                rv = rqe(wc, D)
                if rv > best_val:
                    best_val = rv; best_w = wc
        except Exception:
            continue
    return best_w, (best_w is None)

# ─────────────────────────────────────────────────────────────────────────────
# QEP SURFACE + FRENTE DE PARETO — Bajo Traver (2025), "Optimal deviation levels"
# ─────────────────────────────────────────────────────────────────────────────

def _qep_cell_worker(args):
    """Resuelve UNA celda (δS, rc) de la superficie QEP (pickle-safe, paralelo)."""
    (ds, rc, w_mvo, D, mu, sigma, sr_ref, rf, w_min, w_max, n_starts, starts) = args
    w, failed = solve_mep(w_mvo, D, mu, sigma, sr_ref, ds, rf,
                          w_min, w_max, rc, n_starts, starts=starts)
    if failed or w is None:
        return None
    ret_p, vol_p, sr_p = portfolio_stats(w, mu, sigma, rf)
    return {"w": w, "rqe": rqe(w, D), "sr": sr_p, "ret": ret_p, "vol": vol_p,
            "maxrc": float(risk_contrib_pct(w, sigma).max()),
            "ds": float(ds), "rc": None if rc is None else float(rc)}

def build_qep_surface(w_mvo, sr_ref, ds_max, D, mu, sigma, rf, cfg, rng=None):
    """
    Superficie de carteras cuasi-eficientes (QEP): barrido de la rejilla de
    tolerancias (δS × cap de contribución al riesgo). Cada casilla resuelve una
    MEP → una QEP con sus 3 objetivos {RQE, Sharpe, concentración = máx RC}.
    Adaptación multiactivo: el 3.er eje del paper (duración modificada) se
    sustituye por la contribución al riesgo (Roncalli), que controla la
    concentración. `ds_max` = δS de Lo (2002); el barrido va de una fracción a δS.
    Devuelve la lista de QEPs factibles (la "superficie").
    """
    if rng is None:
        rng = np.random.default_rng(cfg.get("random_seed", 42))
    ds_grid = np.linspace(ds_max * cfg["qep_ds_lo_frac"], ds_max, cfg["qep_n_ds"])
    rc_grid = np.linspace(cfg["qep_rc_lo"], cfg["qep_rc_hi"], cfg["qep_n_rc"])

    # Los multistarts se pre-generan aquí con el MISMO rng secuencial (orden
    # ds×rc) que usaba el modo serie → el resultado es idéntico bit a bit,
    # se resuelva en serie o en paralelo.
    N = len(w_mvo)
    cells, args = [], []
    for ds in ds_grid:
        for rc in rc_grid:
            starts = [w_mvo.copy()]
            for _ in range(cfg["mep_multistart"] - 1):
                w_r = rng.dirichlet(np.ones(N))
                w_r = np.clip(w_r, cfg["w_min"], cfg["w_max"]); w_r /= w_r.sum()
                starts.append(w_r)
            cells.append((ds, rc))
            args.append((ds, rc, w_mvo, D, mu, sigma, sr_ref, rf,
                         cfg["w_min"], cfg["w_max"], cfg["mep_multistart"], starts))

    n_cpu = cfg.get("n_workers") or multiprocessing.cpu_count()
    parallel = cfg.get("qep_parallel", True) and n_cpu > 1 and len(cells) >= 16
    log.info(f"  Superficie QEP: {len(ds_grid)}×{len(rc_grid)} = {len(cells)} celdas "
             f"(w_max={cfg['w_max']*100:.0f}%, {n_cpu} núcleos)" if parallel else
             f"  Superficie QEP: {len(cells)} celdas (w_max={cfg['w_max']*100:.0f}%, serie)")
    t0 = time.time()
    surface = []
    if parallel:
        with ProcessPoolExecutor(max_workers=n_cpu) as ex:
            futs = [ex.submit(_qep_cell_worker, a) for a in args]
            for i, fut in enumerate(futs, 1):     # orden de envío = determinista
                p = fut.result()
                if p is not None:
                    surface.append(p)
                if i % 30 == 0:
                    log.info(f"    superficie: {i}/{len(cells)} celdas ({time.time()-t0:.0f}s)")
    else:
        for i, a in enumerate(args, 1):
            p = _qep_cell_worker(a)
            if p is not None:
                surface.append(p)
            if i % 20 == 0:
                log.info(f"    superficie: {i}/{len(cells)} celdas ({time.time()-t0:.0f}s)")
    log.info(f"    superficie lista: {len(surface)}/{len(cells)} QEPs factibles en {time.time()-t0:.0f}s")
    return surface

def pareto_front(surface):
    """
    Frente de Pareto sobre 3 objetivos: maximizar RQE, maximizar Sharpe,
    minimizar concentración (máx RC). Devuelve las QEPs no dominadas.
    """
    def dominates(b, a):   # ¿b domina a?
        ge = (b["rqe"] >= a["rqe"]) and (b["sr"] >= a["sr"]) and (b["maxrc"] <= a["maxrc"])
        gt = (b["rqe"] >  a["rqe"]) or  (b["sr"] >  a["sr"]) or  (b["maxrc"] <  a["maxrc"])
        return ge and gt
    return [a for a in surface if not any(dominates(b, a) for b in surface if b is not a)]

def select_mep_optim(front, method="ideal", lambdas=(0.34, 0.33, 0.33)):
    """
    Selección de una única cartera del frente de Pareto (MEP_optim). Los 3 objetivos
    (RQE, Sharpe, −concentración) se normalizan min-max a [0,1] ("más alto = mejor").
    Métodos:
      "max_rqe"    — MÁXIMA diversificación dentro de la banda de Sharpe de Lo (= el
                     "MEP_max" de Bajo Traver 2025). Criterio ÚNICO, sin pesos: es
                     literalmente el objetivo del modelo. Por defecto.
      "ideal"      — compromise programming (Zeleny): punto más cercano al ideal (1,1,1).
      "lambda"     — media ponderada con λ (preferencia explícita del inversor).
      "sharpe_rqe" — máximo Sharpe entre las QEP con RQE ≥ 97% del máximo.
    """
    if not front:
        return None
    def nrm(vals):
        lo, hi = min(vals), max(vals)
        return [(v - lo) / (hi - lo) if hi > lo else 1.0 for v in vals]
    H = nrm([p["rqe"]    for p in front])
    S = nrm([p["sr"]     for p in front])
    C = nrm([-p["maxrc"] for p in front])   # menos concentración = mejor
    if method == "max_rqe":
        # máxima diversificación; desempate por menor concentración
        best = max(range(len(front)), key=lambda i: (front[i]["rqe"], -front[i]["maxrc"]))
        score = front[best]["rqe"]
    elif method == "sharpe_rqe":
        rmax = max(p["rqe"] for p in front)
        cand = [i for i, p in enumerate(front) if p["rqe"] >= 0.97 * rmax]
        best = max(cand, key=lambda i: front[i]["sr"]) if cand else int(np.argmax(S))
        score = front[best]["sr"]
    elif method == "lambda":
        sc = [lambdas[0]*h + lambdas[1]*s + lambdas[2]*c for h, s, c in zip(H, S, C)]
        best = int(np.argmax(sc)); score = sc[best]
    else:  # "ideal" — distancia mínima al punto ideal (1,1,1)
        dist = [((1-h)**2 + (1-s)**2 + (1-c)**2) ** 0.5 for h, s, c in zip(H, S, C)]
        best = int(np.argmin(dist)); score = -dist[best]
    return {**front[best], "score": float(score)}

# ─────────────────────────────────────────────────────────────────────────────
# FASE 4 — MVO BASE + GRID FIJO [§3]
# ─────────────────────────────────────────────────────────────────────────────

def build_base_frontier(params: dict, cfg: dict) -> dict:
    log.info("Fase 4: MVO base — frontera eficiente")
    mu, sigma, rf, K = params["mu"], params["sigma"], cfg["rf_annual"], cfg["K"]
    feasible = []
    for r in np.linspace(mu.min() * 0.8, mu.max() * 1.05, 60):
        w = mvo_solve(mu, sigma, r, cfg["w_min"], cfg["w_max"])
        if w is not None:
            feasible.append(portfolio_stats(w, mu, sigma, rf)[0])
    if len(feasible) < 3:
        raise RuntimeError("Frontera MVO base infactible.")
    grid_rets = np.linspace(min(feasible), max(feasible), K)
    log.info(f"  Grid fijo: {K} puntos, retorno [{min(feasible):.4f}, {max(feasible):.4f}]")
    frontier = []
    for k, r_target in enumerate(grid_rets):
        w = mvo_solve(mu, sigma, r_target, cfg["w_min"], cfg["w_max"])
        if w is not None:
            ret, vol, sr = portfolio_stats(w, mu, sigma, rf)
            frontier.append({"k": k, "w": w, "ret": ret, "vol": vol, "sr": sr,
                              "ret_target": r_target})
        else:
            frontier.append(None)
    log.info(f"  Frontera base: {sum(f is not None for f in frontier)}/{K} puntos factibles")
    return {"frontier": frontier, "grid_rets": grid_rets, "K": K}

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP DE BLOQUE [§4.1]
# ─────────────────────────────────────────────────────────────────────────────

def block_bootstrap_sample(R, block_size, rng):
    """
    Stationary Block Bootstrap — Politis & Romano (1994), el método que emplea
    Bajo Traver (2025, JAM). A diferencia del bloque de longitud FIJA (Künsch),
    aquí la longitud de cada bloque es aleatoria ~ Geométrica(p = 1/block_size)
    —media = block_size— y el muestreo es CIRCULAR (envuelve al inicio de la
    serie). Esto hace la serie remuestreada estrictamente estacionaria y elimina
    el sesgo de los extremos propio del bloque fijo. `block_size` pasa a ser la
    longitud MEDIA de bloque (parámetro esperado del SBB).
    """
    T, _ = R.shape
    p   = 1.0 / max(block_size, 1)
    idx = []
    while len(idx) < T:
        start = int(rng.integers(0, T))
        L     = int(rng.geometric(p))          # longitud aleatoria, media 1/p
        for k in range(L):
            idx.append((start + k) % T)         # circular: envuelve al principio
            if len(idx) >= T:
                break
    return R[idx[:T]]

# ─────────────────────────────────────────────────────────────────────────────
# WORKER — una simulación completa (bootstrap + MVO + MEP)
# Se ejecuta en un proceso independiente → sin GIL → i9 al 100%
# ─────────────────────────────────────────────────────────────────────────────

def _sim_worker(args):
    """
    Procesa UNA simulación s completa.
    Recibe todo lo necesario por valor (pickle-safe).
    Retorna (s, mep_weights_k, mep_rqe_k, mep_sharpe_k,
               mvo_sharpe_k, delta_s_k, fallback_k)
    como arrays de longitud K.
    """
    (s, R, ann, rf, grid_rets, K, w_min, w_max,
     rc_k, z_alpha, block_size, mep_multistart, seed) = args

    rng     = np.random.default_rng(seed + s)   # seed distinta por sim
    N_a     = R.shape[1]

    # 1. Bootstrap
    R_s     = block_bootstrap_sample(R, block_size, rng)

    # 2. Re-estimación
    mu_s    = R_s.mean(axis=0) * ann
    sigma_s = np.cov(R_s.T) * ann
    rho_s   = np.corrcoef(R_s.T)
    D_s     = build_distance_matrix(rho_s)
    T_s     = len(R_s)

    # 3. MVO por punto del grid
    mvo_ws  = [mvo_solve(mu_s, sigma_s, r, w_min, w_max) for r in grid_rets]

    # 4. MEP por punto del grid
    out_w   = np.full((K, N_a), np.nan)
    out_rqe = np.full(K, np.nan)
    out_sr_mep = np.full(K, np.nan)
    out_sr_mvo = np.full(K, np.nan)
    out_ds  = np.full(K, np.nan)
    out_fb  = np.zeros(K, dtype=bool)

    for k, w_mvo_k in enumerate(mvo_ws):
        if w_mvo_k is None:
            out_fb[k] = True
            continue
        _, _, sr_ref = portfolio_stats(w_mvo_k, mu_s, sigma_s, rf)
        out_sr_mvo[k] = sr_ref
        delta_s = calibrate_delta_s(sr_ref, T_s, z_alpha)
        out_ds[k] = delta_s
        w_mep, is_fb = solve_mep(
            w_mvo_k, D_s, mu_s, sigma_s, sr_ref, delta_s,
            rf, w_min, w_max, rc_k, mep_multistart, rng=rng,
        )
        if is_fb or w_mep is None:
            out_fb[k] = True
            w_mep = w_mvo_k
        out_w[k]      = w_mep
        _, _, sr_mep  = portfolio_stats(w_mep, mu_s, sigma_s, rf)
        out_sr_mep[k] = sr_mep
        out_rqe[k]    = rqe(w_mep, D_s)

    return s, out_w, out_rqe, out_sr_mep, out_sr_mvo, out_ds, out_fb

# ─────────────────────────────────────────────────────────────────────────────
# FASE 5+6 — RESAMPLING PARALELO [§4+§5]
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(params: dict, base_frontier: dict, cfg: dict) -> dict:
    """
    Distribuye las N_sim simulaciones entre todos los núcleos del CPU.
    Cada worker ejecuta una simulación completa (bootstrap + MVO + MEP).
    La lógica del modelo es idéntica a la versión serie.
    """
    N_sim    = cfg["N_sim"]
    K        = base_frontier["K"]
    N_assets = params["N"]
    n_cpu    = cfg.get("n_workers") or multiprocessing.cpu_count()

    log.info(f"Fase 5+6: Resampling ({N_sim} sims) + MEP·RQE "
             f"(K={K} puntos) — {n_cpu} núcleos en paralelo")

    # Construir argumentos para cada worker
    worker_args = [
        (
            s,
            params["R"], params["ann"], cfg["rf_annual"],
            base_frontier["grid_rets"], K,
            cfg["w_min"], cfg["w_max"],
            rc_k_eff(cfg), cfg["z_alpha"],
            cfg["block_size"], cfg["mep_multistart"],
            cfg["random_seed"],
        )
        for s in range(N_sim)
    ]

    # Arrays de resultado
    mep_weights   = np.full((N_sim, K, N_assets), np.nan)
    mep_rqe       = np.full((N_sim, K), np.nan)
    mep_sharpe    = np.full((N_sim, K), np.nan)
    mvo_sharpe    = np.full((N_sim, K), np.nan)
    delta_s_log   = np.full((N_sim, K), np.nan)
    fallback_mask = np.zeros((N_sim, K), dtype=bool)

    done = 0
    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        futures = {executor.submit(_sim_worker, a): a[0] for a in worker_args}
        for fut in as_completed(futures):
            s, ow, orqe, osrm, osrmvo, ods, ofb = fut.result()
            mep_weights[s]   = ow
            mep_rqe[s]       = orqe
            mep_sharpe[s]    = osrm
            mvo_sharpe[s]    = osrmvo
            delta_s_log[s]   = ods
            fallback_mask[s] = ofb
            done += 1
            if done % 50 == 0:
                pct_fb = fallback_mask[:done].mean() * 100
                log.info(f"  Completadas {done}/{N_sim} sims — "
                         f"fallback acumulado: {pct_fb:.1f}%")

    log.info(f"  Fallback global: {fallback_mask.mean()*100:.1f}% de (sim×punto)")
    for k in range(K):
        log.info(f"    Punto k={k}: {fallback_mask[:,k].mean()*100:.1f}% fallback")

    return {
        "mep_weights":   mep_weights,
        "mep_rqe":       mep_rqe,
        "mep_sharpe":    mep_sharpe,
        "mvo_sharpe":    mvo_sharpe,
        "delta_s_log":   delta_s_log,
        "fallback_mask": fallback_mask,
    }

# ─────────────────────────────────────────────────────────────────────────────
# FASE 7 — AGREGACIÓN POR MEDIANA [§8]
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_frontier(sim_results: dict, params: dict,
                       base_frontier: dict, cfg: dict) -> list:
    log.info("Fase 7: Agregación por mediana")
    mw       = sim_results["mep_weights"]
    K        = base_frontier["K"]
    mu_base  = params["mu"]
    sig_base = params["sigma"]
    D_base   = build_distance_matrix(np.corrcoef(params["R"].T))
    rf       = cfg["rf_annual"]
    assets   = params["assets"]
    sigma_i  = np.sqrt(np.diag(sig_base))

    # RQE máximo teórico: maximizar RQE solo con Σw=1 y w∈[w_min,w_max],
    # SIN restricción de retorno ni Sharpe. Sirve para normalizar (0-1):
    # RQE_norm = RQE / RQE_max → "% de la diversificación máxima alcanzable".
    N_a = params["N"]
    def _neg_rqe(w): return -rqe(w, D_base)
    def _grad(w):    return -(D_base @ w)          # ∇(½wᵀD̃w) = D̃w
    from scipy.optimize import minimize as _min
    best_max = -np.inf
    for _s in range(5):
        w0 = (np.random.default_rng(_s).dirichlet(np.ones(N_a))
              if _s>0 else np.ones(N_a)/N_a)
        try:
            r = _min(_neg_rqe, w0, jac=_grad, method="SLSQP",
                     bounds=[(cfg["w_min"], cfg["w_max"])]*N_a,
                     constraints=[{"type":"eq","fun":lambda w: w.sum()-1.0}],
                     options={"ftol":1e-10,"maxiter":1000})
            if r.success:
                val = rqe(np.clip(r.x,cfg["w_min"],cfg["w_max"])/np.clip(r.x,cfg["w_min"],cfg["w_max"]).sum(), D_base)
                best_max = max(best_max, val)
        except Exception:
            pass
    rqe_max = best_max if best_max > 1e-9 else 1.0
    log.info(f"  RQE máximo teórico (para normalizar): {rqe_max:.4f}")
    final    = []

    for k in range(K):
        valid  = ~np.isnan(mw[:, k, 0])
        w_sims = mw[valid, k, :]
        if len(w_sims) == 0:
            final.append(None); continue

        w_med = np.median(w_sims, axis=0)
        # clip a [w_min, w_max] (no a [0,1]) para respetar el cap por activo
        w_med = np.clip(w_med, cfg["w_min"], cfg["w_max"]); w_med /= w_med.sum()
        w_p5  = np.percentile(w_sims, 5,  axis=0)
        w_p95 = np.percentile(w_sims, 95, axis=0)

        # Banda de Sharpe del punto y referencia MVO (necesarios para validar/proyectar)
        mvo_k  = base_frontier["frontier"][k]
        sr_mvo = mvo_k["sr"] if mvo_k is not None else np.nan
        delta_s= calibrate_delta_s(sr_mvo if not np.isnan(sr_mvo) else 1.0,
                                   params["T"], cfg["z_alpha"])

        # Validar la mediana; si es infactible, proyectarla al conjunto factible
        # (mínima distancia, criterio neutro respecto al Sharpe). [§5.5]
        feasible, viol = validate_weights(
            w_med, mu_base, sig_base, rf, sr_mvo, delta_s,
            cfg["w_min"], cfg["w_max"], rc_k_eff(cfg))
        projected = False
        if not feasible and cfg.get("project_median", False):
            w_proj = project_to_feasible(
                w_med, mu_base, sig_base, rf, sr_mvo, delta_s,
                cfg["w_min"], cfg["w_max"], rc_k_eff(cfg))
            if w_proj is not None:
                w_med = w_proj; projected = True
                feasible, viol = validate_weights(
                    w_med, mu_base, sig_base, rf, sr_mvo, delta_s,
                    cfg["w_min"], cfg["w_max"], rc_k_eff(cfg))
                log.info(f"  Punto k={k}: mediana proyectada al conjunto factible "
                         f"({'OK' if feasible else 'sigue violando: '+', '.join(viol)})")
        if not feasible and not projected:
            log.warning(f"  Punto k={k}: cartera mediana viola restricciones → {', '.join(viol)}")

        # Stats con la cartera final (mediana o proyectada)
        ret, vol, sr = portfolio_stats(w_med, mu_base, sig_base, rf)
        rqe_v  = rqe(w_med, D_base)
        dr     = float(w_med @ sigma_i) / vol if vol > 1e-9 else 0.0
        prc    = risk_contrib_pct(w_med, sig_base)
        fb_pct = float(sim_results["fallback_mask"][:, k].mean() * 100)
        cost_sr= sr_mvo - sr if not np.isnan(sr_mvo) else np.nan

        final.append({
            "k": k, "w": w_med, "w_p5": w_p5, "w_p95": w_p95,
            "w_std": w_sims.std(axis=0),
            "pct_aparicion": (w_sims > 1e-4).mean(axis=0),
            "ret": ret, "vol": vol, "sr": sr,
            "rqe": rqe_v, "rqe_norm": float(rqe_v/rqe_max), "dr": dr,
            "cost_sr": cost_sr,
            "rc_max": float(prc.max()), "rc_pct": prc,
            "fallback_pct": fb_pct, "delta_s": delta_s,
            "feasible": bool(feasible), "viol": viol, "projected": bool(projected),
            "n_valid_sims": int(valid.sum()), "assets": assets,
        })

    valid_pts = [p for p in final if p is not None]
    idx_ms    = int(np.argmax([p["sr"] for p in valid_pts]))
    for i, p in enumerate(valid_pts):
        p["max_sharpe"] = (i == idx_ms)

    log.info(f"  Frontera MEP final: {len(valid_pts)}/{K} puntos")
    log.info(f"  Máximo Sharpe: k={valid_pts[idx_ms]['k']} "
             f"SR={valid_pts[idx_ms]['sr']:.3f} "
             f"vol={valid_pts[idx_ms]['vol']*100:.1f}% "
             f"ret={valid_pts[idx_ms]['ret']*100:.1f}%")
    return final

# ─────────────────────────────────────────────────────────────────────────────
# MOTOR OPCIÓN A — Superficie QEP+Pareto (puntual) → tolerancia fija → resampling
# ─────────────────────────────────────────────────────────────────────────────

def _rqe_max(D, w_min, w_max, n_starts=6, seed=42):
    """RQE máximo teórico con solo Σw=1 y w∈[w_min,w_max] — para normalizar."""
    N = D.shape[0]; rng = np.random.default_rng(seed)
    def neg(w):  return -rqe(w, D)
    def grad(w): return -(D @ w)
    starts = [np.ones(N)/N] + [rng.dirichlet(np.ones(N)) for _ in range(n_starts-1)]
    best = -np.inf
    for w0 in starts:
        r = minimize(neg, w0, jac=grad, method="SLSQP", bounds=[(w_min, w_max)]*N,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum()-1.0}],
                     options={"ftol": 1e-10, "maxiter": 1000})
        if r.success:
            w = np.clip(r.x, w_min, w_max); w /= w.sum()
            best = max(best, rqe(w, D))
    return best if best > 1e-9 else 1.0

def _mep_fixed_worker(args):
    """UNA simulación a tolerancia FIJA (ds*, rc*): bootstrap → mini-MVO → MEP."""
    (s, R, ann, rf, ds_star, rc_star, w_min, w_max,
     block_size, mep_multistart, seed) = args
    rng     = np.random.default_rng(seed + s)
    R_s     = block_bootstrap_sample(R, block_size, rng)
    mu_s    = R_s.mean(axis=0) * ann
    sigma_s = np.cov(R_s.T) * ann
    D_s     = build_distance_matrix(np.corrcoef(R_s.T))
    w_ref   = mvo_max_sharpe(mu_s, sigma_s, rf, w_min, w_max)
    if w_ref is None:
        return s, None, True
    _, _, sr_ref = portfolio_stats(w_ref, mu_s, sigma_s, rf)
    w, failed = solve_mep(w_ref, D_s, mu_s, sigma_s, sr_ref, ds_star, rf,
                          w_min, w_max, rc_star, mep_multistart, rng=rng)
    if failed or w is None:
        return s, w_ref, True          # fallback a la tangente MVO de la sim
    return s, w, False

def run_mep_optim(params: dict, cfg: dict, rng=None) -> dict:
    """
    Motor Opción A (Bajo Traver 2025). Momento 1: sobre las estimaciones puntuales
    de la ventana, construye la superficie QEP → Pareto → MEP_optim, fijando la
    tolerancia ganadora (δS*, rc*). Momento 2: resampling (SBB) a esa tolerancia
    fija → mediana peso a peso + IC p5-p95. La receta se decide en CADA ejecución.
    """
    if rng is None:
        rng = np.random.default_rng(cfg["random_seed"])
    mu, sigma, rho = params["mu"], params["sigma"], params["rho"]
    R, T, N = params["R"], params["T"], params["N"]
    rf, wmn, wmx = cfg["rf_annual"], cfg["w_min"], cfg["w_max"]
    D = build_distance_matrix(rho)

    # ── Momento 1: decidir la receta (superficie QEP + Pareto) ──────────────
    w_ref = mvo_max_sharpe(mu, sigma, rf, wmn, wmx)
    if w_ref is None:
        raise RuntimeError("MVO tangente infactible.")
    _, _, sr_ref = portfolio_stats(w_ref, mu, sigma, rf)
    ds_max  = calibrate_delta_s(sr_ref, T, cfg["z_alpha"])
    surface = build_qep_surface(w_ref, sr_ref, ds_max, D, mu, sigma, rf, cfg, rng)
    front   = pareto_front(surface)
    opt     = select_mep_optim(front, cfg.get("qep_select", "ideal"),
                               cfg.get("qep_lambdas", (0.34, 0.33, 0.33)))
    if opt is None:
        raise RuntimeError("Superficie QEP vacía (sin QEP factibles).")
    ds_star, rc_star = opt["ds"], opt["rc"]
    log.info(f"  QEP: superficie={len(surface)} pareto={len(front)} → "
             f"tolerancia δS*={ds_star:.4f} rc*={fmt_rc(rc_star)}")

    # ── Momento 2: resampling a tolerancia fija → mediana + IC ──────────────
    N_sim = cfg["N_sim"]
    args = [(s, R, params["ann"], rf, ds_star, rc_star, wmn, wmx,
             cfg["block_size"], cfg["mep_multistart"], cfg["random_seed"])
            for s in range(N_sim)]
    W, n_fb = [], 0
    n_cpu = cfg.get("n_workers") or multiprocessing.cpu_count()
    if cfg.get("qep_parallel", True) and N_sim >= 50 and n_cpu > 1:
        with ProcessPoolExecutor(max_workers=n_cpu) as ex:
            for fut in as_completed([ex.submit(_mep_fixed_worker, a) for a in args]):
                _, w, fb = fut.result()
                if w is not None: W.append(w)
                n_fb += int(fb)
    else:
        for a in args:
            _, w, fb = _mep_fixed_worker(a)
            if w is not None: W.append(w)
            n_fb += int(fb)
    W = np.array(W)
    if len(W) == 0:
        raise RuntimeError("Resampling sin carteras factibles.")

    # ── Agregación robusta ──────────────────────────────────────────────────
    w_med = np.median(W, axis=0); w_med = w_med / w_med.sum()
    ic5   = np.percentile(W, 5,  axis=0)
    ic95  = np.percentile(W, 95, axis=0)
    ret_m, vol_m, sr_m = portfolio_stats(w_med, mu, sigma, rf)
    rqe_m   = rqe(w_med, D)
    rqe_max = _rqe_max(D, wmn, wmx, seed=cfg["random_seed"])
    dr      = float((np.sqrt(np.diag(sigma)) @ w_med) / vol_m) if vol_m > 1e-9 else np.nan
    return {
        "w": w_med, "ic5": ic5, "ic95": ic95, "assets": params["assets"],
        "ret": ret_m, "vol": vol_m, "sr": sr_m,
        "rqe": rqe_m, "rqe_norm": float(rqe_m / rqe_max), "rqe_max": rqe_max,
        "dr": dr, "maxrc": float(risk_contrib_pct(w_med, sigma).max()),
        "ds_star": ds_star, "rc_star": rc_star,
        "fallback_pct": 100.0 * n_fb / N_sim, "n_valid": len(W),
        "w_mvo": w_ref, "sr_mvo": sr_ref,
        "surface": surface, "pareto": front, "opt_point": opt,
    }

# ─────────────────────────────────────────────────────────────────────────────
# FASE 8 — OUTPUTS IS [§9]
# ─────────────────────────────────────────────────────────────────────────────

def print_report(res, params, excluded, cfg):
    assets = params["assets"]; w = res["w"]
    rc = risk_contrib_pct(w, params["sigma"])
    print("\n" + "="*72)
    print("  MOTOR MVO + RESAMPLING + MEP·RQE — Opción A (superficie QEP + Pareto)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')} | N_sim={cfg['N_sim']} | "
          f"SBB block={cfg['block_size']}m | superficie={len(res['surface'])} QEPs "
          f"({len(res['pareto'])} Pareto)")
    print("="*72)
    if excluded:
        print(f"\n  ACTIVOS EXCLUIDOS ({len(excluded)}):")
        for e in excluded:
            print(f"    {e['ticker']}: {e['motivo']}")
    print(f"\n  ACTIVOS: {assets}  T={params['T']}")
    print(f"\n  MEP_optim — receta elegida: δS*={res['ds_star']:.4f}  rc*={fmt_rc(res['rc_star'])}")
    print(f"    Ret={res['ret']*100:.2f}%  Vol={res['vol']*100:.2f}%  Sharpe={res['sr']:.3f}  "
          f"(MVO ref SR={res['sr_mvo']:.3f})")
    print(f"    RQE={res['rqe']:.4f} ({res['rqe_norm']*100:.1f}% del máx)  DR={res['dr']:.3f}  "
          f"máx RC={res['maxrc']*100:.1f}%  fallback={res['fallback_pct']:.1f}%")
    print(f"\n  COMPOSICIÓN (mediana de {res['n_valid']} sims a tolerancia fija):")
    print(f"  {'Activo':>8}  {'Peso%':>7}  {'IC5%':>7}  {'IC95%':>7}  {'%RC':>7}  {'Estab':>10}")
    print("  " + "-"*58)
    for i, a in enumerate(assets):
        p5 = res['ic5'][i]*100; p95 = res['ic95'][i]*100; rng = p95 - p5
        estab = "ROBUSTO" if rng < 10 else ("MODERADO" if rng < 20 else "INCIERTO")
        print(f"  {a:>8}  {w[i]*100:>7.1f}  {p5:>7.1f}  {p95:>7.1f}  {rc[i]*100:>7.1f}  {estab:>10}")


def export_excel(res, oos, params, excluded, cfg, filename):
    log.info(f"  Exportando Excel IS: {filename}")
    assets = params["assets"]; w = res["w"]
    rc = risk_contrib_pct(w, params["sigma"])
    pareto_ids = set(id(p) for p in res["pareto"])
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        pd.DataFrame([{
            "Ret%": round(res["ret"]*100,3), "Vol%": round(res["vol"]*100,3),
            "Sharpe": round(res["sr"],4), "Sharpe_MVO": round(res["sr_mvo"],4),
            "RQE": round(res["rqe"],5), "RQE_norm": round(res["rqe_norm"],4),
            "DR": round(res["dr"],4), "MaxRC%": round(res["maxrc"]*100,2),
            "delta_S_star": round(res["ds_star"],5),
            "rc_k_star": "sin cap" if res["rc_star"] is None else round(res["rc_star"],3),
            "Fallback%": round(res["fallback_pct"],2), "N_valid": res["n_valid"],
        }]).to_excel(writer, sheet_name="MEP_optim", index=False)

        pd.DataFrame([{"Activo": a, "Peso%": round(w[i]*100,2),
                       "IC_5%": round(res["ic5"][i]*100,2), "IC_95%": round(res["ic95"][i]*100,2),
                       "%RC": round(rc[i]*100,2)} for i, a in enumerate(assets)]
                     ).to_excel(writer, sheet_name="Pesos_IC_RC", index=False)

        pd.DataFrame([{"delta_S": round(p["ds"],5),
                       "rc_k": "sin cap" if p["rc"] is None else round(p["rc"],3),
                       "Sharpe": round(p["sr"],4), "RQE": round(p["rqe"],5),
                       "MaxRC%": round(p["maxrc"]*100,2), "Pareto": id(p) in pareto_ids}
                      for p in res["surface"]]
                     ).to_excel(writer, sheet_name="Superficie_QEP", index=False)

        if oos and oos.get("metrics"):
            m = oos["metrics"]; jk = oos["jk_test"]
            df_o = pd.DataFrame([{"Estrategia": m[k]["label"],
                                  "Ret%": round(m[k]["ret_ann"]*100,3),
                                  "Vol%": round(m[k]["vol_ann"]*100,3),
                                  "Sharpe": round(m[k]["sharpe"],4),
                                  "MDD%": round(m[k]["mdd"]*100,3)} for k in ("mep","mvo","ew")])
            df_o["JK_z"] = round(jk["z"],4); df_o["JK_p"] = round(jk["pval"],4)
            df_o["JK_sig"] = jk["significant"]
            df_o.to_excel(writer, sheet_name="OOS", index=False)

        pd.DataFrame({"Parametro": ["Fecha","N_activos","Activos","T","N_sim","block",
                                    "w_min","w_max","rf","z_alpha","Seed",
                                    "delta_S*","rc_k*","seleccion","Excluidos"],
                      "Valor": [datetime.now().strftime("%Y-%m-%d %H:%M"),
                                params["N"], str(assets), params["T"], cfg["N_sim"],
                                cfg["block_size"], cfg["w_min"], cfg["w_max"],
                                float(cfg["rf_annual"]), cfg["z_alpha"], cfg["random_seed"],
                                round(res["ds_star"],5),
                                "sin cap" if res["rc_star"] is None else round(res["rc_star"],3),
                                cfg.get("qep_select", "ideal"),
                                str([e["ticker"] for e in excluded])]}
                     ).to_excel(writer, sheet_name="Log_modelo", index=False)
    log.info("  Excel IS exportado.")


def plot_qep_surface(res, params, cfg, filename):
    """Superficie QEP + frente de Pareto (Figs. 8-9 Bajo Traver 2025)."""
    log.info(f"  Generando gráfico: {filename}")
    sigma = params["sigma"]
    surf = res["surface"]; pareto_ids = set(id(p) for p in res["pareto"])
    D = build_distance_matrix(params["rho"])
    opt = res["opt_point"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.4), facecolor="#0d1117")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e"); ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#e6edf3")
        ax.grid(True, color="#21262d", lw=0.5)

    # Panel A: superficie (Sharpe, RQE) coloreada por concentración
    sc = ax1.scatter([p["sr"] for p in surf], [p["rqe"] for p in surf],
                     c=[p["maxrc"]*100 for p in surf], cmap="plasma_r", s=48,
                     edgecolor=["#7cb342" if id(p) in pareto_ids else "#30363d" for p in surf],
                     linewidth=0.8)
    ax1.scatter([opt["sr"]], [opt["rqe"]], s=280, marker="*", c="#e8503a",
                edgecolor="white", zorder=5, label="MEP_optim")
    ax1.scatter([res["sr_mvo"]], [rqe(res["w_mvo"], D)], s=120, marker="D",
                c="#1f6feb", edgecolor="white", zorder=5, label="MVO ref")
    ax1.set_xlabel("Sharpe"); ax1.set_ylabel("RQE (diversificación)")
    ax1.set_title("Superficie QEP + frente de Pareto")
    fig.colorbar(sc, ax=ax1, label="concentración máx RC (%)").ax.yaxis.label.set_color("#c9d1d9")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)

    # Panel B: mapa de tolerancias (δS, rc_k) coloreado por RQE.
    # Sin cap %RC (ablation) el eje rc no existe: se pinta la curva δS → RQE
    # coloreada por concentración resultante.
    cap_off = any(p["rc"] is None for p in surf)
    if cap_off:
        sc2 = ax2.scatter([p["ds"] for p in surf], [p["rqe"] for p in surf],
                          c=[p["maxrc"]*100 for p in surf], cmap="plasma_r", s=90,
                          edgecolor="#30363d", linewidth=0.4)
        ax2.scatter([opt["ds"]], [opt["rqe"]], s=280, marker="*", c="#e8503a",
                    edgecolor="white", zorder=5, label="receta elegida")
        ax2.set_xlabel("δS (tolerancia de Sharpe)"); ax2.set_ylabel("RQE")
        ax2.set_title("Curva de tolerancias — SIN cap %RC (ablation)")
        fig.colorbar(sc2, ax=ax2, label="concentración máx RC (%)").ax.yaxis.label.set_color("#c9d1d9")
    else:
        sc2 = ax2.scatter([p["ds"] for p in surf], [p["rc"] for p in surf],
                          c=[p["rqe"] for p in surf], cmap="viridis", s=90, edgecolor="#30363d", linewidth=0.4)
        ax2.scatter([opt["ds"]], [opt["rc"]], s=280, marker="*", c="#e8503a",
                    edgecolor="white", zorder=5, label="receta elegida")
        ax2.set_xlabel("δS (tolerancia de Sharpe)"); ax2.set_ylabel("rc_k (cap contribución al riesgo)")
        ax2.set_title("Mapa de tolerancias (color = RQE)")
        fig.colorbar(sc2, ax=ax2, label="RQE").ax.yaxis.label.set_color("#c9d1d9")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)

    plt.suptitle("Motor MVO + Resampling + MEP·RQE — Opción A (superficie QEP)",
                 color="#e6edf3", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="#0d1117", edgecolor="none")
    plt.close()
    log.info("  Gráfico guardado.")

# ─────────────────────────────────────────────────────────────────────────────
# FASE 9 — WALK-FORWARD OOS [§10.2]
# ─────────────────────────────────────────────────────────────────────────────

def _oos_select_k(frontier_pts, cfg):
    valid = [p for p in frontier_pts if p is not None]
    if not valid: return None
    k_sel = cfg.get("oos_k_point")
    if k_sel is None:
        return max(valid, key=lambda p: p["sr"])
    matches = [p for p in valid if p["k"] == k_sel]
    return matches[0] if matches else max(valid, key=lambda p: p["sr"])


def jobson_korkie_test(r1, r2, rf_monthly, ann=12):
    """
    Test de igualdad de Sharpe (Jobson-Korkie 1981, corrección de Memmel 2003).

    Varianza asintótica de la diferencia de Sharpes (Sharpes a la frecuencia de
    los datos, p.ej. mensual):
        Var(SR1-SR2) = (1/T)[ 2(1-ρ) + ½(SR1² + SR2²) - SR1·SR2·ρ² ]
    El estadístico z es INVARIANTE a la anualización (numerador y se escalan
    igual), por lo que se calcula con Sharpes mensuales; los Sharpe que se
    reportan sí se anualizan (×√ann).
    """
    from scipy.stats import norm
    T  = len(r1)
    e1 = r1 - rf_monthly; e2 = r2 - rf_monthly
    m1, m2 = e1.mean(), e2.mean()
    s1, s2 = e1.std(ddof=1), e2.std(ddof=1)
    if s1 < 1e-12 or s2 < 1e-12:
        return {"z": 0.0, "pval": 1.0, "sr_mep": 0.0, "sr_bench": 0.0,
                "significant": False}
    s12 = np.cov(e1, e2, ddof=1)[0, 1]
    rho = s12 / (s1 * s2)
    sr1, sr2 = m1 / s1, m2 / s2                      # Sharpe mensual
    var = (1.0 / T) * (2 * (1 - rho)
                       + 0.5 * (sr1**2 + sr2**2)
                       - sr1 * sr2 * rho**2)         # Memmel (2003)
    se = np.sqrt(max(var, 1e-12))
    z  = (sr1 - sr2) / se if se > 1e-9 else 0.0
    pv = 2 * (1 - norm.cdf(abs(z)))
    a  = np.sqrt(ann)
    return {"z": float(z), "pval": float(pv), "sr_mep": float(sr1 * a),
            "sr_bench": float(sr2 * a), "significant": bool(pv < 0.05)}


def deflated_sharpe_stats(surface_sr_ann, sr_is_ann, r_is_excess, r_oos_excess,
                          ann, curve_max_n=1000):
    """
    Robustez estadística — Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".

      · PSR(0) sobre la track OOS  → ¿el Sharpe OOS es real (>0)? SIN deflación: el
        walk-forward ya neutraliza el sesgo de selección, así que el umbral es 0.
      · DSR sobre la selección IS  → deflacta el Sharpe IS del MEP por haber barrido
        N = nº de QEP candidatas (Eqs. 1-2 del paper): el umbral sube a SR0 =
        E[max{SR_n}], el "Sharpe alcanzable por azar" tras N trials.
      · DSR_OOS (secundario)       → misma deflación aplicada a la track OOS. Es un
        DOBLE CONTEO (el WFA ya limpió la selección); sólo como cota conservadora.
      · curve                      → SR0(N) y DSR(N) para el Exhibit 2 del paper
        (gráfico de robustez), con N de 2 a `curve_max_n`.

    Todo se calcula de la corrida — NADA hardcodeado. N sale de len(surface), la
    dispersión de la superficie, y los momentos 3.º/4.º de las series reales. Los
    Sharpe que entran son ANUALIZADOS; internamente se pasan a por-periodo (÷√ann).
    PSR/DSR son probabilidades: bueno si > 0.95 (equivale a p = 1−valor < 0.05).
    """
    from scipy.stats import norm, skew as _skew, kurtosis as _kurt
    GAMMA = 0.5772156649015329          # constante de Euler-Mascheroni
    a = float(np.sqrt(ann))

    def _psr(sr_m, sr0_m, T, sk, ku):
        """PSR(SR*) con corrección de no-normalidad (Lo 2002); None si no es válido."""
        if T is None or T < 2:
            return None
        den = 1.0 - sk * sr_m + (ku - 1.0) / 4.0 * sr_m ** 2
        if den <= 0:
            return None
        return float(norm.cdf((sr_m - sr0_m) * np.sqrt(T - 1) / np.sqrt(den)))

    def _sr0_m(N, sd_m):
        """Umbral deflactado SR0 = E[max{SR_n}] por periodo (Eq. 1)."""
        z1 = norm.ppf(1.0 - 1.0 / N)
        z2 = norm.ppf(1.0 - 1.0 / (N * np.e))
        return sd_m * ((1 - GAMMA) * z1 + GAMMA * z2)

    def _moments(x):
        """(T, skew, kurtosis no-exceso) de una serie; (None,0,3) si insuficiente."""
        if x is None:
            return None, 0.0, 3.0
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 3 or x.std(ddof=1) < 1e-12:
            return (len(x) if len(x) else None), 0.0, 3.0
        return len(x), float(_skew(x, bias=False)), float(_kurt(x, fisher=False, bias=False))

    # ── PSR(0) sobre la track OOS ─────────────────────────────────────────────
    psr_oos = sr_oos_ann = None
    T_oos, sk_oos, ku_oos = _moments(r_oos_excess)
    if T_oos and T_oos >= 3:
        r = np.asarray(r_oos_excess, float)
        s = r.std(ddof=1)
        if s > 1e-12:
            sr_oos_m = r.mean() / s
            sr_oos_ann = sr_oos_m * a
            psr_oos = _psr(sr_oos_m, 0.0, T_oos, sk_oos, ku_oos)

    # ── DSR sobre la selección IS ─────────────────────────────────────────────
    srg = np.asarray(surface_sr_ann, float)
    srg = srg[np.isfinite(srg)]
    N = int(len(srg))
    sr_is_m = float(sr_is_ann) / a
    dsr_is = sr0_ann = sd_grid_ann = None
    T_is, sk_is, ku_is = _moments(r_is_excess)
    if N >= 2:
        sd_grid_ann = float(srg.std(ddof=1))      # Sharpes de la rejilla son anuales
        sd_grid_m = sd_grid_ann / a               # → dispersión por periodo
        sr0_m = _sr0_m(N, sd_grid_m)
        sr0_ann = sr0_m * a
        dsr_is = _psr(sr_is_m, sr0_m, T_is, sk_is, ku_is)

    # ── DSR_OOS (secundario, doble conteo) ────────────────────────────────────
    dsr_oos = None
    if sr0_ann is not None and sr_oos_ann is not None and T_oos:
        dsr_oos = _psr(sr_oos_ann / a, sr0_ann / a, T_oos, sk_oos, ku_oos)

    # ── Curva Exhibit 2: SR0(N) y DSR(N) ──────────────────────────────────────
    curve = None
    if N >= 2 and sd_grid_ann is not None and T_is:
        sd_grid_m = sd_grid_ann / a
        cap = int(max(curve_max_n, N * 4))
        pts = sorted(set([2, 3, 5, 8, 13, 21, 34, 55, N, 89, 144, 233, 377, 610, 987]
                         + list(range(50, cap + 1, 50)) + [cap]))
        cn, csr0, cdsr = [], [], []
        for n in [p for p in pts if 2 <= p <= cap]:
            s0m = _sr0_m(n, sd_grid_m)
            d = _psr(sr_is_m, s0m, T_is, sk_is, ku_is)
            cn.append(int(n))
            csr0.append(round(float(s0m * a), 4))
            cdsr.append(None if d is None else round(float(d), 4))
        curve = {"N": cn, "sr0": csr0, "dsr": cdsr, "n_real": N}

    _r = lambda v, k=4: None if v is None else round(float(v), k)
    return {
        "psr_oos": _r(psr_oos), "dsr_is": _r(dsr_is), "dsr_oos": _r(dsr_oos),
        "N": N, "threshold": 0.95,
        "sr_is_ann": _r(sr_is_ann), "sr_oos_ann": _r(sr_oos_ann),
        "sr0_ann": _r(sr0_ann), "sd_grid_ann": _r(sd_grid_ann),
        "T_is": T_is, "T_oos": T_oos,
        "skew_is": _r(sk_is, 3), "kurt_is": _r(ku_is, 3),
        "skew_oos": _r(sk_oos, 3), "kurt_oos": _r(ku_oos, 3),
        "curve": curve,
    }


def _hold_block(w_tgt, w_prev, R_block, tc):
    """
    Simula MANTENER la cartera objetivo w_tgt durante un bloque de rebalanceo,
    dejando que los pesos DERIVEN con los retornos de mercado (buy-and-hold intra-bloque).
      · turnover = Σ_i |w_tgt_i − w_prev_i|  (compras + ventas) contra los pesos DERIVADOS
        del bloque anterior (w_prev), no contra los objetivo previos.
      · coste de transacción tc·turnover aplicado en el primer mes (mes del rebalanceo).
      · R_block: matriz (F × N) de LOG-retornos mensuales.
    Devuelve (retornos_log_por_mes, pesos_derivados_al_final_del_bloque, turnover).
    """
    turnover = float(np.abs(w_tgt - w_prev).sum())      # volumen negociado = compras+ventas
    w = np.asarray(w_tgt, dtype=float).copy()
    rets = []
    for k in range(len(R_block)):
        rs = np.expm1(R_block[k])                        # log-retorno → retorno simple por activo
        port = float(w @ rs)                             # retorno simple de la cartera ese mes
        if k == 0:
            port = (1.0 + port) * (1.0 - tc * turnover) - 1.0   # descuenta el coste del rebalanceo
        rets.append(np.log1p(port))
        grown = w * (1.0 + rs)                           # deriva de pesos por el mercado
        s = grown.sum()
        if s > 1e-12:
            w = grown / s
    return np.array(rets), w, turnover

def run_walk_forward(df_ret: pd.DataFrame, cfg: dict, rf_series=None, bench_ret=None) -> dict:
    log.info("Fase 9: Walk-Forward OOS [§10.2]")
    R_full = df_ret.values; dates = df_ret.index
    T_full = len(R_full); L = cfg["oos_train_months"]; F = cfg["oos_rebal_freq"]
    ann    = {"monthly":12,"weekly":52,"daily":252}[cfg["freq"]]
    rf_m   = cfg["rf_annual"] / ann; N_a = R_full.shape[1]
    # rf VARIABLE: serie mensual del €STR alineada a las fechas; si no hay, constante.
    if rf_series is not None:
        rf_aligned  = rf_series.reindex(dates).astype(float)
        rf_arr_full = rf_aligned.fillna(rf_aligned.mean()).values
    else:
        rf_arr_full = np.full(T_full, rf_m)
    steps  = list(range(L, T_full - F + 1, F))
    if not steps:
        log.warning(f"  OOS insuficiente: T={T_full}, L={L}, F={F}")
        return {}
    log.info(f"  Ventana train={L}m, rebal={F}m → {len(steps)} pasos OOS")
    cfg_oos = {**cfg, "N_sim": cfg["oos_n_sim"]}
    records = []
    # Se parte de CASH (pesos previos = 0): el primer rebalanceo compra toda la cartera
    # (turnover=1) para las TRES estrategias por igual, como en una implementación real.
    w_prev_mep = np.zeros(N_a); w_prev_mvo = np.zeros(N_a); w_prev_ew = np.zeros(N_a)
    w_ew_tgt   = np.ones(N_a)/N_a

    for si, t in enumerate(steps):
        R_train = R_full[t-L:t]; R_test = R_full[t:t+F]; date_t = dates[t]
        params_t = {"mu": R_train.mean(axis=0)*ann,
                    "sigma": np.cov(R_train.T)*ann,
                    "rho": np.corrcoef(R_train.T),
                    "R": R_train, "T": L, "N": N_a, "ann": ann,
                    "assets": list(df_ret.columns)}
        # rf de ESTA ventana (media del €STR en el periodo de entrenamiento)
        rf_train_ann = float(rf_arr_full[t-L:t].mean()) * ann
        cfg_t = {**cfg_oos, "rf_annual": rf_train_ann}
        try:
            # Motor Opción A: superficie QEP+Pareto (decide receta en ESTA ventana)
            # → resampling a tolerancia fija → mediana. Todo con datos ≤ t (sin look-ahead).
            res = run_mep_optim(params_t, cfg_t)
        except Exception as e:
            log.warning(f"  Paso t={t}: falló — {e}"); continue
        w_mep_t = res["w"]
        w_mvo_t = res["w_mvo"] if res["w_mvo"] is not None else w_ew_tgt
        # Mantenimiento del bloque con DERIVA de pesos por mercado y coste de transacción
        # (compras+ventas) en las TRES estrategias. El turnover se mide contra los pesos ya
        # DERIVADOS del bloque anterior; la EW se rebalancea a 1/N cada periodo (también rota).
        tc = cfg.get("oos_cost_bps", 0) / 1e4
        r_mep_per, w_prev_mep, to_mep = _hold_block(w_mep_t,  w_prev_mep, R_test, tc)
        r_mvo_per, w_prev_mvo, to_mvo = _hold_block(w_mvo_t,  w_prev_mvo, R_test, tc)
        r_ew_per,  w_prev_ew,  to_ew  = _hold_block(w_ew_tgt, w_prev_ew,  R_test, tc)
        records.append({
            "date": date_t, "t": t, "step": si,
            "w_mep": w_mep_t.copy(), "w_mvo": w_mvo_t.copy(),
            "ret_mep": float(r_mep_per.sum()),
            "ret_mvo": float(r_mvo_per.sum()),
            "ret_ew":  float(r_ew_per.sum()),
            "r_mep_per": r_mep_per,
            "r_mvo_per": r_mvo_per,
            "r_ew_per":  r_ew_per,
            "rf_per":    rf_arr_full[t:t+F],
            "turnover_mep": to_mep/2, "turnover_mvo": to_mvo/2, "turnover_ew": to_ew/2,
            "cost_mep": tc*to_mep, "cost_mvo": tc*to_mvo, "cost_ew": tc*to_ew,
            "rqe_is": float(res["rqe"]),
            "sr_is_mep": float(res["sr"]),
            "fb_pct": float(res["fallback_pct"]),
        })
        if (si+1) % 5 == 0:
            log.info(f"  OOS paso {si+1}/{len(steps)} ({date_t.date()})")

    if not records:
        log.error("  OOS: sin resultados."); return {}

    all_r_mep = np.concatenate([r["r_mep_per"] for r in records])
    all_r_mvo = np.concatenate([r["r_mvo_per"] for r in records])
    all_r_ew  = np.concatenate([r["r_ew_per"]  for r in records])
    all_rf    = np.concatenate([r["rf_per"]    for r in records])
    T_oos = len(all_r_mep)
    rf_oos_ann = float(all_rf.mean()) * ann
    log.info(f"  rf OOS medio (€STR): {rf_oos_ann*100:.2f}% anual")

    def oos_metrics(r_arr, label):
        cum  = np.expm1(np.cumsum(r_arr))
        ra   = (1+cum[-1])**(ann/T_oos)-1
        va   = r_arr.std(ddof=1)*np.sqrt(ann)
        ex   = r_arr - all_rf                       # exceso sobre rf variable (€STR)
        se   = ex.std(ddof=1)
        sr   = ex.mean()/se*np.sqrt(ann) if se > 1e-9 else 0.0
        nav  = np.exp(np.cumsum(r_arr))
        rm   = np.maximum.accumulate(nav)
        mdd  = float(((nav-rm)/rm).min())
        return {"label":label,"ret_ann":ra,"vol_ann":va,"sharpe":sr,"mdd":mdd,
                "T":T_oos,"nav":nav}

    m_mep = oos_metrics(all_r_mep,"MEP")
    m_mvo = oos_metrics(all_r_mvo,"MVO")
    m_ew  = oos_metrics(all_r_ew, "EW")
    jk    = jobson_korkie_test(all_r_mep, all_r_mvo, all_rf)

    log.info(f"\n  ── OOS RESULTADOS ──")
    for m in [m_mep, m_mvo, m_ew]:
        log.info(f"  {m['label']:5}: ret={m['ret_ann']*100:.2f}% "
                 f"vol={m['vol_ann']*100:.2f}% SR={m['sharpe']:.3f} "
                 f"MDD={m['mdd']*100:.2f}%")
    log.info(f"  Jobson-Korkie: z={jk['z']:.3f} p={jk['pval']:.4f} "
             f"{'✓ Sig.' if jk['significant'] else '✗ No sig.'}")
    # Benchmark (buy-and-hold de un índice, p.ej. S&P 500) sobre los mismos meses OOS
    all_r_bench = None
    if bench_ret is not None:
        bv = pd.Series(bench_ret).reindex(dates).values
        rb = np.concatenate([bv[r["t"]:r["t"]+len(r["r_mep_per"])] for r in records])
        if not np.any(np.isnan(rb)):
            all_r_bench = rb
    # Correlación entre las series de retorno OOS de las estrategias
    # (+ benchmark de mercado, si hay datos: mide la exposición a mercado de cada cartera)
    corr_labels = ["MEP", "MVO", "EW"]
    corr_series = [all_r_mep, all_r_mvo, all_r_ew]
    if all_r_bench is not None:
        corr_labels.append(cfg.get("benchmark_ticker") or "BENCH")
        corr_series.append(all_r_bench)
    corr = np.corrcoef(corr_series)
    log.info(f"  Corr OOS  MEP-MVO={corr[0,1]:.3f}  MEP-EW={corr[0,2]:.3f}  MVO-EW={corr[1,2]:.3f}")
    if all_r_bench is not None:
        log.info(f"  Corr OOS vs {corr_labels[3]}  MEP={corr[0,3]:.3f}  MVO={corr[1,3]:.3f}  EW={corr[2,3]:.3f}")
    # Fechas MENSUALES del OOS (una por cada retorno realizado), para el eje de los gráficos
    dates_m = []
    for r in records:
        tt = r["t"]
        dates_m += [str(dates[tt + j].date()) for j in range(len(r["r_mep_per"]))]
    out = {"records":records,"metrics":{"mep":m_mep,"mvo":m_mvo,"ew":m_ew},
           "jk_test":jk,"T_oos":T_oos,"dates_m":dates_m,
           "all_r_mep":all_r_mep,"all_r_mvo":all_r_mvo,"all_r_ew":all_r_ew,
           "all_rf":all_rf,
           "corr":{"labels":corr_labels,"matrix":corr.tolist()},
           "assets":list(df_ret.columns)}
    if all_r_bench is not None:
        out["all_r_bench"] = all_r_bench
        out["metrics"]["bench"] = oos_metrics(all_r_bench, "BENCH")
    return out


def export_oos_excel(oos, cfg, filename):
    if not oos: return
    log.info(f"  Exportando OOS Excel: {filename}")
    records = oos["records"]; assets = oos["assets"]
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        m = oos["metrics"]; jk = oos["jk_test"]
        rows = []
        for key, met in m.items():
            rows.append({"Estrategia":met["label"],
                         "Ret_anual%":round(met["ret_ann"]*100,3),
                         "Vol_anual%":round(met["vol_ann"]*100,3),
                         "Sharpe_OOS":round(met["sharpe"],4),
                         "MDD%":round(met["mdd"]*100,3),"T":met["T"]})
        df_s = pd.DataFrame(rows)
        df_s["JK_z"]   = round(jk["z"],4)
        df_s["JK_pval"]= round(jk["pval"],4)
        df_s["JK_sig"] = jk["significant"]
        df_s.to_excel(writer, sheet_name="Resumen_OOS", index=False)

        pd.DataFrame({
            "Fecha":  [r["date"] for r in records],
            "r_MEP":  [r["ret_mep"] for r in records],
            "r_MVO":  [r["ret_mvo"] for r in records],
            "r_EW":   [r["ret_ew"]  for r in records],
            "TO_MEP": [r["turnover_mep"] for r in records],
            "RQE_IS": [r["rqe_is"] for r in records],
            "SR_IS":  [r["sr_is_mep"] for r in records],
            "FB%":    [r["fb_pct"] for r in records],
        }).to_excel(writer, sheet_name="Retornos_OOS", index=False)

        w_rows = []
        for r in records:
            row = {"Fecha": r["date"]}
            for i, a in enumerate(assets):
                row[f"MEP_{a}"] = round(r["w_mep"][i]*100,2)
                row[f"MVO_{a}"] = round(r["w_mvo"][i]*100,2)
            w_rows.append(row)
        pd.DataFrame(w_rows).to_excel(writer, sheet_name="Pesos_OOS", index=False)
    log.info("  OOS Excel exportado.")

# ─────────────────────────────────────────────────────────────────────────────
# FASE 10 — DASHBOARD HTML [§12.2]
# ─────────────────────────────────────────────────────────────────────────────

def _generate_html_dashboard_legacy(final_frontier, oos, params, cfg, filename):  # DEPRECADA (K-frontera; sustituida por generate_static_dashboard, Opción A)
    import json
    log.info(f"  Generando dashboard HTML: {filename}")
    assets = params["assets"]
    valid  = [p for p in final_frontier if p is not None]
    pt_ms  = next((p for p in valid if p.get("max_sharpe")), valid[-1] if valid else None)

    frontier_data = []
    for p in valid:
        frontier_data.append({
            "k":int(p["k"]),"vol":round(float(p["vol"])*100,3),"ret":round(float(p["ret"])*100,3),
            "sr":round(float(p["sr"]),4),"rqe":round(float(p["rqe"]),5),
            "rqe_norm":round(float(p.get("rqe_norm",0)),4),"dr":round(float(p["dr"]),4),
            "cost_sr":round(float(p["cost_sr"]),4),"fb":round(float(p["fallback_pct"]),1),
            "ds":round(float(p["delta_s"]),4),
            "weights":{a:round(float(p["w"][i])*100,2) for i,a in enumerate(assets)},
            "ic5":{a:round(float(p["w_p5"][i])*100,2) for i,a in enumerate(assets)},
            "ic95":{a:round(float(p["w_p95"][i])*100,2) for i,a in enumerate(assets)},
            "rc":{a:round(float(p["rc_pct"][i])*100,2) for i,a in enumerate(assets)},
            "max_sr":bool(p.get("max_sharpe",False)),
        })

    oos_data = {}
    if oos:
        m=oos["metrics"]; jk=oos["jk_test"]
        nav_mep=(np.exp(np.cumsum(oos["all_r_mep"]))*100).tolist()
        nav_mvo=(np.exp(np.cumsum(oos["all_r_mvo"]))*100).tolist()
        nav_ew =(np.exp(np.cumsum(oos["all_r_ew" ]))*100).tolist()
        oos_data={
            "dates":[str(r["date"].date()) for r in oos["records"]],
            "nav_mep":[round(float(v),3) for v in nav_mep],
            "nav_mvo":[round(float(v),3) for v in nav_mvo],
            "nav_ew": [round(float(v),3) for v in nav_ew],
            "metrics":{
                "mep":{"ret":round(float(m["mep"]["ret_ann"])*100,2),"vol":round(float(m["mep"]["vol_ann"])*100,2),"sr":round(float(m["mep"]["sharpe"]),3),"mdd":round(float(m["mep"]["mdd"])*100,2)},
                "mvo":{"ret":round(float(m["mvo"]["ret_ann"])*100,2),"vol":round(float(m["mvo"]["vol_ann"])*100,2),"sr":round(float(m["mvo"]["sharpe"]),3),"mdd":round(float(m["mvo"]["mdd"])*100,2)},
                "ew": {"ret":round(float(m["ew"]["ret_ann"])*100,2),"vol":round(float(m["ew"]["vol_ann"])*100,2),"sr":round(float(m["ew"]["sharpe"]),3),"mdd":round(float(m["ew"]["mdd"])*100,2)},
            },
            "jk":{"z":round(float(jk["z"]),3),"pval":round(float(jk["pval"]),4),
                  "sig":bool(jk["significant"]),"sr_mep":round(float(jk["sr_mep"]),3),
                  "sr_bench":round(float(jk["sr_bench"]),3)},
            "T_oos":int(oos["T_oos"]),
        }

    fd_json  = json.dumps(frontier_data)
    oos_json = json.dumps(oos_data)
    assets_j = json.dumps(assets)
    rc_str   = f"k={cfg.get('rc_k')}" if rc_k_eff(cfg) else "sin cap"
    n_cpu    = cfg.get("n_workers") or multiprocessing.cpu_count()
    run_ts   = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Motor MEP·RQE — Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'IBM Plex Mono',ui-monospace,monospace;background:#0d1117;color:#e6edf3;min-height:100vh}}
header{{background:#161b22;border-bottom:2px solid #e8503a;padding:16px 28px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}}
.logo{{font-size:18px;font-weight:700;letter-spacing:1px}}.logo span{{color:#e8503a}}
.meta{{font-size:11px;color:#8b949e}}
.tabs{{display:flex;border-bottom:1px solid #30363d;padding:0 28px;background:#161b22}}
.tab{{padding:10px 20px;font-size:12px;cursor:pointer;border:none;background:transparent;color:#8b949e;font-family:inherit;border-bottom:2px solid transparent;transition:all .2s;letter-spacing:.5px}}
.tab.active{{color:#e6edf3;border-bottom-color:#e8503a}}.tab:hover{{color:#e6edf3}}
.view{{display:none;padding:24px 28px}}.view.active{{display:block}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}}
.kpi{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 16px}}
.kpi-label{{font-size:10px;color:#8b949e;letter-spacing:1px;margin-bottom:4px}}
.kpi-val{{font-size:22px;font-weight:600}}.kpi-sub{{font-size:11px;color:#6e7681;margin-top:2px}}
.pos{{color:#7cb342}}.neg{{color:#e8503a}}.neu{{color:#d98b2b}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px}}
.card-title{{font-size:11px;color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{font-size:10px;color:#8b949e;padding:6px 8px;border-bottom:1px solid #30363d;text-align:right;white-space:nowrap}}
th:first-child{{text-align:left}}
td{{padding:5px 8px;text-align:right;border-bottom:1px solid #21262d;color:#e6edf3}}
td:first-child{{text-align:left;color:#8b949e}}
.pt-selector{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px}}
.pt-btn{{padding:4px 10px;font-size:11px;border:1px solid #30363d;border-radius:4px;cursor:pointer;background:transparent;color:#8b949e;font-family:inherit;transition:all .2s}}
.pt-btn.active{{background:#e8503a;color:#fff;border-color:#e8503a}}
.oos-kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}}
.oos-card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px}}
.oos-label{{font-size:10px;color:#8b949e;letter-spacing:1px;margin-bottom:6px}}
.oos-row{{display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px}}
.jk-box{{background:#10151c;border:1px solid #21262d;border-radius:6px;padding:12px 16px;font-size:12px;line-height:1.8;margin-top:12px}}
.jk-box strong{{color:#d98b2b}}
.comp-summary{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}}
.cs-card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 14px}}
.cs-label{{font-size:10px;color:#8b949e;letter-spacing:1px;margin-bottom:4px}}
.cs-val{{font-size:18px;font-weight:600}}
.afila{{display:flex;align-items:center;gap:14px;padding:11px 4px;border-bottom:1px solid #21262d}}
.afila:last-child{{border-bottom:none}}
.adot{{width:10px;height:34px;border-radius:3px;flex-shrink:0}}
.aname{{width:70px;flex-shrink:0}}
.atkr{{font-size:13px;font-weight:600;color:#e6edf3}}
.aclass{{font-size:9px;color:#6e7681;letter-spacing:.5px}}
.abar-wrap{{flex:1;min-width:0}}
.abar-track{{position:relative;height:22px;background:#0d1117;border-radius:4px;overflow:hidden}}
.abar-fill{{position:absolute;top:0;left:0;height:100%;border-radius:4px;opacity:0.88;transition:width .4s}}
.abar-ic{{position:absolute;top:0;height:100%;background:rgba(139,148,158,0.18);border-left:1px dashed #6e7681;border-right:1px dashed #6e7681}}
.aweight{{width:64px;text-align:right;flex-shrink:0}}
.aw-big{{font-size:17px;font-weight:700;color:#e6edf3}}
.aw-ic{{font-size:9px;color:#6e7681}}
.arc{{width:80px;text-align:right;flex-shrink:0;font-size:12px}}
.arc-val{{font-weight:600}}
.arc-lbl{{font-size:9px;color:#6e7681}}
.aestab{{width:92px;text-align:right;flex-shrink:0}}
.semaforo{{display:inline-flex;align-items:center;gap:5px;font-size:10px;font-weight:600;padding:3px 9px;border-radius:12px}}
.sem-r{{background:rgba(124,179,66,0.15);color:#7cb342}}
.sem-m{{background:rgba(217,139,43,0.15);color:#d98b2b}}
.sem-i{{background:rgba(232,80,58,0.15);color:#e8503a}}
.sem-dot{{width:6px;height:6px;border-radius:50%}}
</style>
</head>
<body>
<header>
  <div><div class="logo">MVO + RESAMPLING + <span>MEP·RQE</span></div>
  <div class="meta">Funcional v2.3/v2.4 · Bajo Traver (2025) · Michaud (1998) · Lo (2002) · %RC {rc_str} · {n_cpu} núcleos</div></div>
  <div class="meta" style="text-align:right">N_sim={cfg['N_sim']} · K={cfg['K']} · block={cfg['block_size']}m<br>{run_ts}</div>
</header>
<div class="tabs">
  <button class="tab active" onclick="showTab('is',this)">Frontera IS</button>
  <button class="tab" onclick="showTab('comp',this)">Composición</button>
  <button class="tab" onclick="showTab('oos',this)">Walk-Forward OOS</button>
  <button class="tab" onclick="showTab('stats',this)">Estadísticos</button>
</div>
<div id="v-is" class="view active">
  <div class="kpis" id="kpis-is"></div>
  <div class="card"><div class="card-title">Frontera MEP robusta vs MVO base</div>
    <div style="position:relative;height:320px"><canvas id="cFrontera"></canvas></div></div>
  <div class="card"><div class="card-title">Evolución del RQE (diversificación) por punto de frontera</div>
    <div style="position:relative;height:240px"><canvas id="cRQE"></canvas></div></div>
  <div class="card"><div class="card-title">Tabla de frontera</div>
    <table><thead><tr><th>Punto</th><th>Vol%</th><th>Ret%</th><th>Sharpe</th><th>RQE</th><th>DR</th><th>ΔSharpe</th><th>FB%</th><th>δS</th></tr></thead>
    <tbody id="tb-frontera"></tbody></table></div>
</div>
<div id="v-comp" class="view">
  <div class="pt-selector" id="pt-sel"></div>
  <div class="comp-summary" id="comp-summary"></div>
  <div class="card"><div class="card-title">Composición de la cartera — ficha por activo</div>
    <div id="comp-cards"></div>
  </div>
  <div class="card"><div class="card-title">Distribución del riesgo (%RC)</div>
    <div style="position:relative;height:260px"><canvas id="cRC"></canvas></div></div>
</div>
<div id="v-oos" class="view"><div id="oos-content"></div></div>
<div id="v-stats" class="view">
  <div class="card"><div class="card-title">Métricas por punto</div>
    <div style="overflow-x:auto"><table><thead><tr>
      <th>k</th><th>Vol%</th><th>Ret%</th><th>Sharpe</th><th>RQE</th><th>DR</th><th>ΔSR</th><th>FB%</th><th>δS</th>
    </tr></thead><tbody id="tb-stats"></tbody></table></div></div>
  <div class="card"><div class="card-title">Parámetros del modelo</div>
    <table id="tb-params"></table></div>
</div>
<script>
const FD={fd_json};const OOS={oos_json};const ASSETS={assets_j};
const COLORS=['#e8503a','#7cb342','#1f6feb','#d98b2b','#a371f7','#3ab0a0','#f78166','#79c0ff','#56d364','#e3b341','#ff7b72','#bc8cff'];
let selPt=FD.findIndex(p=>p.max_sr);if(selPt<0)selPt=0;
let cComp=null,cRC=null;
const fmt=(v,d=2)=>v==null?'—':v.toFixed(d);
const fmtP=v=>(v>=0?'+':'')+fmt(v)+'%';

function showTab(n,btn){{
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('v-'+n).classList.add('active');
  btn.classList.add('active');
  if(n==='comp')renderComp();
  if(n==='oos')renderOOS();
  if(n==='stats')renderStats();
}}

function renderIS(){{
  const pt=FD[selPt]||FD[0];
  document.getElementById('kpis-is').innerHTML=[
    ['RETORNO IS',fmt(pt.ret)+'%','pos','Máx. Sharpe'],
    ['VOLATILIDAD',fmt(pt.vol)+'%','neu','Máx. Sharpe'],
    ['SHARPE IS',fmt(pt.sr,3),'pos','δS='+fmt(pt.ds,4)],
    ['RQE',fmt(pt.rqe,4),'',(pt.rqe_norm!=null?(pt.rqe_norm*100).toFixed(0)+'% del máx · ':'')+'DR='+fmt(pt.dr,3)],
  ].map(([l,v,c,s])=>`<div class="kpi"><div class="kpi-label">${{l}}</div><div class="kpi-val ${{c}}">${{v}}</div><div class="kpi-sub">${{s}}</div></div>`).join('');
  new Chart(document.getElementById('cFrontera'),{{
    data:{{datasets:[
      {{type:'line',label:'MEP',data:FD.map(p=>{{return{{x:p.vol,y:p.ret,sr:p.sr}}}}),borderColor:'#e8503a',borderWidth:2.5,pointRadius:5,pointBackgroundColor:FD.map(p=>p.max_sr?'#7cb342':'#e8503a'),fill:false,tension:0.3,showLine:true}},
      {{type:'scatter',label:'Máx.SR',data:[{{x:pt.vol,y:pt.ret}}],backgroundColor:'#7cb342',pointRadius:11,pointStyle:'star'}},
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,parsing:false,
      plugins:{{legend:{{labels:{{color:'#8b949e'}}}},tooltip:{{callbacks:{{label:d=>d.raw.sr!==undefined?`vol=${{fmt(d.raw.x)}}% ret=${{fmt(d.raw.y)}}% SR=${{fmt(d.raw.sr,3)}}`:`Máx.SR`}}}}}},
      scales:{{
        x:{{type:'linear',position:'bottom',
            title:{{display:true,text:'Volatilidad (%)',color:'#8b949e'}},
            ticks:{{color:'#8b949e',callback:v=>fmt(v)+'%'}},grid:{{color:'#21262d'}},
            min:Math.floor(Math.min(...FD.map(p=>p.vol))*10)/10-0.2,
            max:Math.ceil(Math.max(...FD.map(p=>p.vol))*10)/10+0.2}},
        y:{{type:'linear',
            title:{{display:true,text:'Retorno (%)',color:'#8b949e'}},
            ticks:{{color:'#8b949e',callback:v=>fmt(v)+'%'}},grid:{{color:'#21262d'}},
            min:Math.floor(Math.min(...FD.map(p=>p.ret)))-0.5,
            max:Math.ceil(Math.max(...FD.map(p=>p.ret)))+0.5}}
      }}
    }}
  }});
  document.getElementById('tb-frontera').innerHTML=FD.map(p=>`<tr>
    <td>${{p.max_sr?'★ ':''}}<b>k=${{p.k}}</b></td><td>${{fmt(p.vol)}}</td><td>${{fmt(p.ret)}}</td>
    <td class="${{p.sr>1?'pos':'neu'}}">${{fmt(p.sr,3)}}</td><td>${{fmt(p.rqe,4)}}</td><td>${{fmt(p.dr,3)}}</td>
    <td class="${{p.cost_sr<0?'pos':'neg'}}">${{fmtP(p.cost_sr)}}</td>
    <td class="${{p.fb>20?'neg':p.fb>5?'neu':'pos'}}">${{fmt(p.fb,1)}}</td><td>${{fmt(p.ds,4)}}</td>
  </tr>`).join('');

  // Gráfico evolución RQE por punto k
  new Chart(document.getElementById('cRQE'),{{type:'line',
    data:{{labels:FD.map(p=>'k='+p.k),datasets:[
      {{label:'RQE',data:FD.map(p=>p.rqe),borderColor:'#a371f7',backgroundColor:'#a371f733',
        fill:true,pointRadius:FD.map(p=>p.max_sr?7:4),
        pointBackgroundColor:FD.map(p=>p.max_sr?'#7cb342':'#a371f7'),tension:0.3,borderWidth:2}},
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:d=>{{const p=FD[d.dataIndex];
          return 'RQE='+fmt(p.rqe,4)+(p.rqe_norm!=null?' ('+(p.rqe_norm*100).toFixed(0)+'% del máx)':'');}}}}}}}},
      scales:{{x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},
        y:{{title:{{display:true,text:'RQE',color:'#8b949e'}},
          ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}}}}
    }}
  }});
}}

function buildPtSel(){{
  document.getElementById('pt-sel').innerHTML=FD.map((p,i)=>`<button class="pt-btn ${{i===selPt?'active':''}}" onclick="selK(${{i}},this)">k=${{p.k}}${{p.max_sr?' ★':''}}</button>`).join('');
}}

function selK(i,btn){{
  selPt=i;document.querySelectorAll('.pt-btn').forEach(b=>b.classList.remove('active'));btn.classList.add('active');renderComp();
}}

function renderComp(){{
  const pt=FD[selPt];
  const rows=ASSETS.map((a,i)=>({{
    a, c:COLORS[i],
    w:pt.weights[a], i5:pt.ic5[a], i95:pt.ic95[a], rc:pt.rc[a]
  }})).sort((x,y)=>y.w-x.w);

  const lim=2/ASSETS.length*100;
  const maxW=Math.max(...rows.map(r=>r.w),1);

  // KPIs resumen
  const nAct=rows.filter(r=>r.w>0.05).length;
  const rcMax=Math.max(...rows.map(r=>r.rc));
  const rcMaxA=rows.find(r=>r.rc===rcMax).a;
  const robustos=rows.filter(r=>(r.i95-r.i5)<10).length;
  document.getElementById('comp-summary').innerHTML=[
    ['ACTIVOS CON PESO',nAct+' / '+ASSETS.length,'#e6edf3'],
    ['RETORNO / VOL',fmt(pt.ret)+'% / '+fmt(pt.vol)+'%','#7cb342'],
    ['RQE',fmt(pt.rqe,4)+(pt.rqe_norm!=null?' ('+(pt.rqe_norm*100).toFixed(0)+'%)':''),'#a371f7'],
    ['MAYOR %RC',rcMaxA+' '+fmt(rcMax)+'%','#d98b2b'],
    ['PESOS ROBUSTOS',robustos+' / '+ASSETS.length,'#1f6feb'],
  ].map(([l,v,c])=>`<div class="cs-card"><div class="cs-label">${{l}}</div><div class="cs-val" style="color:${{c}}">${{v}}</div></div>`).join('');

  // Fichas por activo
  document.getElementById('comp-cards').innerHTML=rows.map(r=>{{
    const rng=r.i95-r.i5;
    const estab=rng<10?['ROBUSTO','sem-r']:rng<20?['MODERADO','sem-m']:['INCIERTO','sem-i'];
    const wpct=(r.w/maxW*100);
    const icL=(r.i5/maxW*100), icW=((r.i95-r.i5)/maxW*100);
    const rcFlag=r.rc>lim?'#e8503a':'#7cb342';
    return `<div class="afila">
      <div class="adot" style="background:${{r.c}}"></div>
      <div class="aname"><div class="atkr">${{r.a}}</div></div>
      <div class="abar-wrap"><div class="abar-track">
        <div class="abar-ic" style="left:${{icL}}%;width:${{icW}}%"></div>
        <div class="abar-fill" style="width:${{wpct}}%;background:${{r.c}}"></div>
      </div></div>
      <div class="aweight"><span class="aw-big">${{fmt(r.w)}}%</span><br><span class="aw-ic">IC ${{fmt(r.i5)}}–${{fmt(r.i95)}}</span></div>
      <div class="arc"><span class="arc-val" style="color:${{rcFlag}}">${{fmt(r.rc)}}%</span><br><span class="arc-lbl">riesgo</span></div>
      <div class="aestab"><span class="semaforo ${{estab[1]}}"><span class="sem-dot" style="background:currentColor"></span>${{estab[0]}}</span></div>
    </div>`;
  }}).join('');

  // Doughnut %RC
  if(cRC)cRC.destroy();
  cRC=new Chart(document.getElementById('cRC'),{{
    type:'doughnut',data:{{labels:ASSETS,datasets:[{{data:ASSETS.map(a=>pt.rc[a]),backgroundColor:COLORS,borderWidth:1,borderColor:'#0d1117'}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{position:'right',labels:{{color:'#8b949e',font:{{size:10}},generateLabels:ch=>ch.data.labels.map((l,i)=>({{text:l+' '+fmt(ch.data.datasets[0].data[i])+'%',fillStyle:COLORS[i],strokeStyle:COLORS[i],fontColor:'#8b949e'}}))}}}},tooltip:{{callbacks:{{label:d=>`${{d.label}}: ${{fmt(d.raw)}}% del riesgo`}}}}}}
    }}
  }});
}}

function renderOOS(){{
  const el=document.getElementById('oos-content');
  if(!OOS||!OOS.dates){{el.innerHTML='<div class="card" style="color:#8b949e;text-align:center;padding:40px">OOS no disponible.</div>';return;}}
  const m=OOS.metrics,jk=OOS.jk;
  el.innerHTML=`<div class="oos-kpis">${{['mep','mvo','ew'].map(k=>{{
    const mm=m[k],lbl={{mep:'MEP',mvo:'MVO max-Sharpe',ew:'Equal Weight'}}[k],c={{mep:'#e8503a',mvo:'#1f6feb',ew:'#8b949e'}}[k];
    return `<div class="oos-card"><div class="oos-label" style="color:${{c}}">${{lbl}}</div>
      <div class="oos-row"><span>Retorno anual</span><b class="${{mm.ret>0?'pos':'neg'}}">${{fmtP(mm.ret)}}</b></div>
      <div class="oos-row"><span>Volatilidad</span><b>${{fmt(mm.vol)}}%</b></div>
      <div class="oos-row"><span>Sharpe OOS</span><b class="${{mm.sr>0.5?'pos':'neu'}}">${{fmt(mm.sr,3)}}</b></div>
      <div class="oos-row"><span>Max Drawdown</span><b class="neg">${{fmt(mm.mdd)}}%</b></div></div>`;
  }}).join('')}}</div>
  <div class="card"><div class="card-title">NAV acumulado OOS (base 100) — ${{OOS.T_oos}} períodos</div>
    <div style="position:relative;height:300px"><canvas id="cOOS"></canvas></div></div>
  <div class="jk-box"><strong>Test Jobson-Korkie (1981) — MEP vs MVO</strong><br>
    H₀: SR_MEP=SR_MVO · Memmel (2003)<br>
    SR_MEP=${{fmt(jk.sr_mep,3)}} · SR_MVO=${{fmt(jk.sr_bench,3)}} · z=${{fmt(jk.z,3)}} · p=${{fmt(jk.pval,4)}} ·
    ${{jk.sig?'<b class="pos">✓ Significativo (α=5%)</b>':'<span class="neu">✗ No significativo — consistente con robustez MEP (Bajo Traver §5)</span>'}}
  </div>`;
  new Chart(document.getElementById('cOOS'),{{
    type:'line',data:{{labels:OOS.dates,datasets:[
      {{label:'MEP',data:OOS.nav_mep,borderColor:'#e8503a',borderWidth:2,pointRadius:0,fill:false,tension:0.2}},
      {{label:'MVO',data:OOS.nav_mvo,borderColor:'#1f6feb',borderWidth:1.5,pointRadius:0,fill:false,tension:0.2,borderDash:[4,3]}},
      {{label:'EW', data:OOS.nav_ew, borderColor:'#6e7681',borderWidth:1,pointRadius:0,fill:false,tension:0.2,borderDash:[2,4]}},
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:'#8b949e'}}}}}},
      scales:{{x:{{ticks:{{color:'#8b949e',maxTicksLimit:12}},grid:{{color:'#21262d'}}}},y:{{title:{{display:true,text:'NAV (base 100)',color:'#8b949e'}},ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}}}}
    }}
  }});
}}

function renderStats(){{
  document.getElementById('tb-stats').innerHTML=FD.map(p=>`<tr>
    <td>k=${{p.k}}${{p.max_sr?' ★':''}}</td><td>${{fmt(p.vol)}}</td><td>${{fmt(p.ret)}}</td>
    <td class="${{p.sr>1?'pos':''}}">${{fmt(p.sr,3)}}</td><td>${{fmt(p.rqe,5)}}</td><td>${{fmt(p.dr,4)}}</td>
    <td class="${{p.cost_sr<0?'pos':''}}">${{fmt(p.cost_sr,4)}}</td>
    <td class="${{p.fb>20?'neg':p.fb>5?'neu':''}}">${{fmt(p.fb,1)}}</td><td>${{fmt(p.ds,4)}}</td>
  </tr>`).join('');
  document.getElementById('tb-params').innerHTML=[
    ['N_sim','{cfg["N_sim"]}'],['K','{cfg["K"]}'],['block','{cfg["block_size"]}m'],
    ['w_max','{cfg["w_max"]*100:.0f}%'],['%RC_k','{rc_str}'],['z_alpha','{cfg["z_alpha"]}'],
    ['rf','{cfg["rf_annual"]*100:.1f}%'],['oos_train','{cfg["oos_train_months"]}m'],
    ['oos_rebal','{cfg["oos_rebal_freq"]}m'],['núcleos','{n_cpu}'],
  ].map(([k,v])=>`<tr><td style="color:#8b949e">${{k}}</td><td>${{v}}</td></tr>`).join('');
}}

renderIS();buildPtSel();
</script>
</body>
</html>"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"  Dashboard HTML guardado: {filename}")

def generate_static_dashboard(res, oos, params, cfg, filename):
    """Dashboard estático compacto (Opción A): embebe la superficie QEP (PNG en
    base64) + tabla MEP_optim + resultados OOS. Autocontenido."""
    import base64
    log.info(f"  Generando dashboard: {filename}")
    assets = params["assets"]; w = res["w"]
    rc = risk_contrib_pct(w, params["sigma"])
    img = ""
    chart = cfg.get("output_chart")
    if chart and os.path.exists(chart):
        with open(chart, "rb") as fh:
            img = "data:image/png;base64," + base64.b64encode(fh.read()).decode()
    def _estab(i):
        rng = (res["ic95"][i] - res["ic5"][i]) * 100
        return "ROBUSTO" if rng < 10 else ("MODERADO" if rng < 20 else "INCIERTO")
    rows = "".join(
        f"<tr><td>{a}</td><td>{w[i]*100:.1f}%</td>"
        f"<td>{res['ic5'][i]*100:.1f}–{res['ic95'][i]*100:.1f}</td>"
        f"<td>{rc[i]*100:.1f}%</td><td>{_estab(i)}</td></tr>"
        for i, a in enumerate(assets))
    oos_html = ""
    if oos and oos.get("metrics"):
        m = oos["metrics"]; jk = oos["jk_test"]
        orows = "".join(
            f"<tr><td>{m[k]['label']}</td><td>{m[k]['ret_ann']*100:.2f}%</td>"
            f"<td>{m[k]['vol_ann']*100:.2f}%</td><td>{m[k]['sharpe']:.3f}</td>"
            f"<td>{m[k]['mdd']*100:.2f}%</td></tr>" for k in ("mep", "mvo", "ew"))
        oos_html = (f"<h2>Walk-Forward OOS ({oos['T_oos']} periodos)</h2>"
                    f"<table><tr><th>Estrategia</th><th>Ret</th><th>Vol</th><th>Sharpe</th><th>MDD</th></tr>{orows}</table>"
                    f"<p>Jobson-Korkie MEP vs MVO: z={jk['z']:.3f} · p={jk['pval']:.4f} · "
                    f"{'significativo' if jk['significant'] else 'no significativo (equivalencia MEP≈MVO)'}</p>")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>MEP·RQE — Opción A</title><style>
body{{background:#0d1117;color:#c9d1d9;font-family:'IBM Plex Mono',Consolas,monospace;max-width:1100px;margin:24px auto;padding:0 16px}}
h1{{color:#e8503a}} h2{{color:#e6edf3;border-bottom:1px solid #30363d;padding-bottom:6px;margin-top:28px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} th,td{{border:1px solid #30363d;padding:6px 10px;text-align:right}}
th{{color:#8b949e}} td:first-child,th:first-child{{text-align:left}} img{{max-width:100%;border:1px solid #30363d;border-radius:8px}}
.kpi{{display:inline-block;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 16px;margin:4px}}
.kpi b{{color:#e6edf3;font-size:18px}}</style></head><body>
<h1>Motor MEP·RQE — Opción A (superficie QEP + Pareto)</h1>
<p>Bajo Traver (2025) · {datetime.now().strftime('%Y-%m-%d %H:%M')} · T={params['T']} · N_sim={cfg['N_sim']} · SBB block={cfg['block_size']}m</p>
<div>
<span class="kpi">Sharpe <b>{res['sr']:.3f}</b> (MVO {res['sr_mvo']:.3f})</span>
<span class="kpi">RQE <b>{res['rqe_norm']*100:.0f}%</b> del máx</span>
<span class="kpi">DR <b>{res['dr']:.2f}</b></span>
<span class="kpi">máx RC <b>{res['maxrc']*100:.1f}%</b></span>
<span class="kpi">receta δS*={res['ds_star']:.3f} · rc*={fmt_rc(res['rc_star'])}</span>
</div>
<h2>Superficie QEP + frente de Pareto</h2>
{('<img src="' + img + '">') if img else '<p>(gráfico no disponible)</p>'}
<h2>MEP_optim — composición (mediana de {res['n_valid']} sims)</h2>
<table><tr><th>Activo</th><th>Peso</th><th>IC 5–95%</th><th>%RC</th><th>Estabilidad</th></tr>{rows}</table>
{oos_html}
</body></html>"""
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info(f"  Dashboard guardado: {filename}")

# ─────────────────────────────────────────────────────────────────────────────
# ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("="*60)
    log.info("Motor MVO + Resampling + MEP·RQE — Funcional v2.3/v2.4")
    log.info(f"  Núcleos disponibles: {multiprocessing.cpu_count()}")
    log.info("="*60)

    cfg        = CONFIG
    ann_factor = {"monthly":12,"weekly":52,"daily":252}[cfg["freq"]]

    df_prices              = fetch_prices_ibkr(cfg)
    # Separar el tipo libre de riesgo (€STR) de los activos
    df_prices, rf_series, rf_ann = split_rf(df_prices, cfg, ann_factor)
    if rf_series is not None and cfg.get("rf_mode") == "estr":
        cfg = {**cfg, "rf_annual": rf_ann}
        log.info(f"  rf (€STR) medio anualizado: {rf_ann*100:.2f}% "
                 f"(sustituye al rf constante)")
    df_ret, excluded       = prepare_returns(df_prices, cfg)
    params                 = estimate_base_params(df_ret, ann_factor)
    # Motor Opción A: superficie QEP + Pareto (decide receta) → resampling → MEP_optim
    res                    = run_mep_optim(params, cfg)

    T_min = cfg["oos_train_months"] + cfg["oos_rebal_freq"]
    if params["T"] >= T_min:
        oos_results = run_walk_forward(df_ret, cfg, rf_series=rf_series)
        if oos_results:
            export_oos_excel(oos_results, cfg, cfg["output_oos"])
    else:
        log.warning(f"  OOS omitido: T={params['T']} < {T_min} meses")
        oos_results = {}

    print_report(res, params, excluded, cfg)
    export_excel(res, oos_results, params, excluded, cfg, cfg["output_excel"])
    plot_qep_surface(res, params, cfg, cfg["output_chart"])
    generate_static_dashboard(res, oos_results, params, cfg, cfg["output_html"])

    log.info("Pipeline completado.")
    log.info(f"  IS Excel  → {cfg['output_excel']}")
    log.info(f"  OOS Excel → {cfg['output_oos']}")
    log.info(f"  Gráfico   → {cfg['output_chart']}")
    log.info(f"  Dashboard → {cfg['output_html']}")


if __name__ == "__main__":
    # CRÍTICO en Windows: ProcessPoolExecutor requiere este guard
    multiprocessing.freeze_support()
    main()
