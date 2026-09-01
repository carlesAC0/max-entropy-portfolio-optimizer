# -*- coding: utf-8 -*-
"""
data_cache.py — Capa de datos para la app interactiva del Motor MEP·RQE.

Responsabilidades:
  · Universo persistente (cache/universe.json): tickers conocidos + conId/exchange/divisa.
  · Cache de precios mensual (cache/precios.csv) con caducidad de 1 DÍA.
      - Si el cache es de HOY  → se reutiliza (re-optimización instantánea, sin IBKR).
      - Si es de ayer o falta  → se redescarga de IBKR (no se pierde el dato del día).
  · Descarga IBKR reutilizando el motor (fetch_prices_ibkr).
  · Resolución y alta de tickers nuevos por búsqueda en IBKR (reqMatchingSymbols).

Reutiliza el pipeline existente sin tocarlo: importa funciones de mvo_resampling_mep.
"""

import io
import os
import re
import json
import logging
import datetime as dt

import numpy as np
import pandas as pd

import mvo_resampling_mep as engine

log = logging.getLogger("MEP.cache")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

UNIVERSE_FILE = os.path.join(CACHE_DIR, "universe.json")
PRICES_FILE   = os.path.join(CACHE_DIR, "precios.csv")
META_FILE     = os.path.join(CACHE_DIR, "precios_meta.json")
EXTERNAL_FILE = os.path.join(CACHE_DIR, "externos.csv")
EXTERNAL_META = os.path.join(CACHE_DIR, "externos_meta.json")

# ─────────────────────────────────────────────────────────────────────────────
# Universo semilla (núcleo del TFM + satélites AIGA/IEMA).
#   core    = entra al resampling por diseño (los 8 estructurales).
#   default = marcado al abrir la app.
# ─────────────────────────────────────────────────────────────────────────────
SEED_UNIVERSE = {
    "XDWT": {"conId": 227264004, "exchange": "IBIS2",  "currency": "EUR", "clase": "RV Tecnología",      "core": True,  "default": True},
    "XDWS": {"conId": 227264011, "exchange": "IBIS2",  "currency": "EUR", "clase": "RV Consumer Staples", "core": True,  "default": True},
    "XDWH": {"conId": 227263992, "exchange": "IBIS2",  "currency": "EUR", "clase": "RV Healthcare",       "core": True,  "default": True},
    "XDW0": {"conId": 227263991, "exchange": "IBIS2",  "currency": "EUR", "clase": "RV Energía",          "core": True,  "default": True},
    "IGLN": {"conId": 86656182,  "exchange": "LSEETF", "currency": "USD", "clase": "Oro (estructural)",   "core": True,  "default": True},
    "EPRA": {"conId": 255731263, "exchange": "SBF",    "currency": "EUR", "clase": "Real Estate",         "core": True,  "default": True},
    "IBGS": {"conId": 54233539,  "exchange": "LSEETF", "currency": "EUR", "clase": "Bono gob. 1-3Y (→MTA)", "core": True, "default": True},
    "IBGM": {"conId": 68489986,  "exchange": "LSEETF", "currency": "EUR", "clase": "Bono gob. 7-10Y (→MTD)", "core": True, "default": True},
    "AIGA": {"conId": 41015861,  "exchange": "LSE",    "currency": "USD", "clase": "Agro/Materiales (sat.)", "core": False, "default": False},
    "IEMA": {"conId": 79000451,  "exchange": "LSEETF", "currency": "USD", "clase": "RV Emergentes (sat.)",   "core": False, "default": False},
}

# ─────────────────────────────────────────────────────────────────────────────
# Universo
# ─────────────────────────────────────────────────────────────────────────────

def load_universe() -> dict:
    if not os.path.exists(UNIVERSE_FILE):
        save_universe(SEED_UNIVERSE)
        return dict(SEED_UNIVERSE)
    with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_universe(u: dict) -> None:
    with open(UNIVERSE_FILE, "w", encoding="utf-8") as f:
        json.dump(u, f, indent=2, ensure_ascii=False)


def _universe_to_cfg(u: dict, tickers) -> dict:
    """Convierte el universo (dict) al formato cfg['universe']: ticker→(conId,exch,cur)."""
    return {t: (u[t]["conId"], u[t]["exchange"], u[t]["currency"])
            for t in tickers if t in u}

# ─────────────────────────────────────────────────────────────────────────────
# Cache de precios (TTL = 1 día)
# ─────────────────────────────────────────────────────────────────────────────

def read_meta() -> dict:
    if not os.path.exists(META_FILE):
        return {}
    with open(META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def cache_is_fresh() -> bool:
    """True solo si el cache se descargó HOY y el CSV existe."""
    m = read_meta()
    return bool(m) and m.get("date") == dt.date.today().isoformat() \
        and os.path.exists(PRICES_FILE)


def read_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    return df


def write_prices(df: pd.DataFrame) -> None:
    df.to_csv(PRICES_FILE)
    meta = {
        "date":    dt.date.today().isoformat(),
        "ts":      dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickers": list(df.columns),
        "rows":    int(len(df)),
        "start":   str(df.index.min().date()) if len(df) else None,
        "end":     str(df.index.max().date()) if len(df) else None,
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _download(tickers, base_cfg=None) -> pd.DataFrame:
    """Descarga de IBKR los `tickers` indicados reutilizando el motor."""
    u = load_universe()
    cfg = dict(base_cfg or engine.CONFIG)
    cfg["universe"] = _universe_to_cfg(u, tickers)
    if not cfg["universe"]:
        raise ValueError("Ningún ticker válido para descargar.")
    log.info(f"Descargando de IBKR: {list(cfg['universe'])}")
    return engine.fetch_prices_ibkr(cfg)


def get_prices(base_cfg=None, force=False, allow_stale=True):
    """
    Devuelve (df_precios_universo_completo, info).
    info = {"source": "cache"|"ibkr"|"stale", "date":..., "stale":bool}

    · cache fresco (hoy) y no force → CSV.
    · si no → intenta IBKR (todo el universo conocido) y reescribe el cache.
    · si IBKR falla pero hay CSV antiguo y allow_stale → usa el antiguo (marcado stale).
    """
    if not force and cache_is_fresh():
        return read_prices(), {"source": "cache", "stale": False,
                               "date": read_meta().get("date")}

    tickers = list(load_universe())
    try:
        df = _download(tickers, base_cfg)
        write_prices(df)
        return df, {"source": "ibkr", "stale": False, "date": dt.date.today().isoformat()}
    except Exception as e:
        log.error(f"Fallo descarga IBKR: {e}")
        if allow_stale and os.path.exists(PRICES_FILE):
            log.warning("Usando cache antiguo (stale) por fallo de IBKR.")
            return read_prices(), {"source": "stale", "stale": True,
                                   "date": read_meta().get("date"), "error": str(e)}
        raise

# ─────────────────────────────────────────────────────────────────────────────
# Fuente secundaria: fondos cargados por XLS/CSV desde la app
#   · Un archivo con TODOS los fondos: 1ª columna = fechas, una columna por
#     fondo (cabecera = nombre). Autodetección de retornos (%, tanto por uno)
#     o niveles (VL), y de frecuencia (diaria → se compone a mensual).
#   · Se persisten como ÍNDICE DE PRECIO mensual (base 100) en cache/externos.csv,
#     así entran al pipeline por el mismo camino que los precios de IBKR.
# ─────────────────────────────────────────────────────────────────────────────

def _num_series(s: pd.Series) -> pd.Series:
    """Columna a numérico tolerando formato español (1.234,56 · 0,35 · '0,35%')."""
    if not pd.api.types.is_numeric_dtype(s):
        st = s.astype(str).str.strip().str.replace("%", "", regex=False)
        if st.str.contains(",", na=False).any():
            st = st.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        s = st
    return pd.to_numeric(s, errors="coerce")


def _quitar_spikes(level: pd.Series, umbral: float = 0.4, max_pasadas: int = 3) -> tuple:
    """
    Elimina errores puntuales de VL (p.ej. un 9,49 entre valores de ~110):
    observaciones cuyo log-retorno supera ±umbral Y revierte con signo contrario
    en el dato siguiente. Un movimiento real de mercado no rebota espejo al día
    siguiente; un dedo/dato corrupto sí. Devuelve (serie_limpia, n_eliminados).
    """
    n_out = 0
    for _ in range(max_pasadas):
        v = level.dropna()
        if len(v) < 3:
            break
        r = np.log(v).diff()
        spike = (r.abs() > umbral) & (r.shift(-1).abs() > umbral) & (r * r.shift(-1) < 0)
        if not spike.any():
            break
        level = level.drop(v.index[spike])
        n_out += int(spike.sum())
    return level, n_out


def _resample_month_end(s: pd.Series) -> pd.Series:
    try:
        return s.resample("ME").last()
    except ValueError:                      # pandas < 2.2 no conoce "ME"
        return s.resample("M").last()


def _parse_external(df_raw: pd.DataFrame, fname: str):
    """
    Interpreta el XLS/CSV → (df_mensual_niveles, info_por_fondo).
    Soporta dos disposiciones:
      · UNA columna de fechas + N columnas de fondos, o
      · PARES repetidos Fecha|Fondo (cada fondo con su propia columna de fechas
        y su propio rango histórico), con columnas vacías de separación.
    Cada columna de valores usa la columna de fechas más reciente a su izquierda.
    Autodetecta POR FONDO si los valores son niveles (VL: todo >0 y mediana >2)
    o retornos (% si |p95|>0.15, imposible en retorno diario en tanto por uno),
    y si la frecuencia es diaria (se compone a mensual) o ya mensual.
    Tolera cabeceras repetidas en mitad del archivo y miles españoles (40.376,22).
    """
    if df_raw.shape[1] < 2:
        raise ValueError("El archivo necesita ≥2 columnas: fechas + fondos.")

    def _es_col_fecha(col_name, s):
        if pd.api.types.is_datetime64_any_dtype(s):
            return True
        if str(col_name).strip().lower().startswith("fecha"):
            return True
        muestra = s.dropna().astype(str).str.strip().head(40)
        if len(muestra) < 5:
            return False
        # Forma de fecha (dd/mm/aaaa o aaaa-mm-dd): evita que dateutil confunda
        # VL con coma decimal ("9,843" → año 843) con fechas de verdad.
        if muestra.str.match(r"^\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}(\s.*)?$").mean() < 0.6:
            return False
        parsed = pd.to_datetime(muestra, dayfirst=True, errors="coerce")
        return (parsed.notna() & parsed.dt.year.between(1990, 2100)).mean() >= 0.6

    reserved = set(load_universe()) | {"EUR_RF", engine.CONFIG.get("benchmark_ticker")}
    out, info = {}, {}
    fechas = None                              # columna de fechas activa (la última vista)
    for c in df_raw.columns:
        s = df_raw[c]
        if s.dropna().empty:
            continue                           # columnas separadoras vacías
        if _es_col_fecha(c, s):
            fechas = pd.to_datetime(s, dayfirst=True, errors="coerce")
            continue
        if fechas is None:
            continue                           # valores sin columna de fechas previa
        name = re.sub(r"[^\w]+", "_", str(c).strip().upper()).strip("_")[:24] or "FONDO"
        name = re.sub(r"_\d+$", "", name) or "FONDO"   # sufijos .1/.2 de cabeceras duplicadas
        if name in reserved or name in out:
            name += "_XLS"
        vals = _num_series(s)
        ok = fechas.notna().values & vals.notna().values
        if ok.sum() < 12:
            log.warning(f"  {fname}: columna '{c}' ignorada (<12 valores numéricos con fecha).")
            continue
        v = pd.Series(vals.values[ok], index=pd.DatetimeIndex(fechas[ok])).sort_index()
        v = v[~v.index.duplicated(keep="last")]
        gap = v.index.to_series().diff().dt.days.median()
        freq = "diaria" if (gap or 1) <= 7 else "mensual"
        if (v > 0).all() and v.median() > 2.0:
            kind, level = "niveles (VL)", v
        else:
            r = v / 100.0 if v.abs().quantile(0.95) > 0.15 else v
            kind = "retornos en %" if r is not v else "retornos"
            level = 100.0 * (1.0 + r).cumprod()
        level, n_spikes = _quitar_spikes(level)
        if n_spikes:
            log.warning(f"  {fname}: '{name}' — {n_spikes} dato(s) corrupto(s) eliminado(s) "
                        f"(salto ±40% con reversión inmediata).")
        m = _resample_month_end(level).dropna()
        out[name] = m
        info[name] = {"file": fname, "kind": kind, "freq": freq,
                      "start": str(m.index.min().date()), "end": str(m.index.max().date()),
                      "n_meses": int(len(m))}
    if not out:
        raise ValueError("Ninguna columna se pudo interpretar como fondo. Formatos válidos: "
                         "una columna de fechas + fondos, o pares Fecha|Fondo repetidos.")
    return pd.DataFrame(out), info


def add_external_file(raw_bytes: bytes, filename: str) -> dict:
    """Parsea el archivo subido y lo fusiona con los externos persistidos
    (una columna nueva con el mismo nombre reemplaza a la anterior)."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".xlsx", ".xls"):
        df_raw = pd.read_excel(io.BytesIO(raw_bytes))
    elif ext == ".csv":
        for enc in ("utf-8-sig", "latin-1"):
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        df_raw = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    else:
        raise ValueError("Formato no soportado: usa .xlsx, .xls o .csv.")
    dfm, info = _parse_external(df_raw, filename)

    old_df, old_meta = load_external()
    if old_df is not None:
        keep = [c for c in old_df.columns if c not in dfm.columns]
        if keep:
            dfm = merge_external(old_df[keep], dfm)
        old_meta = {k: v for k, v in old_meta.items() if k in keep}
    meta = {**old_meta, **info}
    dfm.to_csv(EXTERNAL_FILE)
    with open(EXTERNAL_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info(f"Externos: {filename} → {list(info)} ({info[list(info)[0]]['freq']})")
    return info


def load_external():
    """(df_mensual_niveles, meta) de los fondos XLS persistidos, o (None, {})."""
    if not os.path.exists(EXTERNAL_FILE):
        return None, {}
    df = pd.read_csv(EXTERNAL_FILE, index_col=0, parse_dates=True)
    meta = {}
    if os.path.exists(EXTERNAL_META):
        with open(EXTERNAL_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return df, meta


def remove_external(name: str) -> bool:
    """Elimina un fondo XLS (o todos, con name='*')."""
    df, meta = load_external()
    if df is None:
        return False
    if name == "*":
        os.remove(EXTERNAL_FILE)
        if os.path.exists(EXTERNAL_META):
            os.remove(EXTERNAL_META)
        return True
    if name not in df.columns:
        raise ValueError(f"{name} no es un fondo XLS conocido.")
    df = df.drop(columns=[name])
    meta.pop(name, None)
    if df.empty or not len(df.columns):
        return remove_external("*")
    df.to_csv(EXTERNAL_FILE)
    with open(EXTERNAL_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return True


def merge_external(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Une dos tablas mensuales POR MES CALENDARIO: fuentes distintas pueden
    cerrar el mes en días distintos (último hábil vs fin de mes natural)."""
    if df_a is None:
        return df_b
    if df_b is None:
        return df_a
    a, b = df_a.copy(), df_b.copy()
    a.index = pd.DatetimeIndex(a.index).to_period("M")
    b.index = pd.DatetimeIndex(b.index).to_period("M")
    a, b = a[~a.index.duplicated(keep="last")], b[~b.index.duplicated(keep="last")]
    out = a.join(b, how="outer").sort_index()
    out.index = out.index.to_timestamp(how="end").normalize()
    return out


def get_prices_cached_only():
    """Cache de IBKR si existe (aunque esté caducado), SIN intentar descargar.
    Para el modo 'solo fondos XLS': aporta €STR/benchmark si están, nada más."""
    if os.path.exists(PRICES_FILE):
        fresh = cache_is_fresh()
        return read_prices(), {"source": "cache" if fresh else "stale",
                               "stale": not fresh, "date": read_meta().get("date")}
    return None, None

# ─────────────────────────────────────────────────────────────────────────────
# Alta de tickers nuevos
# ─────────────────────────────────────────────────────────────────────────────

def resolve_symbol(query: str, base_cfg=None):
    """
    Busca un símbolo en IBKR (reqMatchingSymbols) y devuelve candidatos:
    [{symbol, conId, exchange, currency, secType, name}].
    Requiere TWS abierto.
    """
    from ib_insync import IB
    cfg = dict(base_cfg or engine.CONFIG)
    ib = IB()
    ib.connect(cfg["ibkr_host"], cfg["ibkr_port"], clientId=cfg["ibkr_client"] + 5)
    try:
        matches = ib.reqMatchingSymbols(query)
        out = []
        for m in matches or []:
            c = m.contract
            if c.secType not in ("STK", "ETF"):
                continue
            out.append({
                "symbol":   c.symbol,
                "conId":    c.conId,
                "exchange": c.primaryExchange or c.exchange or "",
                "currency": c.currency,
                "secType":  c.secType,
                "name":     getattr(m, "description", "") or "",
            })
        return out
    finally:
        ib.disconnect()


def add_ticker(symbol, conId, exchange, currency, clase="", core=False, base_cfg=None):
    """
    Registra un ticker nuevo en el universo, descarga su histórico y lo fusiona
    al cache de precios (si existe). Devuelve la entrada del universo.
    """
    symbol = symbol.upper().strip()
    u = load_universe()
    u[symbol] = {
        "conId": int(conId), "exchange": exchange, "currency": currency,
        "clase": clase or "Añadido", "core": bool(core), "default": False,
    }
    save_universe(u)

    df_new = _download([symbol], base_cfg)
    if os.path.exists(PRICES_FILE):
        df = read_prices()
        if symbol in df.columns:
            df = df.drop(columns=[symbol])
        df = df.join(df_new[[symbol]], how="outer")
        write_prices(df)
    else:
        write_prices(df_new)
    return u[symbol]


def remove_ticker(symbol, base_cfg=None):
    """Elimina un ticker del universo y del cache (no permite borrar el núcleo)."""
    symbol = symbol.upper().strip()
    u = load_universe()
    if symbol in u and u[symbol].get("core"):
        raise ValueError(f"{symbol} es núcleo estructural; no se elimina, solo se desmarca.")
    u.pop(symbol, None)
    save_universe(u)
    if os.path.exists(PRICES_FILE):
        df = read_prices()
        if symbol in df.columns:
            df = df.drop(columns=[symbol])
            write_prices(df)
    return True
