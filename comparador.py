# -*- coding: utf-8 -*-
"""
comparador.py — Pestaña "Paper vs Real": réplica del MEP en la cuenta paper
frente a la cartera real.

Idea
  · No hace falta una foto diaria: basta con saber QUÉ tiene cada cuenta (fotos
    de posiciones + ejecuciones) y los CIERRES DIARIOS de cada activo, que se
    descargan a posteriori. Con eso se reconstruye la rentabilidad día a día.
  · La rentabilidad se calcula sobre POSICIONES (cantidad × precio), no sobre el
    NLV: una aportación o retirada de efectivo no se confunde con ganancia.

Datos (cache/comparador/)
  · snapshots.json   fotos de posiciones por cuenta {"paper":[...], "real":[...]}
  · ejecuciones.json ejecuciones por cuenta (dedup por execId; IBKR solo da 7 días)
  · precios.csv      cierres diarios en EUR (+ EUR_RF = XEON, €STR) desde el inicio

Seguridad
  · Todas las conexiones son readonly=True: este módulo NUNCA envía órdenes.
  · La cuenta se identifica por su NÚMERO, no por el puerto: si el puerto 7497
    devolviera una cuenta real (o el 7496 una paper) se rechaza la foto.

Independiente del motor: no modifica nada del pipeline MEP.
"""

import os
import json
import logging
import datetime as dt

import numpy as np
import pandas as pd

log = logging.getLogger("MEP.comparador")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CMP_DIR   = os.path.join(CACHE_DIR, "comparador")
SNAP_FILE = os.path.join(CMP_DIR, "snapshots.json")
EXEC_FILE = os.path.join(CMP_DIR, "ejecuciones.json")
PX_FILE   = os.path.join(CMP_DIR, "precios.csv")
LAST_FILE = os.path.join(CACHE_DIR, "last_result.json")
INICIO_FILE = os.path.join(CMP_DIR, "inicio.json")   # inicio fijado (no se mueve al reoptimizar)

HOST = "127.0.0.1"
# Los números de cuenta son datos personales: viven en cuentas_ibkr.json (en
# .gitignore, nunca se publica). Plantilla: cuentas_ibkr.example.json.
CUENTAS_FILE = os.path.join(BASE_DIR, "cuentas_ibkr.json")
CUENTAS = {
    "paper": {"account": None, "port": 7497, "client": 31},
    "real":  {"account": None, "port": 7496, "client": 32},
}
if os.path.exists(CUENTAS_FILE):
    with open(CUENTAS_FILE, "r", encoding="utf-8") as _f:
        for _t, _v in json.load(_f).items():
            if _t in CUENTAS:
                CUENTAS[_t]["account"] = _v
CLIENT_PRECIOS = 33
TZ_LOCAL = "Europe/Madrid"
ANN = 252
MIN_DIAS_ANUAL = 20          # sesiones mínimas para mostrar cifras anualizadas
RF_CONTRATO = (46041702, "IBIS2", "EUR")      # XEON (€STR), igual que el motor
# Activo de ejecución → ticker del modelo (inverso de CONFIG["proxy_map"])
EJEC_A_MODELO = {"MTA": "IBGS", "MTD": "IBGM"}


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia
# ─────────────────────────────────────────────────────────────────────────────

def _load(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(path, obj):
    os.makedirs(CMP_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def _modelo(symbol):
    return EJEC_A_MODELO.get(symbol, symbol)


def _load_snaps():
    """Fotos ordenadas por hora (la última de cada lista es la más reciente)."""
    snaps = _load(SNAP_FILE, {"paper": [], "real": []})
    for t in ("paper", "real"):
        snaps[t] = sorted(snaps.get(t, []), key=lambda s: s["ts"])
    return snaps


def _core():
    obj = _objetivo()
    return set(obj["assets"]) if obj else set()


def _core_invertido():
    """Activos del núcleo con peso > 0 en el objetivo (w_min = 0 permite pesos nulos,
    que la paper no compra)."""
    obj = _objetivo()
    return {a for a, w in obj["w"].items() if w > 0} if obj else set()


# ─────────────────────────────────────────────────────────────────────────────
# Lectura de cuentas (solo lectura)
# ─────────────────────────────────────────────────────────────────────────────

def _conectar(port, client):
    from ib_insync import IB
    ib = IB()
    ib.connect(HOST, port, clientId=client, readonly=True, timeout=6)
    return ib


def _es_paper(account):
    return str(account).startswith("DU")


def foto_cuenta(tipo):
    """Lee posiciones + ejecuciones de una cuenta y las persiste.
    Devuelve un dict de estado {ok, msg}."""
    cfg = CUENTAS[tipo]
    if not cfg["account"]:
        return {"ok": False, "msg": f"sin número de cuenta {tipo}: crea cuentas_ibkr.json "
                                    f"a partir de cuentas_ibkr.example.json"}
    try:
        ib = _conectar(cfg["port"], cfg["client"])
    except Exception as e:
        return {"ok": False, "msg": f"TWS {tipo} no disponible en el puerto {cfg['port']} ({e.__class__.__name__})"}
    try:
        cuentas = ib.managedAccounts()
        # Guarda: la cuenta tiene que ser la esperada y del tipo esperado.
        if cfg["account"] not in cuentas:
            return {"ok": False, "msg": f"el puerto {cfg['port']} está en {cuentas}, no en {cfg['account']}: foto rechazada"}
        if _es_paper(cfg["account"]) != (tipo == "paper"):
            return {"ok": False, "msg": f"cuenta {cfg['account']} no es de tipo {tipo}: foto rechazada"}
        acct = cfg["account"]

        fx = {"EUR": 1.0}
        for v in ib.accountValues(acct):
            if v.tag == "ExchangeRate" and v.currency not in ("", "BASE"):
                try:
                    fx[v.currency] = float(v.value)       # EUR por 1 unidad de divisa
                except ValueError:
                    pass

        pos = []
        port = [p for p in ib.portfolio(acct) if p.contract.secType == "STK" and p.position]
        # TWS no siempre publica ExchangeRate (p.ej. sin saldo en esa divisa): se pide el par
        from ib_insync import Forex
        for cur in {p.contract.currency for p in port} - set(fx):
            try:
                b = ib.reqHistoricalData(Forex("EUR" + cur), endDateTime="", durationStr="5 D",
                                         barSizeSetting="1 day", whatToShow="MIDPOINT",
                                         useRTH=False, formatDate=1)
                if b:
                    fx[cur] = 1.0 / b[-1].close
            except Exception:
                pass
        if port:
            for p in port:
                c = p.contract
                pos.append({"conId": c.conId, "symbol": c.symbol,
                            "exchange": c.primaryExchange or c.exchange, "currency": c.currency,
                            "qty": float(p.position), "precio": float(p.marketPrice),
                            "valor_eur": float(p.marketValue) * fx.get(c.currency, np.nan)})
        else:
            for p in ib.positions(acct):
                c = p.contract
                if c.secType == "STK" and p.position:
                    pos.append({"conId": c.conId, "symbol": c.symbol,
                                "exchange": c.primaryExchange or c.exchange, "currency": c.currency,
                                "qty": float(p.position), "precio": None, "valor_eur": None})

        snaps = _load_snaps()
        snaps.setdefault(tipo, []).append({
            "ts": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "account": acct, "source": f"TWS {cfg['port']}", "positions": pos})
        _save(SNAP_FILE, snaps)

        ex = _load(EXEC_FILE, {"paper": {}, "real": {}})
        bucket = ex.setdefault(tipo, {})
        n_new = 0
        for f in ib.reqExecutions():
            e, c = f.execution, f.contract
            if e.acctNumber != acct or c.secType != "STK" or e.execId in bucket:
                continue
            t = pd.Timestamp(e.time)
            t = (t.tz_convert(TZ_LOCAL) if t.tzinfo else t.tz_localize("UTC").tz_convert(TZ_LOCAL)).tz_localize(None)
            bucket[e.execId] = {"time": t.strftime("%Y-%m-%d %H:%M:%S"), "conId": c.conId,
                                "symbol": c.symbol, "exchange": c.primaryExchange or c.exchange,
                                "currency": c.currency,
                                "qty": float(e.shares) * (1 if e.side == "BOT" else -1),
                                "price": float(e.price)}
            n_new += 1
        _save(EXEC_FILE, ex)
        return {"ok": True, "msg": f"{acct}: {len(pos)} posiciones, {n_new} ejecuciones nuevas"}
    except Exception as e:
        log.exception(f"foto {tipo}")
        return {"ok": False, "msg": f"error leyendo {tipo}: {e}"}
    finally:
        ib.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# Reconstrucción de cantidades por día
# ─────────────────────────────────────────────────────────────────────────────

def _contratos(snaps, ex):
    """conId → {symbol, exchange, currency} de todo lo que han tenido las cuentas."""
    out = {}
    for tipo in ("paper", "real"):
        for s in snaps.get(tipo, []):
            for p in s["positions"]:
                out[p["conId"]] = {k: p[k] for k in ("symbol", "exchange", "currency")}
        for e in ex.get(tipo, {}).values():
            out[e["conId"]] = {k: e[k] for k in ("symbol", "exchange", "currency")}
    return out


def _qty_al_cierre(snaps_t, execs_t, dia, solo_pasado=False):
    """Cantidades por conId al CIERRE del día `dia`.
    Se parte de la foto más cercana POSTERIOR y se deshacen las ejecuciones entre
    el día y la foto; si no hay foto posterior, la anterior + ejecuciones.
    solo_pasado=True usa siempre la anterior + ejecuciones: no da por hecho nada
    que no se haya visto ya ese día (si faltan ejecuciones, la foto posterior
    adelantaría cambios que aún no habían ocurrido)."""
    fin = pd.Timestamp(dia) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    ts = [pd.Timestamp(s["ts"]) for s in snaps_t]
    post = [] if solo_pasado else [i for i, t in enumerate(ts) if t >= fin]
    if post:
        s = snaps_t[post[0]]
        q = {p["conId"]: p["qty"] for p in s["positions"]}
        for e in execs_t:
            te = pd.Timestamp(e["time"])
            if fin < te <= ts[post[0]]:
                q[e["conId"]] = q.get(e["conId"], 0.0) - e["qty"]
    else:
        prev = [i for i, t in enumerate(ts) if t < fin]
        if not prev:
            return {}
        s = snaps_t[prev[-1]]
        q = {p["conId"]: p["qty"] for p in s["positions"]}
        for e in execs_t:
            te = pd.Timestamp(e["time"])
            if ts[prev[-1]] < te <= fin:
                q[e["conId"]] = q.get(e["conId"], 0.0) + e["qty"]
    return {k: v for k, v in q.items() if abs(v) > 1e-9}


def _avisos_consistencia(tipo, snaps_t, execs_t):
    """Si dos fotos consecutivas difieren y las ejecuciones guardadas no lo
    explican, hubo operaciones no registradas (IBKR solo devuelve 7 días)."""
    out = []
    for a, b in zip(snaps_t, snaps_t[1:]):
        ta, tb = pd.Timestamp(a["ts"]), pd.Timestamp(b["ts"])
        q = {p["conId"]: p["qty"] for p in a["positions"]}
        for e in execs_t:
            if ta < pd.Timestamp(e["time"]) <= tb:
                q[e["conId"]] = q.get(e["conId"], 0.0) + e["qty"]
        qb = {p["conId"]: p["qty"] for p in b["positions"]}
        if any(abs(q.get(k, 0) - qb.get(k, 0)) > 1e-6 for k in set(q) | set(qb)):
            out.append(f"{tipo}: cambios de posición entre {a['ts'][:10]} y {b['ts'][:10]} "
                       f"sin ejecuciones registradas; en ese tramo el día exacto es aproximado")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Precios diarios (EUR)
# ─────────────────────────────────────────────────────────────────────────────

def descargar_precios(desde):
    """Cierres diarios en EUR de todos los contratos que han tenido las cuentas,
    más XEON (EUR_RF). Usa la sesión real (suscripción de datos); si no está,
    la paper. Devuelve estado {ok, msg}."""
    from ib_insync import Contract, Forex, util
    snaps = _load_snaps()
    ex = _load(EXEC_FILE, {"paper": {}, "real": {}})
    contratos = _contratos({"real": snaps.get("real", [])}, {"real": ex.get("real", {})})
    core = _core()
    for cid, info in _contratos({"paper": snaps.get("paper", [])}, {"paper": ex.get("paper", {})}).items():
        if _modelo(info["symbol"]) in core:          # de la paper solo interesa el núcleo
            contratos.setdefault(cid, info)
    if not contratos:
        return {"ok": False, "msg": "sin posiciones registradas todavía"}

    ib, puerto = None, None
    for tipo in ("real", "paper"):
        try:
            ib = _conectar(CUENTAS[tipo]["port"], CLIENT_PRECIOS)
            puerto = CUENTAS[tipo]["port"]
            break
        except Exception:
            ib = None
    if ib is None:
        return {"ok": False, "msg": "ninguna sesión TWS disponible para descargar precios"}

    dias = (dt.date.today() - pd.Timestamp(desde).date()).days + 15
    dur = f"{max(dias, 30)} D" if dias <= 365 else f"{int(np.ceil(dias / 365))} Y"

    def _serie(contract, what):
        try:
            b = ib.reqHistoricalData(contract, endDateTime="", durationStr=dur,
                                     barSizeSetting="1 day", whatToShow=what,
                                     useRTH=what != "MIDPOINT", formatDate=1)
        except Exception:
            b = None
        if not b:
            return None
        d = util.df(b)[["date", "close"]]
        s = pd.Series(d["close"].values, index=pd.to_datetime(d["date"]))
        return s[~s.index.duplicated(keep="last")]

    try:
        cols, faltan, fxs = {}, [], {}
        for cid, info in contratos.items():
            c = Contract(conId=int(cid), exchange="SMART", primaryExchange=info["exchange"],
                         currency=info["currency"])
            s = _serie(c, "ADJUSTED_LAST")
            if s is None:
                s = _serie(c, "TRADES")
            if s is None:
                faltan.append(info["symbol"]); continue
            cur = info["currency"]
            if cur != "EUR":
                if cur not in fxs:
                    fx = _serie(Forex("EUR" + cur), "MIDPOINT")     # cur por 1 EUR
                    fxs[cur] = fx
                fx = fxs[cur]
                if fx is None:
                    faltan.append(f"{info['symbol']} (FX {cur})"); continue
                s = s / fx.reindex(s.index.union(fx.index)).ffill().reindex(s.index)
            cols[str(cid)] = s
        rf_c = Contract(conId=RF_CONTRATO[0], exchange="SMART", primaryExchange=RF_CONTRATO[1],
                        currency=RF_CONTRATO[2])
        rf = _serie(rf_c, "ADJUSTED_LAST")
        if rf is None:
            rf = _serie(rf_c, "TRADES")
        if rf is not None:
            cols["EUR_RF"] = rf
    finally:
        ib.disconnect()

    if not cols:
        return {"ok": False, "msg": f"IBKR no devolvió precios por el puerto {puerto} "
                                    f"(¿datos de mercado bloqueados por otra sesión?)"}
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    os.makedirs(CMP_DIR, exist_ok=True)
    df.to_csv(PX_FILE)
    msg = f"{len(df.columns)} series diarias vía puerto {puerto} hasta {df.index.max().date()}"
    if faltan:
        msg += f" · sin datos: {', '.join(faltan)}"
    return {"ok": not faltan, "msg": msg}


def actualizar():
    """Foto de las dos cuentas (las que estén abiertas) + precios. Solo lectura."""
    estado = {"paper": foto_cuenta("paper"), "real": foto_cuenta("real")}
    inicio = _inicio_fijo(_load_snaps(),
                          _load(EXEC_FILE, {"paper": {}, "real": {}}))
    desde = (inicio or pd.Timestamp(dt.date.today())) - pd.Timedelta(days=10)
    estado["precios"] = descargar_precios(desde)
    return estado


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo
# ─────────────────────────────────────────────────────────────────────────────

def _inicio(snaps, ex):
    """Primer día en que la paper tiene TODO el NÚCLEO al cierre (la réplica arranca
    ese día). Días con la réplica a medias (una orden sin ejecutar) no cuentan:
    sesgarían la comparación. Operaciones ajenas a la réplica en la misma cuenta
    paper (p.ej. otro sistema de trading) tampoco."""
    req = _core_invertido()
    st, et = snaps.get("paper", []), list(ex.get("paper", {}).values())
    cid_sym = {str(k): _modelo(v["symbol"]) for k, v in _contratos(snaps, ex).items()}
    cands = sorted({pd.Timestamp(e["time"]).normalize() for e in et} |
                   {pd.Timestamp(s["ts"]).normalize() for s in st if s["positions"]})
    for d in cands:
        q = _qty_al_cierre(st, et, d, solo_pasado=True)   # réplica completa CONFIRMADA
        if req and req <= {cid_sym.get(str(c)) for c, v in q.items() if v > 0}:
            return d
    return None


def _inicio_fijo(snaps, ex):
    """Inicio de la comparación. Se calcula una sola vez (la primera vez que la
    réplica está completa) y se guarda: reoptimizar con otro universo o
    rebalancear no lo mueve; esos cambios forman parte de la estrategia y se
    ven en la curva. Para empezar de cero: reiniciar_inicio()."""
    guardado = _load(INICIO_FILE, {}).get("inicio")
    if guardado:
        return pd.Timestamp(guardado)
    d = _inicio(snaps, ex)
    if d is not None:
        _save(INICIO_FILE, {"inicio": str(d.date()),
                            "fijado": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "objetivo": (_objetivo() or {}).get("ts")})
    return d


def reiniciar_inicio():
    """Borra el inicio fijado: se recalculará con el núcleo del objetivo actual."""
    if os.path.exists(INICIO_FILE):
        os.remove(INICIO_FILE)


def _metricas(r, rf_d):
    n = len(r)
    if n == 0:
        return {"n": 0}
    w = np.cumprod(1 + r)
    cum = float(w[-1] - 1)
    peak = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
    mdd = float((w / peak - 1).min())
    out = {"n": n, "cum": cum * 100, "mdd": mdd * 100}
    # Con muy pocos días anualizar dispara las cifras: rentabilidad anualizada,
    # Sharpe y Calmar solo a partir de MIN_DIAS_ANUAL sesiones.
    ann = (1 + cum) ** (ANN / n) - 1
    if n >= 2:
        vol = float(np.std(r, ddof=1) * np.sqrt(ANN))
        out["vol"] = vol * 100
    if n >= MIN_DIAS_ANUAL:
        out["ann"] = ann * 100
        if out.get("vol"):
            out["sharpe"] = float(np.mean(r - rf_d) * ANN / (out["vol"] / 100))
        out["calmar"] = float(ann / abs(mdd)) if mdd < 0 else None
    return out


def _objetivo():
    lr = _load(LAST_FILE, {})
    fr = (lr.get("frontier") or [{}])[0]
    if not fr.get("weights"):
        return None
    return {"ts": lr.get("ts"), "assets": lr.get("assets") or list(fr["weights"]),
            "w": fr["weights"], "ic5": fr.get("ic5", {}), "ic95": fr.get("ic95", {})}


def _pesos_snapshot(snap, core):
    """Pesos (%) de una foto: núcleo normalizado a 100 y satélites sobre el total."""
    if not snap:
        return None
    px_last = None
    if os.path.exists(PX_FILE):
        px_last = pd.read_csv(PX_FILE, index_col=0).ffill().iloc[-1]
    vals = {}
    for p in snap["positions"]:
        v = p.get("valor_eur")
        if v is None or not np.isfinite(v):
            # Sin valor en EUR en la foto: cantidad × último cierre en EUR
            c = str(p["conId"])
            if px_last is None or c not in px_last or pd.isna(px_last[c]):
                continue
            v = p["qty"] * float(px_last[c])
        vals[_modelo(p["symbol"])] = vals.get(_modelo(p["symbol"]), 0.0) + v
    tot = sum(vals.values())
    tot_core = sum(v for k, v in vals.items() if k in core)
    if tot <= 0:
        return None
    return {"core": {k: (vals.get(k, 0.0) / tot_core * 100 if tot_core else 0.0) for k in core},
            "sat": {k: v / tot * 100 for k, v in vals.items() if k not in core},
            "sat_valor": {k: v for k, v in vals.items() if k not in core},
            "core_pct_total": tot_core / tot * 100, "ts": snap["ts"], "source": snap.get("source")}


def calcular():
    snaps = _load_snaps()
    ex = _load(EXEC_FILE, {"paper": {}, "real": {}})
    obj = _objetivo()
    core = obj["assets"] if obj else []
    avisos = []

    estado = {t: ({"ts": snaps[t][-1]["ts"], "account": snaps[t][-1]["account"],
                   "source": snaps[t][-1].get("source"), "n": len(snaps[t][-1]["positions"])}
                  if snaps.get(t) else None) for t in ("paper", "real")}

    # ── Pesos actuales (última foto) vs bandas IC del modelo ───────────────
    wp = _pesos_snapshot(snaps["paper"][-1] if snaps.get("paper") else None, core)
    wr = _pesos_snapshot(snaps["real"][-1] if snaps.get("real") else None, core)
    pesos = []
    for a in core:
        i5, i95 = obj["ic5"].get(a), obj["ic95"].get(a)
        rw = wr["core"].get(a) if wr else None
        pw = wp["core"].get(a) if wp else None
        dentro = lambda v: None if v is None or i5 is None else bool(i5 - 1e-9 <= v <= i95 + 1e-9)
        pesos.append({"ticker": a, "ejec": next((k for k, v in EJEC_A_MODELO.items() if v == a), a),
                      "modelo": obj["w"].get(a), "ic5": i5, "ic95": i95,
                      "paper": pw, "real": rw, "paper_dentro": dentro(pw), "real_dentro": dentro(rw)})
    satelites = ([{"symbol": k, "peso": v, "valor": wr["sat_valor"][k]}
                  for k, v in sorted(wr["sat"].items(), key=lambda kv: -kv[1])] if wr else [])
    if wp and wp["sat"]:
        avisos.append("La cuenta paper tiene posiciones fuera del núcleo (" + ", ".join(sorted(wp["sat"])) +
                      "), probablemente de otro sistema que opera en ella: se excluyen de la réplica.")

    payload = {"ok": True, "objetivo": ({"ts": obj["ts"]} if obj else None), "estado": estado,
               "pesos": pesos, "satelites": satelites,
               "real_core_pct": wr["core_pct_total"] if wr else None,
               "series": None, "metricas": None, "avisos": avisos}

    # ── Serie diaria desde el inicio de la réplica ─────────────────────────
    inicio = _inicio_fijo(snaps, ex)
    payload["inicio"] = str(inicio.date()) if inicio is not None else None
    if inicio is None:
        avisos.append("La paper aún no tiene todos los activos del modelo: la comparación empieza "
                      "el primer día que la réplica esté completa.")
        return payload
    if not os.path.exists(PX_FILE):
        avisos.append("Sin precios diarios todavía: pulsa «Actualizar» con TWS abierto.")
        return payload

    px = pd.read_csv(PX_FILE, index_col=0, parse_dates=True).sort_index().ffill()
    rf_px = px.pop("EUR_RF") if "EUR_RF" in px.columns else None
    px = px[px.index >= inicio - pd.Timedelta(days=10)]
    dias = px.index[px.index >= inicio]
    if len(dias) < 2:
        avisos.append("Todavía no hay al menos dos cierres desde el inicio: la curva aparecerá tras la próxima sesión.")
        return payload

    for t in ("paper", "real"):
        avisos += _avisos_consistencia(t, snaps.get(t, []), list(ex.get(t, {}).values()))

    contratos = _contratos(snaps, ex)
    sym = {str(k): _modelo(v["symbol"]) for k, v in contratos.items()}
    tw = pd.Series({a: obj["w"].get(a, 0) / 100 for a in core})

    def _serie_cartera(tipo):
        st, et = snaps.get(tipo, []), list(ex.get(tipo, {}).values())
        rets, acts, sin_px = [], [], set()
        for d0, d1 in zip(dias[:-1], dias[1:]):
            q = {str(k): v for k, v in _qty_al_cierre(st, et, d0).items()}
            if tipo == "paper":     # la réplica = solo el núcleo (la cuenta paper es compartida)
                q = {c: v for c, v in q.items() if sym.get(c) in core}
            v0 = v1 = 0.0
            for cid, qty in q.items():
                if cid not in px.columns or pd.isna(px.at[d0, cid]) or pd.isna(px.at[d1, cid]):
                    sin_px.add(sym.get(cid, cid)); continue
                v0 += qty * px.at[d0, cid]; v1 += qty * px.at[d1, cid]
            rets.append(v1 / v0 - 1 if v0 > 0 else np.nan)
            # Deriva: active share del núcleo frente al objetivo al cierre de d1
            q1 = {str(k): v for k, v in _qty_al_cierre(st, et, d1).items()}
            vc = pd.Series(dtype=float)
            for cid, qty in q1.items():
                a = sym.get(cid, cid)
                if a in core and cid in px.columns and not pd.isna(px.at[d1, cid]):
                    vc[a] = vc.get(a, 0.0) + qty * px.at[d1, cid]
            acts.append(float(0.5 * (vc.reindex(core).fillna(0) / vc.sum() - tw).abs().sum() * 100)
                        if vc.sum() > 0 else np.nan)
        if sin_px:
            avisos.append(f"{tipo}: sin precio diario para {', '.join(sorted(sin_px))} (excluido del cálculo)")
        return np.array(rets, dtype=float), np.array(acts, dtype=float)

    rp, ap = _serie_cartera("paper")
    rr, ar = _serie_cartera("real")
    if rf_px is not None:
        rf_d = rf_px.reindex(dias).ffill().pct_change().iloc[1:].fillna(0).values
    else:
        rf_d = np.zeros(len(rp))
    ok = ~np.isnan(rp) & ~np.isnan(rr)
    rp, rr, rf_d = np.where(ok, rp, 0.0), np.where(ok, rr, 0.0), rf_d

    def _cum(r):
        return list(np.round((np.cumprod(1 + r) - 1) * 100, 4))

    def _dd(r):
        w = np.cumprod(1 + r)
        pk = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
        return list(np.round((w / pk - 1) * 100, 4))

    fechas = [str(d.date()) for d in dias]
    payload["series"] = {
        "dates": fechas,
        "paper_cum": [0.0] + _cum(rp), "real_cum": [0.0] + _cum(rr),
        "paper_dd": [0.0] + _dd(rp), "real_dd": [0.0] + _dd(rr),
        "paper_as": [None] + [None if np.isnan(x) else round(x, 3) for x in ap],
        "real_as": [None] + [None if np.isnan(x) else round(x, 3) for x in ar],
    }
    mp, mr = _metricas(rp, rf_d), _metricas(rr, rf_d)
    diff = rr - rp
    te = float(np.std(diff, ddof=1) * np.sqrt(ANN) * 100) if len(diff) >= 2 else None
    payload["metricas"] = {"paper": mp, "real": mr, "te": te,
                           "rf_ann": float(np.mean(rf_d) * ANN * 100) if len(rf_d) else None}
    return payload
