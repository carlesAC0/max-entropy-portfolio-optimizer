# -*- coding: utf-8 -*-
"""
app.py — Servidor Flask del dashboard interactivo del Motor MEP·RQE.

Arranque:
    conda activate mep_tfm
    cd "C:\\Users\\Usuario\\Desktop\\Optimizador de cartera"
    python app.py
    → abre http://127.0.0.1:5000

Endpoints
    GET  /                 dashboard (HTML estático, datos por fetch)
    GET  /api/state        universo + estado del cache + config
    POST /api/optimize     {tickers:[...], config:{N_sim,...}} → payload del dashboard
    POST /api/refresh      fuerza descarga IBKR del universo completo
    GET  /api/search?q=    busca símbolos en IBKR (TWS abierto)
    POST /api/add_ticker   {symbol,conId,exchange,currency,clase,core}
    POST /api/remove       {symbol}  (solo no-núcleo)

Nota Windows: el pipeline usa multiprocessing; se desactiva el reloader de Flask
para no duplicar el árbol de procesos.
"""

import os
import re
import sys
import json
import logging
import multiprocessing
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime

# Consola Windows en UTF-8: el motor loguea μ, σ, →, ─, ✗ (cp1252 los rompe).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from flask import Flask, jsonify, request, send_from_directory

import mvo_resampling_mep as engine
import data_cache as dc
import engine_api as eapi

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("MEP.app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAST_FILE = os.path.join(BASE_DIR, "cache", "last_result.json")
INFORMES_DIR = os.path.join(BASE_DIR, "informes")
app = Flask(__name__)


def _find_browser():
    """Localiza un navegador Chromium para imprimir a PDF (Edge viene con Windows)."""
    candidates = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "templates"), "dashboard_app.html")


@app.route("/api/last")
def api_last():
    """Último resultado de optimización guardado (para recargar sin reejecutar)."""
    if os.path.exists(LAST_FILE):
        try:
            with open(LAST_FILE, "r", encoding="utf-8") as fh:
                return jsonify(json.load(fh))
        except Exception:
            pass
    return jsonify({})


@app.route("/api/state")
def api_state():
    u = dict(dc.load_universe())
    # Fondos XLS: entran al mismo dict de universo con flag ext (el front los
    # pinta como grupo aparte); no tienen conId ni exchange — no van a IBKR.
    df_ext, ext_meta = dc.load_external()
    if df_ext is not None:
        for name in df_ext.columns:
            mi = ext_meta.get(name, {})
            u[name] = {"clase": f"XLS · {mi.get('file', '?')} · {mi.get('kind', '')} · "
                                f"{mi.get('n_meses', '?')}m desde {mi.get('start', '?')}",
                       "core": False, "default": False, "ext": True}
    meta = dc.read_meta()
    return jsonify({
        "universe": u,
        "cache": {
            "fresh":   dc.cache_is_fresh(),
            "date":    meta.get("date"),
            "ts":      meta.get("ts"),
            "tickers": meta.get("tickers", []),
            "rows":    meta.get("rows"),
            "start":   meta.get("start"),
            "end":     meta.get("end"),
        },
        "config": {
            "N_sim": engine.CONFIG["N_sim"], "K": engine.CONFIG["K"],
            "block_size": engine.CONFIG["block_size"], "w_max": engine.CONFIG["w_max"],
            "rc_k": engine.CONFIG["rc_k"], "rf_annual": engine.CONFIG["rf_annual"],
            "ibkr_port": engine.CONFIG["ibkr_port"],
        },
    })


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    data = request.get_json(force=True) or {}
    tickers = data.get("tickers") or []
    overrides = data.get("config") or {}
    if len(tickers) < 2:
        return jsonify({"error": "Selecciona al menos 2 activos."}), 400
    # Fuente de datos: si TODOS los tickers son fondos XLS, no se toca IBKR
    # (el cache, si existe, aporta €STR/benchmark); si hay tickers del universo
    # IBKR, comportamiento de siempre (cache fresco o descarga).
    df_ext, _ = dc.load_external()
    ext_cols = set(df_ext.columns) if df_ext is not None else set()
    if all(t in ext_cols for t in tickers):
        df, info = dc.get_prices_cached_only()
        info = info or {"source": "externos", "stale": False, "date": None}
    else:
        try:
            df, info = dc.get_prices(allow_stale=True)
        except Exception as e:
            return jsonify({"error": f"Sin datos en cache y no se pudo conectar a IBKR "
                                     f"(¿TWS abierto en {engine.CONFIG['ibkr_port']}?). Detalle: {e}"}), 503
    if df_ext is not None:
        df = dc.merge_external(df, df_ext)
    try:
        payload = eapi.run_optimization(df, tickers, overrides)
    except Exception as e:
        log.exception("Error en optimización")
        return jsonify({"error": str(e)}), 400
    payload["data_info"] = info
    payload["tickers"] = tickers
    payload["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        with open(LAST_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception as e:
        log.warning(f"No se pudo guardar last_result: {e}")
    return jsonify(payload)


@app.route("/api/upload_xls", methods=["POST"])
def api_upload_xls():
    """Carga un XLS/CSV con fondos (1ª col = fechas, una col por fondo)."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No llegó ningún archivo."}), 400
    try:
        info = dc.add_external_file(f.read(), f.filename)
    except Exception as e:
        log.exception("Error interpretando XLS")
        return jsonify({"error": f"No se pudo interpretar '{f.filename}': {e}"}), 400
    return jsonify({"ok": True, "funds": info})


@app.route("/api/remove_external", methods=["POST"])
def api_remove_external():
    d = request.get_json(force=True) or {}
    name = d.get("name")
    if not name:
        return jsonify({"error": "Falta 'name'."}), 400
    try:
        dc.remove_external(name)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True})


@app.route("/api/export_pdf", methods=["POST"])
def api_export_pdf():
    """Exporta el último resultado a informes/ como PDF (Edge headless) + HTML
    autocontenido (payload incrustado, sin servidor). El HTML de impresión se
    genera como fichero para que Edge no tenga que pedirle nada a este Flask
    single-thread (se bloquearían mutuamente)."""
    if not os.path.exists(LAST_FILE):
        return jsonify({"error": "No hay resultado que exportar. Ejecuta una optimización primero."}), 400
    with open(LAST_FILE, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    with open(os.path.join(BASE_DIR, "templates", "dashboard_app.html"), "r", encoding="utf-8") as fh:
        tpl = fh.read()
    # Payload incrustado ANTES de las librerías: el script principal lo detecta
    # (window.__PRINT__) y arranca en modo impresión. Se escapa </ para no
    # cerrar el <script> si algún texto del payload lo contuviera.
    inject = "<script>window.__PRINT__=" + json.dumps(payload).replace("</", "<\\/") + ";</script>"
    html = tpl.replace("<!--PRINT_PAYLOAD-->", inject)

    os.makedirs(INFORMES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tks = "-".join(payload.get("tickers", []))[:80]
    # Ablation "sin límite de peso" (w_max=100%) → el nombre del informe lo hace explícito
    w_max = (payload.get("meta") or {}).get("w_max")
    tag = "_sinWmax" if (w_max is not None and w_max >= 0.999) else ""
    base = re.sub(r"[^\w\-+.]", "", f"MEP_{ts}{tag}_{tks}") or f"MEP_{ts}{tag}"
    html_path = os.path.join(INFORMES_DIR, base + ".html")
    pdf_path = os.path.join(INFORMES_DIR, base + ".pdf")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    browser = _find_browser()
    if not browser:
        return jsonify({"error": "No se encontró Edge ni Chrome para generar el PDF. "
                                 "Se guardó igualmente el HTML: " + html_path,
                        "html": os.path.relpath(html_path, BASE_DIR)}), 500
    # user-data-dir temporal: evita el bloqueo del perfil si Edge está abierto.
    udd = os.path.join(BASE_DIR, "cache", "edge_pdf_profile")
    cmd = [browser, "--headless", f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer",
           "--virtual-time-budget=20000", "--window-size=1400,1000",
           f"--user-data-dir={udd}", "--no-first-run", "--disable-extensions",
           Path(html_path).as_uri()]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Edge tardó demasiado generando el PDF. "
                                 "Se guardó el HTML: " + html_path,
                        "html": os.path.relpath(html_path, BASE_DIR)}), 500
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
        err = (proc.stderr or b"").decode("utf-8", "replace")[-400:]
        log.error(f"print-to-pdf falló (rc={proc.returncode}): {err}")
        return jsonify({"error": "El navegador no generó el PDF (¿sin internet para los CDN?). "
                                 "Se guardó el HTML: " + html_path,
                        "html": os.path.relpath(html_path, BASE_DIR)}), 500
    log.info(f"Informe exportado: {pdf_path}")
    return jsonify({"ok": True,
                    "pdf": os.path.relpath(pdf_path, BASE_DIR),
                    "html": os.path.relpath(html_path, BASE_DIR)})


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    try:
        df, info = dc.get_prices(force=True, allow_stale=False)
    except Exception as e:
        return jsonify({"error": f"No se pudo refrescar desde IBKR "
                                 f"(¿TWS abierto en {engine.CONFIG['ibkr_port']}?). Detalle: {e}"}), 503
    return jsonify({"ok": True, "tickers": list(df.columns), "rows": len(df),
                    "info": info})


@app.route("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"results": []})
    try:
        return jsonify({"results": dc.resolve_symbol(q)})
    except Exception as e:
        return jsonify({"error": f"Búsqueda IBKR fallida (¿TWS abierto?): {e}"}), 503


@app.route("/api/add_ticker", methods=["POST"])
def api_add_ticker():
    d = request.get_json(force=True) or {}
    req = ("symbol", "conId", "exchange", "currency")
    if not all(d.get(k) for k in req):
        return jsonify({"error": f"Faltan campos: {req}"}), 400
    try:
        info = dc.add_ticker(d["symbol"], d["conId"], d["exchange"], d["currency"],
                             d.get("clase", ""), d.get("core", False))
    except Exception as e:
        return jsonify({"error": f"Alta fallida (¿TWS abierto?): {e}"}), 503
    return jsonify({"ok": True, "symbol": d["symbol"].upper(), "info": info})


@app.route("/api/remove", methods=["POST"])
def api_remove():
    d = request.get_json(force=True) or {}
    sym = d.get("symbol")
    if not sym:
        return jsonify({"error": "Falta 'symbol'."}), 400
    try:
        dc.remove_ticker(sym)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True})


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # ib_insync necesita un event loop en el hilo que hace la llamada a IBKR.
    # Ejecutamos Flask en un único hilo para que las peticiones usen este loop.
    asyncio.set_event_loop(asyncio.new_event_loop())

    log.info("=" * 60)
    log.info("Dashboard MEP·RQE interactivo  →  http://127.0.0.1:5000")
    log.info(f"  Núcleos: {multiprocessing.cpu_count()}  ·  IBKR port: {engine.CONFIG['ibkr_port']}")
    log.info("=" * 60)

    # use_reloader=False: imprescindible con multiprocessing en Windows.
    # threaded=False: evita que IBKR se ejecute en Thread-X sin event loop.
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=False,
    )