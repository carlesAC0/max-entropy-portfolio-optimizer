# -*- coding: utf-8 -*-
"""
engine_api.py — Adaptador entre la app Flask y el pipeline MVO+Resampling+MEP.

Expone run_optimization(df_prices, tickers, overrides) que ejecuta el pipeline
completo sobre un SUBCONJUNTO de activos y devuelve exactamente el mismo payload
JSON que ya alimenta el dashboard (frontier_data / oos_data / assets), de modo
que el front reutiliza intactos los render de Chart.js.

No reescribe lógica científica: orquesta las funciones del motor.
"""

import logging
import multiprocessing

import numpy as np

import mvo_resampling_mep as engine

log = logging.getLogger("MEP.api")

ANN = {"monthly": 12, "weekly": 52, "daily": 252}


def _build_payload(res, oos, params, cfg) -> dict:
    """
    Payload del dashboard para el motor Opción A (Bajo Traver 2025):
      · frontier  → UN solo punto (MEP_optim) para reutilizar intactos los render
                    de Composición/Estadísticos.
      · surface   → superficie QEP (Sharpe, RQE, concentración) + flags Pareto/optim.
      · mvo_ref   → cartera de referencia MVO (máx. Sharpe).
    """
    assets = params["assets"]
    w   = res["w"]
    rc  = engine.risk_contrib_pct(w, params["sigma"])
    ds_star, rc_star = res["ds_star"], res["rc_star"]

    frontier_data = [{
        "k": 0, "vol": round(float(res["vol"]) * 100, 3),
        "ret": round(float(res["ret"]) * 100, 3), "sr": round(float(res["sr"]), 4),
        "rqe": round(float(res["rqe"]), 5), "rqe_norm": round(float(res["rqe_norm"]), 4),
        "dr": round(float(res["dr"]), 4), "cost_sr": round(float(res["sr"] - res["sr_mvo"]), 4),
        "fb": round(float(res["fallback_pct"]), 1), "ds": round(float(ds_star), 4),
        "weights": {a: round(float(w[i]) * 100, 2) for i, a in enumerate(assets)},
        "ic5":  {a: round(float(res["ic5"][i]) * 100, 2) for i, a in enumerate(assets)},
        "ic95": {a: round(float(res["ic95"][i]) * 100, 2) for i, a in enumerate(assets)},
        "rc":   {a: round(float(rc[i]) * 100, 2) for i, a in enumerate(assets)},
        "max_sr": True, "feasible": True, "viol": [],
    }]

    # Superficie QEP + frente de Pareto (para el gráfico de isocurvas)
    pareto_ids = set(id(p) for p in res.get("pareto", []))
    surface_data = [{
        "sr":    round(float(p["sr"]), 4),
        "rqe":   round(float(p["rqe"]), 5),
        "maxrc": round(float(p["maxrc"]) * 100, 2),
        "ds":    round(float(p["ds"]), 4),
        "rc":    None if p["rc"] is None else round(float(p["rc"]), 3),
        "pareto": id(p) in pareto_ids,
        "opt":   abs(p["ds"] - ds_star) < 1e-9 and (
                 (p["rc"] is None) == (rc_star is None)
                 and (rc_star is None or abs(p["rc"] - rc_star) < 1e-9)),
    } for p in res.get("surface", [])]
    mvo_ref = {"sr": round(float(res["sr_mvo"]), 4),
               "rqe": round(float(engine.rqe(res["w_mvo"], engine.build_distance_matrix(params["rho"]))), 5),
               "maxrc": round(float(engine.risk_contrib_pct(res["w_mvo"], params["sigma"]).max()) * 100, 2),
               "weights": {a: round(float(res["w_mvo"][i]) * 100, 2) for i, a in enumerate(assets)}}

    oos_data = {}
    if oos:
        m = oos["metrics"]; jk = oos["jk_test"]
        nav_mep = (np.exp(np.cumsum(oos["all_r_mep"])) * 100).tolist()
        nav_mvo = (np.exp(np.cumsum(oos["all_r_mvo"])) * 100).tolist()
        nav_ew  = (np.exp(np.cumsum(oos["all_r_ew"]))  * 100).tolist()
        nav_bench = None
        if "all_r_bench" in oos:
            nav_bench = [round(float(v), 3) for v in (np.exp(np.cumsum(oos["all_r_bench"])) * 100)]
        oos_data = {
            "dates":   [str(r["date"].date()) for r in oos["records"]],
            "nav_mep": [round(float(v), 3) for v in nav_mep],
            "nav_mvo": [round(float(v), 3) for v in nav_mvo],
            "nav_ew":  [round(float(v), 3) for v in nav_ew],
            "metrics": {
                k: {"ret": round(float(m[k]["ret_ann"]) * 100, 2),
                    "vol": round(float(m[k]["vol_ann"]) * 100, 2),
                    "sr":  round(float(m[k]["sharpe"]), 3),
                    "mdd": round(float(m[k]["mdd"]) * 100, 2)}
                for k in ("mep", "mvo", "ew")
            },
            "jk": {"z": round(float(jk["z"]), 3), "pval": round(float(jk["pval"]), 4),
                   "sig": bool(jk["significant"]), "sr_mep": round(float(jk["sr_mep"]), 3),
                   "sr_bench": round(float(jk["sr_bench"]), 3)},
            "corr": oos.get("corr"),
            "dates_m": oos.get("dates_m"),
            "nav_bench": nav_bench,
            "bench_label": cfg.get("benchmark_ticker") if nav_bench else None,
            "T_oos": int(oos["T_oos"]),
        }

    # Robustez estadística — PSR / DSR (Bailey & López de Prado 2014).
    # DSR sobre la selección IS (deflacta por N = nº de QEP candidatas) y PSR(0)
    # sobre la track OOS. Todo calculado de la corrida; nada hardcodeado.
    dsr_stats = None
    try:
        ann_f = int(params["ann"])
        surf_sr = [float(p["sr"]) for p in res.get("surface", [])]
        rf_m = float(cfg["rf_annual"]) / ann_f
        r_is_excess = (params["R"] @ res["w"]) - rf_m           # retornos IS del MEP, exceso
        r_oos_excess = None
        if oos and "all_r_mep" in oos and "all_rf" in oos:
            r_oos_excess = np.asarray(oos["all_r_mep"], float) - np.asarray(oos["all_rf"], float)
        dsr_stats = engine.deflated_sharpe_stats(
            surf_sr, float(res["sr"]), r_is_excess, r_oos_excess, ann_f)
    except Exception as e:
        log.warning(f"DSR stats no calculado: {e}")
        dsr_stats = {"error": str(e)}

    n_cpu = cfg.get("n_workers") or multiprocessing.cpu_count()
    return {
        "frontier": frontier_data,
        "surface": surface_data,
        "mvo_ref": mvo_ref,
        "oos": oos_data,
        "dsr": dsr_stats,
        "assets": assets,
        "meta": {
            "T": int(params["T"]), "N": len(assets),
            "N_sim": cfg["N_sim"], "block": cfg["block_size"],
            "w_max": cfg["w_max"], "rf": float(cfg["rf_annual"]), "z_alpha": cfg["z_alpha"],
            "ds_star": round(float(ds_star), 4),
            "rc_star": None if rc_star is None else round(float(rc_star), 3),
            "n_surface": len(surface_data), "n_pareto": len(pareto_ids),
            "oos_train": cfg["oos_train_months"], "oos_rebal": cfg["oos_rebal_freq"],
            "n_cpu": n_cpu,
        },
    }


def run_optimization(df_prices, tickers, overrides=None) -> dict:
    """
    Ejecuta MVO → Resampling → MEP + Walk-Forward OOS sobre el subconjunto `tickers`.
    `df_prices`: DataFrame de precios (índice fecha, columnas = tickers del universo).
    `overrides`: dict opcional para sobrescribir CONFIG (p.ej. {"N_sim": 200}).
    Devuelve el payload JSON-serializable.
    """
    cfg = dict(engine.CONFIG)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    ann = ANN[cfg["freq"]]

    # Tipo libre de riesgo €STR (columna EUR_RF del cache), si está disponible
    rf_series = None
    if "EUR_RF" in df_prices.columns and cfg.get("rf_mode") == "estr":
        _, rf_series, rf_ann = engine.split_rf(df_prices[["EUR_RF"]], cfg, ann)
        cfg["rf_annual"] = rf_ann

    bench_tk = cfg.get("benchmark_ticker")
    cols = [t for t in tickers if t in df_prices.columns and t not in ("EUR_RF", bench_tk)]
    if len(cols) < 2:
        raise ValueError(f"Se requieren ≥2 activos con datos. Recibidos: {cols}")

    df = df_prices[cols].dropna(how="all").copy()

    df_ret, excluded = engine.prepare_returns(df, cfg)
    if df_ret.shape[1] < 2:
        raise ValueError(
            f"Tras limpieza quedan <2 activos. Excluidos por min_obs/ventana: {excluded}. "
            f"Activos válidos: {list(df_ret.columns)}")

    params = engine.estimate_base_params(df_ret, ann)
    # Motor Opción A: superficie QEP+Pareto (decide receta) → resampling → MEP_optim
    res = engine.run_mep_optim(params, cfg)

    # Benchmark (S&P 500, p.ej. CSPX): serie de retornos log alineada a los meses del modelo.
    # No entra en la optimización; sólo se dibuja en los gráficos OOS.
    bench_ret = None
    if bench_tk and bench_tk in df_prices.columns:
        bp = df_prices[bench_tk].astype(float).dropna()
        if len(bp) > 2:
            bench_ret = np.log(bp).diff().reindex(df_ret.index)
            log.info(f"Benchmark {bench_tk}: {bench_ret.notna().sum()} meses alineados")

    T_min = cfg["oos_train_months"] + cfg["oos_rebal_freq"]
    if params["T"] >= T_min:
        oos = engine.run_walk_forward(df_ret, cfg, rf_series=rf_series, bench_ret=bench_ret) or {}
    else:
        log.warning(f"OOS omitido: T={params['T']} < {T_min}")
        oos = {}

    payload = _build_payload(res, oos, params, cfg)
    payload["meta"]["excluded"] = excluded
    payload["meta"]["used"] = list(df_ret.columns)
    return payload
