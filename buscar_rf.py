# Busca en IBKR los ETF de tipo libre de riesgo EUR (€STR) para obtener su conId.
# Ejecutar con TWS abierto:  python buscar_rf.py
import sys, multiprocessing
sys.path.insert(0, r"C:\Users\Usuario\Desktop\Optimizador de cartera")

def main():
    from ib_insync import IB
    import mvo_resampling_mep as engine
    cfg = engine.CONFIG
    ib = IB()
    ib.connect(cfg["ibkr_host"], cfg["ibkr_port"], clientId=30)
    print("Conectado. Buscando candidatos de rf (€STR)...\n")
    for q in ["XEON", "CSH2", "ESTR", "EUR Overnight"]:
        print(f"=== Búsqueda: '{q}' ===")
        try:
            matches = ib.reqMatchingSymbols(q) or []
            if not matches:
                print("  (sin resultados)")
            for m in matches:
                c = m.contract
                exch = c.primaryExchange or c.exchange or "?"
                desc = getattr(m, "description", "") or ""
                print(f"  {c.symbol:8} conId={c.conId:>11}  {c.secType:4} {exch:8} {c.currency:3}  {desc}")
        except Exception as e:
            print("  error:", e)
        print()
    ib.disconnect()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
