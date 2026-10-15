"""Espera a que el servidor Flask escuche y abre el navegador.

Lo lanza Iniciar_Dashboard.bat en paralelo al servidor. Un retardo fijo no
sirve: el arranque de app.py (imports de numpy/scipy + mvo_resampling_mep)
tarda mas o menos segun la cache del disco, y si el navegador llega antes
de que el puerto escuche recibe ERR_CONNECTION_REFUSED y no reintenta.
"""

import socket
import time
import webbrowser

HOST = "127.0.0.1"
PORT = 5000
TIMEOUT_S = 90


def esperar_puerto() -> bool:
    limite = time.time() + TIMEOUT_S
    while time.time() < limite:
        try:
            with socket.create_connection((HOST, PORT), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.4)
    return False


if __name__ == "__main__":
    if esperar_puerto():
        webbrowser.open(f"http://{HOST}:{PORT}")
    else:
        print(f"El servidor no respondio en {TIMEOUT_S}s. Abre http://{HOST}:{PORT} a mano.")
