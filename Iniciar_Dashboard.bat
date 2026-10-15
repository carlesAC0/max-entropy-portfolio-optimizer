@echo off
title Dashboard MEP-RQE
cd /d "%~dp0"
echo ============================================================
echo   Dashboard MEP-RQE
echo   El navegador se abrira solo en unos segundos.
echo   Para cerrar el servidor: cierra esta ventana o pulsa Ctrl+C.
echo ============================================================
rem Abre el navegador solo cuando el puerto 5000 ya escucha
start "" /min "C:\Users\Usuario\anaconda3\envs\mep_tfm\python.exe" _abrir_navegador.py
"C:\Users\Usuario\anaconda3\envs\mep_tfm\python.exe" app.py
echo.
echo El servidor se ha detenido.
pause
