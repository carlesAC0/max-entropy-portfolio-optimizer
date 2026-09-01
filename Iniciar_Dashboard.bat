@echo off
title Dashboard MEP-RQE
cd /d "C:\Users\Usuario\Desktop\Optimizador de cartera"
echo ============================================================
echo   Dashboard MEP-RQE
echo   El navegador se abrira solo en unos segundos.
echo   Para cerrar el servidor: cierra esta ventana o pulsa Ctrl+C.
echo ============================================================
start "" /min cmd /c "timeout /t 4 >nul & start http://127.0.0.1:5000"
"C:\Users\Usuario\anaconda3\envs\mep_tfm\python.exe" app.py
echo.
echo El servidor se ha detenido.
pause
