# Optimizador cuantitativo de carteras: MVO · Resampling · MEP·RQE

Optimizador de carteras multiactivo que va más allá del máximo Sharpe: **maximiza la
diversificación estructural** mediante la cartera de máxima entropía de Rao (MEP·RQE),
combinada con el resampling de Michaud para ganar estabilidad. Incluye una **aplicación web
interactiva** (Flask + Plotly) conectada a Interactive Brokers, con validación
*walk-forward* fuera de muestra.

Implementación del modelo del **Máster en Finanzas Cuantitativas (MFIA)**, que extiende a un
universo multiactivo el marco de máxima entropía de **Bajo Traver (2025, _Journal of Asset
Management_)**. Metodología y resultados completos, en el paper; este repositorio es el
software (motor + aplicación).

**Stack:** Python · NumPy · SciPy · pandas · Flask · Plotly · Chart.js · ib_insync ·
multiprocessing.

---

## El aplicativo

Dashboard interactivo para seleccionar el universo, ejecutar el modelo y explorar los
resultados. La vista principal muestra el **frente de Pareto 3D** y la **superficie de
carteras cuasi-eficientes (QEP)**, con el óptimo (`MEP_optim`) y el MVO marcados.

![Dashboard: frente de Pareto y superficie QEP](docs/img/dashboard_is.png)

![Dashboard: composición de la cartera](docs/img/dashboard_composicion.png)

---

## Cómo funciona (resumen)

- **Punto de partida.** La optimización media-varianza (MVO) es inestable: la covarianza
  suele estar mal condicionada y la solución es muy sensible al error de estimación, hasta
  el punto de que fuera de muestra la cartera 1/N la bate a menudo (DeMiguel et al. 2009).
- **Objetivo de diversificación.** En lugar del máximo Sharpe, se maximiza la **entropía
  cuadrática de Rao** `H_D(w) = ½·wᵀ·D̃·w`, donde `D̃` es una distancia entre activos en dos
  pasos (Gower + «distancia de distancias» de López de Prado): mide la similitud de cada
  activo con **todo** el universo, no solo por pares.
- **Selección (QEP + Pareto).** Se barre una rejilla de tolerancias, una **banda de Sharpe
  δS** (calibrada por el error del Sharpe, Lo 2002) y un **cap de contribución al riesgo**
  por activo, generando una **superficie de carteras cuasi-eficientes (QEP)**. Su **frente
  de Pareto** (diversificación · Sharpe · concentración) fija la de **máxima diversificación
  dentro de la banda**.
- **Estabilidad.** Un **Stationary Block Bootstrap** (Politis & Romano 1994) preserva
  autocorrelación y colas gruesas de las series y agrega la solución por mediana de pesos
  con intervalos de confianza.
- **Validación.** *Walk-forward* fuera de muestra y contrastes de robustez: Jobson-Korkie /
  Memmel (igualdad de Sharpe), PSR y DSR (Bailey & López de Prado) frente a muestra corta,
  no-normalidad y sesgo de selección.

## Resultado

En validación *walk-forward* fuera de muestra, la cartera MEP·RQE **mantiene el ratio de
Sharpe** de la optimización media-varianza **reduciendo la volatilidad y el máximo
drawdown**, y supera a la cartera equiponderada (1/N). En otras palabras: mayor
diversificación y un perfil de riesgo más contenido **a igualdad de eficiencia**, sin coste
de rentabilidad ajustada por riesgo estadísticamente detectable.

---

## Arquitectura

Diseño **por capas con dependencias unidireccionales**: la lógica científica (motor) queda
separada de los datos y la presentación, y tanto la app como la ejecución por lotes invocan
el mismo pipeline (una única fuente de verdad, resultados reproducibles).

```mermaid
flowchart TD
    UI["Presentación · dashboard web<br/>HTML · Plotly · Chart.js<br/><i>templates/dashboard_app.html</i>"]
    WEB["Web · servidor Flask (API REST)<br/><i>app.py</i>"]
    AD["Adaptador + Capa de datos<br/>universo · cache · IBKR · XLS<br/><i>engine_api.py · data_cache.py</i>"]
    ENG["Motor de cálculo · funciones puras<br/>MVO · QEP/Pareto · entropía · bootstrap · OOS<br/><i>mvo_resampling_mep.py</i>"]
    UI --> WEB --> AD --> ENG
```

| Módulo | Rol |
|---|---|
| `mvo_resampling_mep.py` | Motor: pipeline científico completo (funciones puras) |
| `engine_api.py` | Adaptador app ↔ motor; devuelve el payload JSON del dashboard |
| `data_cache.py` | Universo, cache de precios (TTL 1 día), alta IBKR e ingesta XLS/CSV |
| `app.py` | Servidor Flask (API REST) que sirve el dashboard |
| `templates/dashboard_app.html` | Dashboard interactivo (Plotly + Chart.js) |

---

## Uso

Requiere Python 3.11+. Para descargar precios en vivo, la estación de **Interactive Brokers
(TWS o IB Gateway)** debe estar abierta con la API habilitada; si hay cache del día, corre
sin conexión.

```bash
pip install -r requirements.txt
python app.py            # dashboard en http://127.0.0.1:5000
```

---

## Referencias

Métodos descritos en la literatura académica (obra de sus autores; aquí se implementan y se
citan): Bajo Traver (2025); Markowitz (1952); Michaud (1989, 1998); Rao (1982); Gower
(1966); López de Prado (2016, 2019); Lo (2002); Politis & Romano (1994); Jagannathan & Ma
(2003); DeMiguel, Garlappi & Uppal (2009); Maillard, Roncalli & Teïletche (2010); Bailey &
López de Prado (2012, 2014); Jobson & Korkie (1981); Memmel (2003).

---

## Licencia

© 2026 **Carles Aznar Carrique (MFIA)**. Todos los derechos reservados. Ver
[`LICENSE`](LICENSE). Repositorio publicado **solo para consulta y referencia** (TFM y
portfolio profesional): no se concede permiso para usar, copiar, modificar ni redistribuir
el código sin autorización escrita del autor. El copyright cubre el código fuente original;
los métodos teóricos pertenecen a sus respectivos autores.

---

> **Aviso.** Herramienta de investigación basada en modelos estadísticos. **No constituye
> asesoramiento financiero ni recomendación de inversión**, ni una oferta o solicitud para
> comprar o vender ningún instrumento financiero. Las rentabilidades pasadas o simuladas no
> garantizan resultados futuros; toda inversión conlleva riesgo, incluida la posible pérdida
> del capital. El uso de esta herramienta es responsabilidad exclusiva del usuario.
