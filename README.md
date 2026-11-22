# Documentación — Simulador de memoria de un servidor (versión 2: modelo generalizado)

> **Actualización (correcciones de la docente):** el caso constante se presenta en **forma generalizada** `dA/dt + kA = Q`, con `k` el coeficiente de depuración. La demostración del programa resuelve primero con símbolos y luego con números, calcula el porcentaje, el tráfico máximo y la depuración mínima en forma general, y admite k = 0 y k < 0. El crecimiento del tráfico del escenario lineal pasó a llamarse `g` para no confundirlo con `k`.

**Proyecto:** Modelo de Acumulación y Depuración Dinámica de Memoria en un Servidor (Búfer de Datos)
**Asignatura:** Ecuaciones Diferenciales — Colegio Mayor del Cauca
**Archivo principal:** `simulador_memoria.py` (la versión 1 queda en `version_anterior/`)

---

## 1. Qué es

Una aplicación en Python que implementa la sección **4.2 "Modelado del problema real"** del documento: toma la EDO del proyecto, la resuelve y la muestra con gráficas interactivas.

La versión 2 añade dos cosas:

1. **Modelo generalizado.** El tráfico Q y la limpieza P pueden variar con el tiempo, Q(t) y P(t). El caso del documento (Q y P constantes) queda como uno de cinco escenarios.
2. **Límite de RAM ajustable de 1 GB a 128 GB** (antes era fijo en 8 GB).

| Lo que pide el documento (4.2) | Dónde está en la aplicación |
|---|---|
| Sliders para el tráfico (Q), la limpieza de RAM (k) y la memoria inicial (A₀) | Columna izquierda, "Controles" |
| Gráfica dinámica de A(t) frente a la línea límite de RAM | Gráfica grande del centro |
| Semáforo Verde / Rojo con el % de uso | Columna derecha, arriba |
| Demostración matemática paso a paso según los valores configurados | Columna derecha, abajo |

Extras: selector de escenario, límite de RAM ajustable, campo de direcciones, balance de flujos, comprobación numérica con Runge-Kutta 4, luz amarilla de "en riesgo", instante de colapso, animación y exportación de imágenes.

---

## 2. El modelo matemático

### 2.1 Caso general

La ley de balance de flujo (lo que entra menos lo que el Garbage Collector limpia) da la EDO lineal de primer orden en su **forma estándar**, la misma que escribe el documento:

    dA/dt + P(t)·A = Q(t) ,   A(0) = A₀

| Símbolo | Significado | Valor del documento |
|---|---|---|
| A(t) | Memoria RAM ocupada en el búfer [MB] | incógnita |
| Q(t) | Tasa de entrada de datos [MB/s] | 180 (constante) |
| P(t) | Coeficiente de depuración del Garbage Collector [1/s] | 0.04 (constante) |
| A₀ | Memoria ocupada en t = 0 [MB] | 512 |
| A_máx | Límite físico de la RAM [MB] | 8192 (8 GB); ahora ajustable hasta 131072 (128 GB) |

**Factor integrante y fórmula general:**

    μ(t) = e^( ∫₀ᵗ P(s) ds )

    A(t) = (1/μ(t)) · [ A₀ + ∫₀ᵗ μ(s)·Q(s) ds ]

Esta fórmula vale para **cualquier** Q(t) y P(t). Lo único que cambia de un escenario a otro es si la integral ∫μ·Q se puede resolver a mano.

### 2.2 Forma generalizada del caso constante

Con limpieza constante P(t) = k y tráfico constante Q:

    dA/dt + k·A = Q   ⟹   A(t) = Q/k + (A₀ − Q/k)·e^(−kt)

| Resultado | Fórmula general | Datos del documento |
|---|---|---|
| Equilibrio (k > 0) | Q/k | 4500 MB |
| Porcentaje de uso | Q/(k·A_máx) × 100 | 54.93 % |
| Memoria libre | A_máx − Q/k | 3692 MB |
| Condición de no colapso | Q < k·A_máx | 180 < 327.68 ✓ |
| Tráfico máximo soportado | Q_máx = k·A_máx | 327.68 MB/s |
| Depuración mínima | k_mín = Q/A_máx | 0.022 1/s |

Según el signo de k: si k > 0 la memoria se estabiliza en Q/k; si k = 0, A = A₀ + Q·t crece sin límite; si k < 0, crece de forma exponencial. En los dos últimos casos el servidor siempre colapsa.

El panel de demostración del programa sigue este orden: forma generalizada → valores → factor integrante → solución con símbolos → sustitución → solución particular → límite → porcentaje → condición de no colapso.

### 2.3 Los cinco escenarios

| Escenario | Q(t) | P(t) | ¿Fórmula cerrada? | Solución |
|---|---|---|---|---|
| 1. Constante (documento) | Q | k | Sí | A = Q/k + (A₀ − Q/k)·e^(−kt) |
| 2. Ráfagas periódicas (senoidal) | Q₀ + Qa·sen(ωt), ω = 2π/Tp | k | Sí (por partes dos veces) | A = Q₀/k + a·sen(ωt) + b·cos(ωt) + C·e^(−kt) |
| 3. Pico temporal (pulso) | Q₀, y Q₀ + ΔQ entre t = 40 y t = 40 + d | k | Sí (por tramos) | En cada tramo, la fórmula del documento |
| 4. Tráfico creciente (lineal) | Q₀ + g·t | k | Sí (por partes) | A = α + β·t + C·e^(−kt) |
| 5. Fuga de memoria (GC se degrada) | Q₀ | k·e^(−r·t) | **No** (sin primitiva elemental) | Fórmula general evaluada numéricamente |

**Detalles de cada fórmula cerrada**

- *Senoidal:* a = Qa·k/(k² + ω²), b = −Qa·ω/(k² + ω²), C = A₀ − Q₀/k − b. A largo plazo la memoria oscila alrededor de Q₀/k con amplitud Qa/√(k² + ω²), así que el **techo** es Q₀/k + Qa/√(k² + ω²). Si Qa > Q₀, el programa limita la amplitud a Q₀ para que el tráfico no sea negativo. El periodo se mide en segundos (ráfagas que se repiten cada minuto o pocos minutos). Para horas pico reales, con un ciclo de 24 h, el sistema reacciona mucho más rápido que el tráfico (1/k = 25 s): la memoria sigue al tráfico, A(t) ≈ Q(t)/k, y el máximo es (Q₀ + Qa)/k = 6500 MB (79.35 % de 8 GB).
- *Pulso:* tramo 1 hasta t₁ = 40 s con nivel Q₀/k; tramo 2 durante el pico con nivel (Q₀ + ΔQ)/k; tramo 3 después del pico, de vuelta a Q₀/k. Cada tramo empieza donde terminó el anterior. El máximo se alcanza al final del pico.
- *Lineal:* β = g/k, α = Q₀/k − g/k², C = A₀ − α. Si g > 0, A(t) ≈ α + β·t crece sin límite: **siempre colapsa**, solo cambia cuándo.
- *Fuga:* μ(t) = e^((k/r)(1 − e^(−rt))). Como ∫₀^∞ P dt = k/r es finita, el factor integrante se estanca, P(t) → 0 y dA/dt → Q₀: **siempre colapsa**.

### 2.4 Resultados que calcula el programa

- **Techo de memoria:** el nivel más alto al que llega la memoria (equilibrio Q/k en el caso constante; cresta de la oscilación en el senoidal; valor al final del pico en el pulso; infinito en el lineal y en la fuga).
- **% de uso:** techo / A_máx × 100.
- **Instante de colapso t\*:** en el caso constante se despeja exactamente, t\* = −(1/k)·ln[(A_máx − Q/k)/(A₀ − Q/k)] (vale también para k < 0; si k = 0, t\* = (A_máx − A₀)/Q). En los demás se busca el primer instante en que A(t) ≥ A_máx, ampliando el horizonte si la memoria crece sin límite.

**Regla del semáforo**

| Color | Condición | Significado |
|---|---|---|
| Verde | Techo < 85 % de la RAM | Sistema seguro |
| Amarillo | Techo entre 85 % y 100 % | No colapsa, pero con poco margen |
| Rojo | Alcanza A_máx en algún momento, o crece sin límite | Colapso por Out of Memory |

---

## 3. Cómo se resuelve numéricamente la fórmula general

Para la fuga de memoria (y para dibujar el campo de direcciones en cualquier escenario) el programa evalúa la fórmula general **intervalo por intervalo**. En cada intervalo pequeño [tₙ, tₙ + h] toma P y Q en el punto medio y aplica la solución exacta del documento en ese tramo:

    Aₙ₊₁ = Aₙ·e^(−P·h) + (Q/P)·(1 − e^(−P·h))

Es la fórmula general del factor integrante aplicada a cada tramo. Así se evita calcular e^(P·t) para tiempos grandes, que da números demasiado grandes para la computadora.

Las tres soluciones se comparan entre sí (ver la sección 7): fórmula cerrada, fórmula general numérica y Runge-Kutta 4.

---

## 4. Las gráficas y cómo explicarlas

1. **Memoria RAM ocupada (principal).** Curva azul = solución (fórmula cerrada, o fórmula general numérica en la fuga). Puntos naranjas = RK4. Línea roja = límite de RAM. Línea verde punteada = **isoclina Q(t)/P(t)**: donde está la curva verde, dA/dt = 0. En el caso constante es la recta de equilibrio Q/k. El eje derecho muestra la altura en % de la RAM. Si hay colapso aparece una ✕ roja y la curva sigue en línea discontinua (proyección teórica).
   *Con límites grandes (por ejemplo 128 GB) la gráfica se acerca a la curva para que se vea bien y el límite aparece como un letrero rojo arriba a la derecha.*

2. **Campo de direcciones.** Cada flechita tiene pendiente dA/dt = Q(t) − P(t)·A. Las curvas grises son soluciones con otras memorias iniciales. En el caso constante todas convergen al equilibrio (equilibrio estable). En el senoidal todas terminan en la misma oscilación: el sistema "olvida" la condición inicial porque el término C·e^(−kt) se apaga.

3. **Balance de flujos.** Naranja = entrada Q(t). Verde = salida P(t)·A(t). El área sombreada es dA/dt. Donde se cruzan, la memoria está en un máximo o un mínimo (en el senoidal se ve el retraso entre el tráfico y la limpieza).

---

## 5. Estructura del código

El archivo está dividido en 7 secciones numeradas; cada función y método tiene su comentario explicativo.

| Sección | Contenido | Qué hace |
|---|---|---|
| 1. Datos | constantes del documento, `RAM_GB_MAXIMA = 128`, diccionario `ESCENARIOS` | Valores iniciales y configuración de cada escenario |
| 2. Modelo | clase `Modelo` | Todas las matemáticas, sin gráficas |
| 3. Demostración | `pasos_demostracion`, `pasos_constante`, `num`, `termino` | Arma los pasos del factor integrante con los valores actuales |
| 4. Dibujo | `dibujar_memoria`, `dibujar_campo`, `dibujar_flujos`, `dibujar_semaforo`, `escala_vertical` | Una función por gráfica |
| 5. Aplicación | clase `SimuladorMemoria` | Ventana, selector de escenario, sliders, botones, animación |
| 6. Modos sin ventana | `exportar_graficas`, `verificar` | Exportar PNG y comprobar resultados |
| 7. Entrada | bloque `if __name__ == "__main__"` | Decide qué modo ejecutar |

**Métodos de la clase `Modelo`**

| Método | Qué hace |
|---|---|
| `Q(t)`, `P(t)` | Coeficientes de la EDO según el escenario |
| `derivada(t, A)` | La EDO: Q(t) − P(t)·A |
| `solucion_cerrada(t)` | Fórmula exacta (None en la fuga) |
| `solucion_general(t)` | Fórmula general del factor integrante, evaluada numéricamente |
| `solucion(t)` | Usa la cerrada si existe; si no, la general |
| `runge_kutta4(T, pasos)` | Método numérico de comprobación, programado a mano |
| `techo()` | Nivel máximo de la memoria |
| `tiempo_de_colapso(T)` | Instante en que A(t) = A_máx, o None |
| `diagnostico(T)` | Reúne resultados y decide el color del semáforo |

---

## 6. Instalación y uso

**Requisitos:** Python 3.8 o superior, `numpy` y `matplotlib`.

    pip install numpy matplotlib

**Forma fácil (Windows):** doble clic en `ejecutar.bat`.

| Comando | Qué hace |
|---|---|
| `python simulador_memoria.py` | Abre la aplicación interactiva |
| `python simulador_memoria.py --verificar` | Compara con el documento, compara las tres soluciones en cada escenario y prueba 128 GB |
| `python simulador_memoria.py --exportar` | Guarda las gráficas a 300 dpi en la carpeta `graficas` |

**Controles**

| Control | Rango | Notas |
|---|---|---|
| Escenario | 5 opciones | Al cambiarlo aparecen sus sliders propios |
| Tráfico Q | 0 a 2000 MB/s | Ampliado para poder saturar 128 GB |
| Limpieza k | −0.05 a 0.15 1/s | k ≤ 0 solo en el escenario constante (casos teóricos); en los demás el mínimo es 0.005. En la fuga es el valor inicial |
| Memoria inicial A₀ | 0 al límite de RAM | Su máximo sigue al slider de RAM |
| Límite de RAM | 1 a 128 GB | Por defecto 8 GB |
| Tiempo simulado | 50 a 600 s | |
| Sliders del escenario | según el escenario | Senoidal: Qa y Tp. Pulso: ΔQ y d. Lineal: k. Fuga: r |

**Botones:** *Animar / Detener*, *Datos del documento* (vuelve al caso constante con 8 GB) y *Guardar imagen*.

---

## 7. Pruebas realizadas

Resultado de `--verificar`:

**Datos del documento (8 GB):** equilibrio 4500 MB, C = −3988, 54.93 % de uso, 3692 MB libres, no colapsa. Coincide con el documento.

**Las tres soluciones en cada escenario (valores por defecto, de 0 a 200 s):**

| Escenario | Cerrada vs general | General vs RK4 | Colapso (8 GB) |
|---|---|---|---|
| Constante | 0.00000000005 MB | 0.0000000002 MB | No |
| Ráfagas periódicas | 0.0013 MB | 0.0013 MB | No |
| Pulso | 0.00000000009 MB | 1.25 MB* | No |
| Lineal | 0.0002 MB | 0.0002 MB | t = 172.8 s |
| Fuga | sin fórmula | 0.0003 MB | t = 104.2 s |

\* En el pulso Q da saltos bruscos y RK4 pierde algo de precisión justo en ellos (1.25 MB sobre unos 7300, menos del 0.02 %). La fórmula cerrada y la general coinciden casi exactamente.

**128 GB:** con los datos del documento el techo es el 3.43 % de la RAM (verde). Con Q = 1500 y k = 0.01 el equilibrio sería 150 000 MB y el colapso ocurre en t = 206.66 s, igual al valor despejado a mano.

**Forma generalizada:** tráfico máximo k·A_máx = 327.68 MB/s. Para k = 0.04, k = 0 y k = −0.01 la fórmula coincide con RK4 (diferencias menores a 0.000001 MB) y el instante de colapso despejado coincide con el obtenido recorriendo la solución numérica (k = 0: 42.67 s; k = −0.01: 34.7 s).

**Limitación:** las pruebas se hicieron generando las imágenes sin abrir la ventana. Conviene abrir la ventana interactiva una vez en el computador de la sustentación.

---

## 8. Escenarios sugeridos para la sustentación

| Escenario | Ajustes | Qué se ve |
|---|---|---|
| Documento | Constante, Q = 180, k = 0.04, 8 GB | Verde, equilibrio 4500 MB (54.93 %) |
| Ráfagas periódicas | Senoidal, Qa = 80, Tp = 60 | Verde, oscila con techo 5214 MB (63.64 %) |
| Pico de tráfico | Pulso, ΔQ = 150, d = 40 | Amarillo, pico de 7330 MB (89.48 %) y luego baja |
| Pico que tumba el servidor | Pulso, ΔQ = 300, d = 40 | Rojo, colapso a los 59.5 s |
| Tráfico creciente | Lineal, g = 1 | Rojo, colapso a los 172.8 s |
| Fuga de memoria | Fuga, r = 0.01 | Rojo, colapso a los 104.2 s |
| Servidor grande | Constante, Q = 1500, k = 0.01, 128 GB | Rojo, colapso a los 206.7 s |
| Más RAM no siempre salva | Fuga, r = 0.01, 128 GB | Rojo igual: solo retrasa el colapso a 924.5 s |

El último es un buen cierre: con una fuga de memoria, ampliar la RAM solo compra tiempo; la solución real es arreglar el Garbage Collector.

## 9. Supuestos del modelo

- La limpieza es proporcional a la memoria ocupada (modelo lineal en A).
- Los escenarios usan formas sencillas de Q(t) y P(t); el tráfico real es más irregular, pero la fórmula general sirve igual para cualquier forma.
- No se modelan las pausas del Garbage Collector ni el comportamiento del sistema después del colapso.
