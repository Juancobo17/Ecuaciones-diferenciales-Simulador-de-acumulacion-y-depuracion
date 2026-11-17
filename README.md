# Documentación — Simulador de memoria de un servidor (EDO de primer orden)

**Proyecto:** Modelo de Acumulación y Depuración Dinámica de Memoria en un Servidor (Búfer de Datos)
**Asignatura:** Ecuaciones Diferenciales — Colegio Mayor del Cauca
**Archivo principal:** `simulador_memoria.py`

---

## 1. Qué es

Una aplicación en Python que implementa la sección **4.2 "Modelado del problema real"** del documento. Toma la EDO del proyecto, la resuelve y la muestra con gráficas interactivas:

| Lo que pide el documento (4.2) | Dónde está en la aplicación |
|---|---|
| Sliders para el tráfico (Q), la limpieza de RAM (P) y la memoria inicial (A₀) | Columna izquierda, "Controles" |
| Gráfica dinámica de A(t) frente a la línea límite de 8 GB | Gráfica grande del centro |
| Semáforo Verde / Rojo con el % de uso | Columna izquierda, "Semáforo de seguridad" |
| Demostración matemática paso a paso según los valores configurados | Panel derecho |

Extras que agregué para sustentar mejor el tema: campo de direcciones, balance de flujos, comprobación numérica con Runge-Kutta 4, luz amarilla de "en riesgo", cálculo del instante de colapso, animación y exportación de imágenes.

---

## 2. El modelo matemático

**Variables**

| Símbolo | Significado | Valor del documento |
|---|---|---|
| A(t) | Memoria RAM ocupada en el búfer [MB] | incógnita |
| Q | Tasa de entrada de datos [MB/s] | 180 |
| P | Coeficiente de depuración del Garbage Collector [1/s] | 0.04 |
| A₀ | Memoria ocupada en t = 0 [MB] | 512 |
| A_máx | Límite físico de la RAM [MB] | 8192 (8 GB) |

**Ecuación (balance de flujo: lo que entra menos lo que sale)**

    dA/dt = Q − P·A ,   A(0) = A₀

**Solución por factor integrante** μ(t) = e^(P·t):

    A(t) = Q/P + (A₀ − Q/P)·e^(−P·t)

Con los datos del documento: **A(t) = 4500 − 3988·e^(−0.04t) MB**.

**Resultados que calcula el programa**

- **Equilibrio:** A_eq = Q/P (es el límite cuando t → ∞, y el punto donde dA/dt = 0).
- **Constante de integración:** C = A₀ − A_eq.
- **Constante de tiempo:** τ = 1/P. Tras unos 5τ el sistema ya está prácticamente en equilibrio.
- **% de uso:** A_eq / A_máx × 100.
- **Instante de colapso** (solo si A_eq > A_máx), despejando A(t\*) = A_máx:
  t\* = −(1/P)·ln[(A_máx − A_eq)/(A₀ − A_eq)]

**Regla del semáforo**

| Color | Condición | Significado |
|---|---|---|
| Verde | A_eq < 85 % de la RAM | Sistema seguro |
| Amarillo | A_eq entre 85 % y 100 % | No colapsa, pero con poco margen |
| Rojo | A_eq > A_máx (o A₀ ≥ A_máx) | Colapso por Out of Memory |

El amarillo no está en el documento (que solo menciona verde y rojo); es un añadido mío. Si no lo quieren, se quita poniendo `UMBRAL_RIESGO = 100.0` al inicio del código.

---

## 3. Las gráficas y cómo explicarlas

1. **Memoria RAM ocupada (principal).** Curva azul = solución analítica. Puntos naranjas = solución numérica RK4. Línea roja discontinua = límite de 8 GB. Línea verde punteada = equilibrio Q/P. El eje derecho muestra lo mismo en % de RAM. Si hay colapso aparece una ✕ roja en el instante exacto y la curva sigue en línea discontinua (proyección teórica: en la realidad el servidor ya se cayó).

2. **Campo de direcciones.** Cada flechita tiene pendiente dA/dt = Q − P·A. Las curvas grises son soluciones con otras condiciones iniciales. *Idea clave para la sustentación:* todas las soluciones, empiecen arriba o abajo, convergen a la línea de equilibrio, así que el equilibrio es **estable**. Por encima de Q/P las flechas bajan (el GC limpia más de lo que entra); por debajo suben.

3. **Balance de flujos.** Línea naranja = entrada Q (constante). Curva verde = salida P·A(t). El área sombreada entre ambas es dA/dt. Cuando las dos curvas se juntan, entra lo mismo que sale y la memoria deja de crecer.

---

## 4. Estructura del código

El archivo está dividido en 7 secciones numeradas, cada función tiene su comentario explicativo.

| Sección | Funciones | Qué hacen |
|---|---|---|
| 1. Datos | constantes `Q_INICIAL`, `P_INICIAL`, `A0_INICIAL`, `A_MAX` | Valores del documento. Se cambian aquí. |
| 2. Modelo | `derivada`, `memoria_equilibrio`, `solucion_analitica`, `runge_kutta4`, `tiempo_de_colapso`, `diagnostico` | Solo matemáticas, sin gráficas. |
| 3. Demostración | `pasos_demostracion`, `num` | Arma los 9 pasos del factor integrante con los valores actuales. |
| 4. Dibujo | `dibujar_memoria`, `dibujar_campo`, `dibujar_flujos`, `dibujar_semaforo` | Una función por gráfica. |
| 5. Aplicación | clase `SimuladorMemoria` | Ventana, sliders, botones, animación. `actualizar()` redibuja todo al mover un slider. |
| 6. Modos sin ventana | `exportar_graficas`, `verificar` | Exportar PNG y comprobar resultados. |
| 7. Entrada | bloque `if __name__ == "__main__"` | Decide qué modo ejecutar. |

**¿Por qué Runge-Kutta 4?** Es un método numérico que resuelve la EDO usando únicamente la derivada, sin conocer la fórmula exacta. Si sus puntos caen sobre la curva analítica, queda comprobado de forma independiente que el despeje a mano es correcto. Está programado a mano (no se usa ninguna librería que lo haga) para poder explicarlo.

---

## 5. Instalación y uso

**Requisitos:** Python 3.8 o superior, `numpy` y `matplotlib`.

    pip install numpy matplotlib

**Forma fácil (Windows):** doble clic en `ejecutar.bat` (instala lo necesario y abre la aplicación).

**Desde la terminal**, dentro de la carpeta del proyecto:

| Comando | Qué hace |
|---|---|
| `python simulador_memoria.py` | Abre la aplicación interactiva |
| `python simulador_memoria.py --verificar` | Compara los resultados con los del documento y con RK4 |
| `python simulador_memoria.py --exportar` | Guarda las gráficas a 300 dpi en la carpeta `graficas` |

**Botones de la aplicación**

- **Animar / Detener:** un punto recorre la curva mostrando t, A(t) y % de RAM. Si hay colapso, se detiene ahí.
- **Datos del documento:** vuelve a Q = 180, P = 0.04, A₀ = 512.
- **Guardar imagen:** guarda la ventana actual en `graficas/`.

---

## 6. Pruebas realizadas

Ejecutando `--verificar` con los datos del documento:

| Magnitud | Programa | Documento |
|---|---|---|
| Memoria de equilibrio | 4500 MB | 4500 MB |
| Constante C | −3988 | −3988 |
| Porcentaje de uso | 54.93 % | 54.93 % |
| Memoria libre | 3692 MB | 3692 MB |
| ¿Colapsa? | No | No |

Diferencia máxima entre la solución analítica y RK4 (paso 0.5 s, 200 s simulados): **0.000002 MB**.

También se probó un caso de colapso (Q = 420, P = 0.03, A₀ = 512): equilibrio teórico 14 000 MB, semáforo en rojo, colapso en t ≈ 28.1 s.

**Limitación:** estas pruebas las corrí en un entorno Linux sin ventana (generando las imágenes). La ventana interactiva en Windows no la pude abrir yo; usa funciones estándar de matplotlib, pero conviene abrirla una vez antes de la sustentación.

---

## 7. Escenarios sugeridos para la sustentación

| Escenario | Q | P | A₀ | Qué se ve |
|---|---|---|---|---|
| Documento | 180 | 0.04 | 512 | Verde, equilibrio 4500 MB (54.93 %) |
| Hora pico | 300 | 0.04 | 512 | Amarillo, equilibrio 7500 MB (91.6 %) |
| Sobrecarga | 400 | 0.04 | 512 | Rojo, colapso a los ≈ 41.5 s |
| GC lento | 180 | 0.02 | 512 | Rojo, equilibrio 9000 MB, colapso a los ≈ 117.6 s |
| Arranque lleno | 180 | 0.04 | 8000 | Verde: la memoria **baja** hasta 4500 MB |

El último es bueno para mostrar que el equilibrio no depende de A₀, solo de Q/P.

## 8. Supuestos del modelo

- Q y P son constantes (el tráfico real varía con el tiempo).
- La limpieza es proporcional a la memoria ocupada (modelo lineal).
- No se modelan las pausas del Garbage Collector ni el comportamiento del sistema después del colapso.
