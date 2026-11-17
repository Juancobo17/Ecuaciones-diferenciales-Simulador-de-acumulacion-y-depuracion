# -*- coding: utf-8 -*-
"""
==============================================================================
 SIMULADOR: Acumulación y Depuración Dinámica de Memoria en un Servidor
            (Búfer de Datos)  -  Proyecto de Ecuaciones Diferenciales
==============================================================================

 ¿QUÉ HACE ESTE PROGRAMA?
 ------------------------
 Modela la memoria RAM ocupada A(t) [MB] en el búfer de un servidor mediante
 una Ecuación Diferencial Ordinaria (EDO) lineal de primer orden:

        dA/dt = R_entrada - R_salida = Q - P·A ,     A(0) = A0

   Q  = tasa de entrada de datos (tráfico)              [MB/s]
   P  = coeficiente de depuración del Garbage Collector [1/s]
   A0 = memoria ocupada en el instante inicial          [MB]

 La EDO se resuelve por el MÉTODO DEL FACTOR INTEGRANTE (igual que en el
 documento del proyecto) y se obtiene la solución exacta:

        A(t) = Q/P + (A0 - Q/P)·e^(-P·t)

 Con los datos del documento (Q=180, P=0.04, A0=512):

        A(t) = 4500 - 3988·e^(-0.04·t)   [MB]

 La aplicación tiene lo que pide la sección 4.2 del documento:
   1. Controles (sliders) para cambiar Q, P y A0 en tiempo real.
   2. Gráfica dinámica de A(t) frente a la línea límite de 8 GB.
   3. Semáforo de seguridad (verde / amarillo / rojo) con el % de uso.
   4. Demostración matemática paso a paso con los valores elegidos.
 Y además, para sustentar mejor el tema:
   5. Campo de direcciones (pendientes) de la EDO.
   6. Gráfica del balance de flujos (entrada vs salida).
   7. Comprobación numérica con Runge-Kutta de orden 4 (RK4).

 CÓMO SE EJECUTA
 ---------------
   python simulador_memoria.py              -> abre la aplicación interactiva
   python simulador_memoria.py --verificar  -> comprueba los números del documento
   python simulador_memoria.py --exportar   -> guarda las gráficas en /graficas

 Requisitos: Python 3.8 o superior, numpy y matplotlib
   pip install numpy matplotlib
==============================================================================
"""

import os
import sys

# Si solo vamos a exportar imágenes no necesitamos abrir ventanas, así que
# elegimos el motor de dibujo "Agg" (sin interfaz). Esto debe hacerse ANTES
# de importar pyplot.
MODO_EXPORTAR = "--exportar" in sys.argv
MODO_VERIFICAR = "--verificar" in sys.argv

import matplotlib

if MODO_EXPORTAR or MODO_VERIFICAR:
    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.widgets import Button, Slider

# ============================================================================
# 1. DATOS DEL PROBLEMA (los mismos del documento del proyecto)
# ============================================================================
Q_INICIAL = 180.0     # Tasa de entrada de datos            [MB/s]
P_INICIAL = 0.04      # Coeficiente de depuración del GC    [1/s]
A0_INICIAL = 512.0    # Memoria ocupada en t = 0            [MB]
A_MAX = 8192.0        # Límite físico de la RAM (8 GB)      [MB]
T_INICIAL = 200.0     # Tiempo que se simula                [s]

# Si el equilibrio supera este porcentaje de la RAM, el semáforo pasa a amarillo
UMBRAL_RIESGO = 85.0

# Paleta de colores (se distinguen bien entre sí, también impresos)
AZUL = "#1f5fbf"
NARANJA = "#e07b00"
VERDE = "#1a9850"
ROJO = "#d62728"
AMARILLO = "#f2b600"
GRIS = "#6b7280"
FONDO = "#f6f8fb"
APAGADO = "#d5d9e0"


# ============================================================================
# 2. MODELO MATEMÁTICO
#    Estas funciones no dibujan nada: solo contienen las matemáticas.
# ============================================================================
def derivada(t, A, Q, P):
    """
    Lado derecho de la EDO:  dA/dt = Q - P·A

    Es la ley de balance de flujo: lo que se acumula por segundo es lo que
    entra (Q) menos lo que el Garbage Collector limpia (P·A). La limpieza es
    proporcional a la memoria ocupada: entre más basura hay, más se elimina.
    (t no se usa porque la ecuación es autónoma, pero se deja por claridad).
    """
    return Q - P * A


def memoria_equilibrio(Q, P):
    """
    Estado estacionario: se alcanza cuando dA/dt = 0.
        0 = Q - P·A   ->   A_eq = Q / P
    Coincide con el límite de A(t) cuando t -> infinito.
    Si P = 0 no hay limpieza y la memoria crece sin límite (infinito).
    """
    if P <= 0:
        return np.inf if Q > 0 else 0.0
    return Q / P


def solucion_analitica(t, Q, P, A0):
    """
    Solución EXACTA obtenida con el factor integrante mu(t) = e^(P·t):

        A(t) = A_eq + C·e^(-P·t) ,   con  A_eq = Q/P   y   C = A0 - A_eq

    Caso especial P = 0 (sin Garbage Collector): dA/dt = Q  ->  A = A0 + Q·t
    """
    t = np.asarray(t, dtype=float)
    if P <= 0:
        return A0 + Q * t
    A_eq = Q / P
    C = A0 - A_eq
    return A_eq + C * np.exp(-P * t)


def runge_kutta4(f, t0, A0, t_final, pasos, Q, P):
    """
    Método numérico de Runge-Kutta de orden 4 (RK4).

    Resuelve la EDO SIN usar la fórmula exacta: avanza pasito a pasito usando
    solo la derivada. Sirve para COMPROBAR que la solución analítica está bien
    (si las dos curvas coinciden, el despeje del documento es correcto).

    En cada paso de tamaño h se calculan 4 pendientes y se promedian:
        k1 = f(t, A)
        k2 = f(t + h/2, A + h·k1/2)
        k3 = f(t + h/2, A + h·k2/2)
        k4 = f(t + h,   A + h·k3)
        A_siguiente = A + (h/6)·(k1 + 2·k2 + 2·k3 + k4)
    """
    h = (t_final - t0) / pasos
    t = np.linspace(t0, t_final, pasos + 1)
    A = np.zeros(pasos + 1)
    A[0] = A0
    for i in range(pasos):
        k1 = f(t[i], A[i], Q, P)
        k2 = f(t[i] + h / 2, A[i] + h * k1 / 2, Q, P)
        k3 = f(t[i] + h / 2, A[i] + h * k2 / 2, Q, P)
        k4 = f(t[i] + h, A[i] + h * k3, Q, P)
        A[i + 1] = A[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, A


def tiempo_de_colapso(Q, P, A0, A_max=A_MAX):
    """
    Instante t* en el que la memoria alcanza el límite físico (error OOM).
    Devuelve None si el servidor NUNCA colapsa.

    Se despeja t de  A(t*) = A_max:
        A_eq + C·e^(-P·t*) = A_max
        e^(-P·t*) = (A_max - A_eq) / (A0 - A_eq)
        t* = -(1/P)·ln[ (A_max - A_eq) / (A0 - A_eq) ]
    """
    if A0 >= A_max:
        return 0.0                      # ya arranca saturado
    if P <= 0:                          # sin limpieza: recta A0 + Q·t
        return (A_max - A0) / Q if Q > 0 else None
    A_eq = Q / P
    if A_eq <= A_max:
        return None                     # la curva se queda debajo del límite
    return -(1.0 / P) * np.log((A_max - A_eq) / (A0 - A_eq))


def diagnostico(Q, P, A0, A_max=A_MAX):
    """
    Reúne todos los resultados importantes del modelo en un diccionario
    y decide el color del semáforo.
    """
    A_eq = memoria_equilibrio(Q, P)
    t_col = tiempo_de_colapso(Q, P, A0, A_max)
    porcentaje = (A_eq / A_max) * 100 if np.isfinite(A_eq) else np.inf

    if t_col is not None:
        estado, color = "COLAPSO (OOM)", ROJO
    elif porcentaje >= UMBRAL_RIESGO:
        estado, color = "EN RIESGO", AMARILLO
    else:
        estado, color = "SISTEMA SEGURO", VERDE

    return {
        "A_eq": A_eq,                                   # memoria de equilibrio
        "C": A0 - A_eq if np.isfinite(A_eq) else None,  # constante de integración
        "tau": 1.0 / P if P > 0 else np.inf,            # constante de tiempo
        "porcentaje": porcentaje,                       # % de RAM en equilibrio
        "libre": A_max - A_eq,                          # MB libres en equilibrio
        "t_colapso": t_col,
        "estado": estado,
        "color": color,
    }


# ============================================================================
# 3. TEXTO DE LA DEMOSTRACIÓN PASO A PASO
# ============================================================================
def num(x, decimales=2):
    """Convierte un número a texto sin ceros sobrantes: 4500.00 -> '4500'."""
    texto = f"{x:.{decimales}f}"
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    return "0" if texto in ("-0", "") else texto


def pasos_demostracion(Q, P, A0, A_max=A_MAX):
    """
    Genera la resolución por FACTOR INTEGRANTE con los valores actuales.
    Devuelve una lista de pares (título del paso, fórmula en notación
    matemática de matplotlib). Sigue el mismo orden del documento.
    """
    q, p, a0 = num(Q), num(P, 4), num(A0)
    d = diagnostico(Q, P, A0, A_max)

    if P <= 0:  # caso sin Garbage Collector
        return [
            ("1. Balance de flujos", rf"$\frac{{dA}}{{dt}} = {q}$   (sin depuración, $P=0$)"),
            ("2. Integración directa", rf"$A(t) = {a0} + {q}\,t$"),
            ("3. Límite", r"$\lim_{t\to\infty} A(t) = \infty$   (colapsa siempre)"),
        ]

    aeq = num(d["A_eq"])
    C = d["C"]
    signo = "-" if C < 0 else "+"
    c_abs = num(abs(C))

    pasos = [
        ("1. Balance de flujos (entrada - salida)",
         rf"$\frac{{dA}}{{dt}} = R_{{entrada}} - R_{{salida}} = {q} - {p}A$"),
        ("2. Forma estándar  dA/dt + P(t)A = Q(t)",
         rf"$\frac{{dA}}{{dt}} + {p}A = {q}, \quad A(0) = {a0}$"),
        ("3. Factor integrante",
         rf"$\mu(t) = e^{{\int {p}\,dt}} = e^{{{p}t}}$"),
        ("4. Multiplicar y agrupar (regla del producto)",
         rf"$\frac{{d}}{{dt}}\left[e^{{{p}t}}A\right] = {q}\,e^{{{p}t}}$"),
        ("5. Integrar ambos lados",
         rf"$e^{{{p}t}}A = \frac{{{q}}}{{{p}}}e^{{{p}t}} + C = {aeq}\,e^{{{p}t}} + C$"),
        ("6. Solución general (dividir entre el factor)",
         rf"$A(t) = {aeq} + C\,e^{{-{p}t}}$"),
        ("7. Condición inicial  A(0) = A0",
         rf"${a0} = {aeq} + C \;\Rightarrow\; C = {num(C)}$"),
        ("8. Solución particular",
         rf"$A(t) = {aeq} {signo} {c_abs}\,e^{{-{p}t}}$  [MB]"),
        ("9. Límite (estado estacionario)",
         rf"$\lim_{{t\to\infty}} A(t) = {aeq}$ MB  $\to$  {num(d['porcentaje'])} % de la RAM"),
    ]
    return pasos


# ============================================================================
# 4. FUNCIONES DE DIBUJO (cada una pinta una gráfica en unos ejes dados)
#    Se usan tanto en la aplicación interactiva como al exportar imágenes.
# ============================================================================
def estilo_ejes(ax):
    """Aspecto común de todas las gráficas: limpio y fácil de leer."""
    ax.set_facecolor("white")
    ax.grid(True, color="#e3e7ee", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(labelsize=8.5)


def dibujar_memoria(ax, Q, P, A0, T, A_max=A_MAX, t_marcador=None):
    """
    GRÁFICA PRINCIPAL: memoria ocupada A(t) frente al límite de 8 GB.
      - Línea azul      : solución analítica (factor integrante).
      - Puntos naranjas : solución numérica RK4 (comprobación).
      - Línea roja      : límite físico de la RAM (A_max).
      - Línea verde     : memoria de equilibrio A_eq = Q/P.
    """
    estilo_ejes(ax)
    d = diagnostico(Q, P, A0, A_max)
    t = np.linspace(0, T, 600)
    A = solucion_analitica(t, Q, P, A0)
    y_tope = A_max * 1.22

    # Zona prohibida: por encima del límite físico el servidor ya se cayó
    ax.axhspan(A_max, y_tope, color=ROJO, alpha=0.10)
    ax.axhline(A_max, color=ROJO, linestyle="--", linewidth=1.8,
               label=f"Límite de RAM = {num(A_max)} MB (8 GB)")

    # Línea de equilibrio (solo si cabe en la gráfica)
    if np.isfinite(d["A_eq"]) and d["A_eq"] <= y_tope:
        ax.axhline(d["A_eq"], color=VERDE, linestyle=":", linewidth=1.8,
                   label=f"Equilibrio Q/P = {num(d['A_eq'])} MB")

    t_col = d["t_colapso"]
    if t_col is not None and t_col <= T:
        # Hasta el colapso la curva es real; después es solo una proyección
        antes = t <= t_col
        ax.plot(t[antes], A[antes], color=AZUL, linewidth=2.8,
                label="A(t) analítica (factor integrante)")
        ax.plot(t[~antes], A[~antes], color=AZUL, linewidth=1.5, linestyle="--",
                alpha=0.55, label="Proyección sin límite físico")
        ax.plot([t_col], [A_max], marker="X", color=ROJO, markersize=13,
                markeredgecolor="white", zorder=6, linestyle="none",
                label=f"Colapso OOM en t = {num(t_col, 1)} s")
        ax.axvline(t_col, color=ROJO, linewidth=1, alpha=0.5)
    else:
        ax.plot(t, A, color=AZUL, linewidth=2.8,
                label="A(t) analítica (factor integrante)")

    # Comprobación numérica con RK4 (se dibujan pocos puntos para que se vea)
    t_rk, A_rk = runge_kutta4(derivada, 0, A0, T, 400, Q, P)
    ax.plot(t_rk[::16], A_rk[::16], "o", color=NARANJA, markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.6,
            label="Numérica RK4 (comprobación)")

    # Punto inicial
    ax.plot([0], [A0], "s", color=AZUL, markersize=7, markeredgecolor="white",
            zorder=6)

    # Marcador móvil de la animación
    if t_marcador is not None:
        A_m = float(solucion_analitica(t_marcador, Q, P, A0))
        ax.plot([t_marcador], [min(A_m, y_tope)], "o", color="black",
                markersize=9, markeredgecolor="white", zorder=7)
        ax.annotate(f"t = {num(t_marcador, 1)} s\nA = {num(A_m, 1)} MB\n"
                    f"{num(A_m / A_max * 100, 1)} % RAM",
                    xy=(t_marcador, min(A_m, y_tope)), xytext=(12, -38),
                    textcoords="offset points", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRIS))

    ax.set_xlim(0, T)
    ax.set_ylim(0, y_tope)
    ax.set_xlabel("Tiempo  t  [s]", fontsize=10)
    ax.set_ylabel("Memoria ocupada  A(t)  [MB]", fontsize=10)
    ax.set_title("Memoria RAM ocupada en el búfer del servidor",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    # Eje derecho: la misma altura expresada en % de la RAM total
    eje_pct = ax.secondary_yaxis(
        "right", functions=(lambda y: y / A_max * 100, lambda p: p * A_max / 100))
    eje_pct.set_ylabel("% de la RAM (8 GB)", fontsize=9)
    eje_pct.tick_params(labelsize=8.5)


def dibujar_campo(ax, Q, P, A0, T, A_max=A_MAX):
    """
    CAMPO DE DIRECCIONES de la EDO.
    En cada punto (t, A) se dibuja una flechita con pendiente dA/dt = Q - P·A.
    Cualquier solución debe "seguir" las flechas. Se ve que TODAS las curvas,
    empiecen donde empiecen, se dirigen a la línea de equilibrio A = Q/P:
    por eso el equilibrio es ESTABLE.
    """
    estilo_ejes(ax)
    y_tope = A_max * 1.22
    tt, aa = np.meshgrid(np.linspace(0, T, 17), np.linspace(0, y_tope, 13))
    pendiente = derivada(tt, aa, Q, P)

    # Las flechas se normalizan (misma longitud) para que solo indiquen dirección.
    # Se trabaja en proporciones de la gráfica para que el ángulo se vea correcto.
    u = np.ones_like(pendiente) / T
    v = pendiente / y_tope
    largo = np.hypot(u, v)
    ax.quiver(tt, aa, u / largo, v / largo, angles="uv", pivot="mid",
              color="#9aa7b8", scale=32, width=0.0035, headwidth=3.2)

    # Familia de soluciones con distintas condiciones iniciales
    t = np.linspace(0, T, 300)
    for a_ini in np.linspace(0, y_tope, 7):
        ax.plot(t, solucion_analitica(t, Q, P, a_ini), color=GRIS,
                linewidth=0.9, alpha=0.65)

    # La solución de NUESTRO problema, resaltada
    ax.plot(t, solucion_analitica(t, Q, P, A0), color=AZUL, linewidth=2.4,
            label=f"Solución con A(0) = {num(A0)}")
    A_eq = memoria_equilibrio(Q, P)
    if np.isfinite(A_eq) and A_eq <= y_tope:
        ax.axhline(A_eq, color=VERDE, linestyle=":", linewidth=1.6,
                   label="Equilibrio (dA/dt = 0)")
    ax.axhline(A_max, color=ROJO, linestyle="--", linewidth=1.3)

    ax.set_xlim(0, T)
    ax.set_ylim(0, y_tope)
    ax.set_xlabel("t  [s]", fontsize=9)
    ax.set_ylabel("A  [MB]", fontsize=9)
    ax.set_title("Campo de direcciones de la EDO", fontsize=10.5,
                 fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95)


def dibujar_flujos(ax, Q, P, A0, T, A_max=A_MAX):
    """
    BALANCE DE FLUJOS: tasa de entrada (constante Q) contra tasa de salida P·A(t).
    La distancia entre ambas curvas es justamente dA/dt. Cuando se tocan,
    entra lo mismo que sale y la memoria deja de cambiar (equilibrio).
    """
    estilo_ejes(ax)
    t = np.linspace(0, T, 400)
    salida = P * solucion_analitica(t, Q, P, A0)
    entrada = np.full_like(t, Q)

    ax.plot(t, entrada, color=NARANJA, linewidth=2.2,
            label=f"Entrada  Q = {num(Q)} MB/s")
    ax.plot(t, salida, color=VERDE, linewidth=2.2,
            label="Salida  P·A(t)  (Garbage Collector)")
    ax.fill_between(t, salida, entrada, color=AZUL, alpha=0.13,
                    label="Diferencia = dA/dt")

    ax.set_xlim(0, T)
    ax.set_ylim(0, max(Q, salida.max(), 1) * 1.3)
    ax.set_xlabel("t  [s]", fontsize=9)
    ax.set_ylabel("Tasa  [MB/s]", fontsize=9)
    ax.set_title("Balance de flujos: entrada vs salida", fontsize=10.5,
                 fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95)


def dibujar_semaforo(ax, d):
    """SEMÁFORO de seguridad: enciende la luz que corresponde al estado."""
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Caja negra del semáforo y sus tres luces
    ax.add_patch(FancyBboxPatch((0.2, 0.9), 5.0, 2.2,
                                boxstyle="round,pad=0.15,rounding_size=0.5",
                                fc="#20242c", ec="none"))
    luces = [(1.2, VERDE), (2.7, AMARILLO), (4.2, ROJO)]
    for x, color in luces:
        encendida = (color == d["color"])
        ax.add_patch(Circle((x, 2.0), 0.62, fc=color if encendida else "#4a4f5a",
                            ec="white" if encendida else "none", lw=1.5))

    ax.text(5.9, 2.55, d["estado"], fontsize=11.5, fontweight="bold",
            color=d["color"] if d["color"] != AMARILLO else "#a87f00", va="center")
    if np.isfinite(d["porcentaje"]):
        detalle = f"Uso en equilibrio: {num(d['porcentaje'])} %"
    else:
        detalle = "La memoria crece sin límite"
    ax.text(5.9, 1.45, detalle, fontsize=9, color="#30343b", va="center")


def texto_resultados(Q, P, A0, d):
    """Resumen numérico que se muestra debajo del semáforo."""
    lineas = []
    if np.isfinite(d["A_eq"]):
        lineas.append(f"Equilibrio  A_eq = Q/P = {num(d['A_eq'])} MB")
        lineas.append(f"Constante   C = A0 - A_eq = {num(d['C'])}")
        lineas.append(f"Const. de tiempo  1/P = {num(d['tau'], 1)} s")
        if d["libre"] >= 0:
            lineas.append(f"Memoria libre: {num(d['libre'])} MB "
                          f"({num(100 - d['porcentaje'])} %)")
        else:
            lineas.append(f"Faltarían {num(-d['libre'])} MB de RAM")
    else:
        lineas.append("Sin depuración (P = 0): no hay equilibrio")
    if d["t_colapso"] is not None:
        lineas.append(f"Colapso OOM en  t = {num(d['t_colapso'], 1)} s")
    else:
        lineas.append("Nunca alcanza el límite de 8 GB")
    return "\n".join(lineas)


# ============================================================================
# 5. APLICACIÓN INTERACTIVA
# ============================================================================
class SimuladorMemoria:
    """
    Ventana principal. Está organizada en tres columnas:
      IZQUIERDA : sliders, semáforo, resultados y botones.
      CENTRO    : gráfica principal + campo de direcciones + balance de flujos.
      DERECHA   : demostración matemática paso a paso.
    Cada vez que se mueve un slider se llama a self.actualizar(), que vuelve a
    calcular el modelo y a dibujar todo con los nuevos valores.
    """

    def __init__(self):
        self.t_marcador = None      # posición del punto animado (None = oculto)
        self.animando = False

        self.fig = plt.figure(figsize=(14.5, 8), facecolor=FONDO)
        try:
            self.fig.canvas.manager.set_window_title(
                "Simulador de memoria del servidor - Ecuaciones Diferenciales")
        except Exception:
            pass

        # ---- Encabezado ----------------------------------------------------
        self.fig.text(0.02, 0.955,
                      "Modelo de Acumulación y Depuración Dinámica de Memoria en un Servidor",
                      fontsize=15, fontweight="bold", color="#17202e")
        self.fig.text(0.02, 0.922,
                      r"EDO lineal de primer orden:  $\frac{dA}{dt} = Q - P\,A$,"
                      r"   $A(0)=A_0$     —  resuelta por factor integrante",
                      fontsize=10.5, color="#3b4658")

        # ---- Columna central: las tres gráficas -----------------------------
        self.ax_mem = self.fig.add_axes([0.305, 0.44, 0.365, 0.42])
        self.ax_campo = self.fig.add_axes([0.305, 0.08, 0.165, 0.25])
        self.ax_flujos = self.fig.add_axes([0.525, 0.08, 0.165, 0.25])

        # ---- Columna derecha: demostración ----------------------------------
        self.ax_pasos = self.fig.add_axes([0.745, 0.04, 0.245, 0.86])

        # ---- Columna izquierda: controles -----------------------------------
        self.fig.text(0.02, 0.868, "CONTROLES", fontsize=10, fontweight="bold",
                      color=GRIS)
        self.s_Q = self._crear_slider(0.80, "Tráfico de entrada  Q  [MB/s]",
                                      0, 600, Q_INICIAL, 1, NARANJA)
        self.s_P = self._crear_slider(0.715, "Limpieza del Garbage Collector  P  [1/s]",
                                      0.005, 0.15, P_INICIAL, 0.001, VERDE)
        self.s_A0 = self._crear_slider(0.63, "Memoria inicial  A0  [MB]",
                                       0, A_MAX, A0_INICIAL, 16, AZUL)
        self.s_T = self._crear_slider(0.545, "Tiempo simulado  [s]",
                                      50, 600, T_INICIAL, 10, GRIS)

        self.fig.text(0.02, 0.475, "SEMÁFORO DE SEGURIDAD", fontsize=10,
                      fontweight="bold", color=GRIS)
        self.ax_semaforo = self.fig.add_axes([0.02, 0.355, 0.24, 0.115])

        self.fig.text(0.02, 0.335, "RESULTADOS", fontsize=10, fontweight="bold",
                      color=GRIS)
        self.txt_resultados = self.fig.text(0.02, 0.315, "", fontsize=9,
                                            va="top", family="monospace",
                                            color="#17202e", linespacing=1.55)

        # ---- Botones --------------------------------------------------------
        self.b_animar = self._crear_boton([0.02, 0.085, 0.075, 0.045], "Animar",
                                          self.animar)
        self.b_reset = self._crear_boton([0.102, 0.085, 0.075, 0.045],
                                         "Datos del\ndocumento", self.restablecer)
        self.b_guardar = self._crear_boton([0.184, 0.085, 0.075, 0.045],
                                           "Guardar\nimagen", self.guardar)
        self.txt_aviso = self.fig.text(0.02, 0.045, "", fontsize=8.5, color=GRIS)

        # Temporizador que mueve el punto durante la animación
        self.temporizador = self.fig.canvas.new_timer(interval=40)
        self.temporizador.add_callback(self._paso_animacion)

        self.actualizar()

    # ------------------------------------------------------------------ util
    def _crear_slider(self, y, etiqueta, minimo, maximo, inicial, paso, color):
        """Crea un slider con su título encima y conecta el evento de cambio."""
        self.fig.text(0.02, y + 0.034, etiqueta, fontsize=9, color="#17202e")
        ax = self.fig.add_axes([0.02, y, 0.19, 0.026])
        slider = Slider(ax, "", minimo, maximo, valinit=inicial, valstep=paso,
                        color=color)
        slider.valtext.set_fontsize(9.5)
        slider.valtext.set_fontweight("bold")
        slider.on_changed(lambda _valor: self.actualizar())
        return slider

    def _crear_boton(self, posicion, texto, funcion):
        ax = self.fig.add_axes(posicion)
        boton = Button(ax, texto, color="white", hovercolor="#dfe8f7")
        boton.label.set_fontsize(8.5)
        boton.on_clicked(funcion)
        return boton

    def valores(self):
        """Lee los valores actuales de los sliders."""
        return (float(self.s_Q.val), float(self.s_P.val),
                float(self.s_A0.val), float(self.s_T.val))

    # ------------------------------------------------------------ redibujado
    def actualizar(self):
        """Recalcula el modelo y vuelve a dibujar TODO con los valores actuales."""
        Q, P, A0, T = self.valores()
        d = diagnostico(Q, P, A0)

        for ax in (self.ax_mem, self.ax_campo, self.ax_flujos):
            ax.clear()
        dibujar_memoria(self.ax_mem, Q, P, A0, T, t_marcador=self.t_marcador)
        dibujar_campo(self.ax_campo, Q, P, A0, T)
        dibujar_flujos(self.ax_flujos, Q, P, A0, T)
        dibujar_semaforo(self.ax_semaforo, d)
        self.txt_resultados.set_text(texto_resultados(Q, P, A0, d))
        self._dibujar_pasos(Q, P, A0, d)
        self.fig.canvas.draw_idle()

    def _dibujar_pasos(self, Q, P, A0, d):
        """Escribe la demostración paso a paso en el panel de la derecha."""
        ax = self.ax_pasos
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc="white", ec="#d5dbe5", lw=1.2))
        ax.text(0.06, 0.955, "Demostración paso a paso", fontsize=11.5,
                fontweight="bold", color="#17202e", va="center")
        ax.text(0.06, 0.925, "Método del factor integrante", fontsize=9,
                color=GRIS, va="center")

        pasos = pasos_demostracion(Q, P, A0)
        y = 0.875
        salto = 0.088
        for titulo, formula in pasos:
            ax.text(0.06, y, titulo, fontsize=8.6, color=AZUL, fontweight="bold",
                    va="center")
            ax.text(0.09, y - 0.040, formula, fontsize=10.5, color="#17202e",
                    va="center")
            y -= salto

        # Conclusión, con el color del semáforo
        if d["t_colapso"] is not None:
            conclusion = (f"CONCLUSIÓN: el búfer se llena en t = "
                          f"{num(d['t_colapso'], 1)} s.\nEl servidor colapsa por OOM.")
        elif d["estado"] == "EN RIESGO":
            conclusion = ("CONCLUSIÓN: no colapsa, pero el equilibrio\n"
                          "queda muy cerca del límite (poco margen).")
        else:
            conclusion = ("CONCLUSIÓN: la memoria se estabiliza por\n"
                          "debajo de 8 GB. El servidor nunca colapsa.")
        color_texto = "#a87f00" if d["color"] == AMARILLO else d["color"]
        ax.text(0.06, 0.055, conclusion, fontsize=9, fontweight="bold",
                color=color_texto, va="center", linespacing=1.4)

    # --------------------------------------------------------------- botones
    def animar(self, _evento=None):
        """Recorre la curva con un punto para ver cómo se llena la memoria."""
        if self.animando:               # si ya estaba animando, se detiene
            self.temporizador.stop()
            self.animando = False
            self.t_marcador = None
            self.b_animar.label.set_text("Animar")
            self.actualizar()
            return
        self.animando = True
        self.t_marcador = 0.0
        self.b_animar.label.set_text("Detener")
        self.temporizador.start()

    def _paso_animacion(self):
        """Avanza el punto un poquito; al llegar al final (o al colapso) para."""
        Q, P, A0, T = self.valores()
        fin = T
        t_col = tiempo_de_colapso(Q, P, A0)
        if t_col is not None:
            fin = min(T, t_col)         # el servidor "muere" al colapsar
        self.t_marcador += T / 120.0
        if self.t_marcador >= fin:
            self.t_marcador = fin
            self.temporizador.stop()
            self.animando = False
            self.b_animar.label.set_text("Animar")
        self.ax_mem.clear()
        dibujar_memoria(self.ax_mem, Q, P, A0, T, t_marcador=self.t_marcador)
        self.fig.canvas.draw_idle()

    def restablecer(self, _evento=None):
        """Vuelve a los datos del documento (Q=180, P=0.04, A0=512)."""
        self.t_marcador = None
        for slider in (self.s_Q, self.s_P, self.s_A0, self.s_T):
            slider.reset()
        self.actualizar()

    def guardar(self, _evento=None):
        """Guarda la ventana completa como imagen PNG de alta resolución."""
        carpeta = carpeta_graficas()
        Q, P, A0, _T = self.valores()
        nombre = f"simulador_Q{num(Q)}_P{num(P, 3)}_A0{num(A0)}.png"
        ruta = os.path.join(carpeta, nombre)
        self.fig.savefig(ruta, dpi=250, facecolor=self.fig.get_facecolor())
        self.txt_aviso.set_text(f"Imagen guardada en: graficas/{nombre}")
        self.fig.canvas.draw_idle()


# ============================================================================
# 6. MODOS SIN VENTANA: exportar gráficas y verificar resultados
# ============================================================================
def carpeta_graficas():
    """Carpeta 'graficas' junto a este archivo (se crea si no existe)."""
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficas")
    os.makedirs(carpeta, exist_ok=True)
    return carpeta


def exportar_graficas():
    """
    Genera imágenes PNG en alta resolución (300 dpi) listas para pegar en el
    documento o en las diapositivas, usando los datos del proyecto.
    """
    carpeta = carpeta_graficas()
    Q, P, A0, T = Q_INICIAL, P_INICIAL, A0_INICIAL, T_INICIAL
    trabajos = [
        ("1_memoria_vs_tiempo.png", dibujar_memoria, (9, 5.2)),
        ("2_campo_de_direcciones.png", dibujar_campo, (7.5, 5)),
        ("3_balance_de_flujos.png", dibujar_flujos, (7.5, 5)),
    ]
    for nombre, funcion, medida in trabajos:
        fig, ax = plt.subplots(figsize=medida)
        funcion(ax, Q, P, A0, T)
        fig.tight_layout()
        fig.savefig(os.path.join(carpeta, nombre), dpi=300)
        plt.close(fig)
        print("  guardada:", nombre)

    # Comparación de escenarios: el del documento contra uno que SÍ colapsa
    fig, ax = plt.subplots(figsize=(9, 5.2))
    estilo_ejes(ax)
    t = np.linspace(0, 300, 600)
    escenarios = [
        (180, 0.04, AZUL, "Documento: Q=180, P=0.04  ->  equilibrio 4500 MB (seguro)"),
        (300, 0.04, AMARILLO, "Hora pico: Q=300, P=0.04  ->  equilibrio 7500 MB (en riesgo)"),
        (400, 0.04, ROJO, "Sobrecarga: Q=400, P=0.04  ->  equilibrio 10000 MB (colapsa)"),
    ]
    for q, p, color, etiqueta in escenarios:
        ax.plot(t, solucion_analitica(t, q, p, A0), color=color, linewidth=2.5,
                label=etiqueta)
    ax.axhline(A_MAX, color=ROJO, linestyle="--", linewidth=1.6,
               label="Límite de RAM = 8192 MB")
    ax.axhspan(A_MAX, A_MAX * 1.3, color=ROJO, alpha=0.10)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, A_MAX * 1.3)
    ax.set_xlabel("Tiempo  t  [s]")
    ax.set_ylabel("Memoria ocupada  A(t)  [MB]")
    ax.set_title("Comparación de escenarios de tráfico", fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "4_comparacion_escenarios.png"), dpi=300)
    plt.close(fig)
    print("  guardada: 4_comparacion_escenarios.png")

    # Captura de la aplicación completa
    app = SimuladorMemoria()
    app.fig.savefig(os.path.join(carpeta, "5_aplicacion_completa.png"), dpi=200,
                    facecolor=app.fig.get_facecolor())
    plt.close(app.fig)
    print("  guardada: 5_aplicacion_completa.png")
    print("\nListo. Las imágenes están en:", carpeta)


def verificar():
    """
    Comprueba que el programa da los MISMOS resultados que el documento y que
    la solución analítica coincide con la numérica (RK4).
    """
    Q, P, A0 = Q_INICIAL, P_INICIAL, A0_INICIAL
    d = diagnostico(Q, P, A0)
    print("=" * 66)
    print(" VERIFICACIÓN CON LOS DATOS DEL DOCUMENTO (Q=180, P=0.04, A0=512)")
    print("=" * 66)
    comprobaciones = [
        ("Memoria de equilibrio Q/P [MB]", d["A_eq"], 4500.0),
        ("Constante de integración C", d["C"], -3988.0),
        ("Porcentaje de uso de RAM [%]", d["porcentaje"], 54.931640625),
        ("Memoria libre [MB]", d["libre"], 3692.0),
        ("A(0) [MB]", float(solucion_analitica(0, Q, P, A0)), 512.0),
    ]
    todo_bien = True
    for nombre, obtenido, esperado in comprobaciones:
        correcto = abs(obtenido - esperado) < 1e-6
        todo_bien &= correcto
        print(f" {nombre:<34} {obtenido:>14.6f}   esperado {esperado:>12.6f}"
              f"   {'OK' if correcto else 'ERROR'}")
    print(f" {'¿Colapsa? (tiempo de colapso)':<34} {str(d['t_colapso']):>14}"
          f"   esperado {'None':>12}   {'OK' if d['t_colapso'] is None else 'ERROR'}")
    todo_bien &= d["t_colapso"] is None

    print("\n Analítica (factor integrante) vs numérica (RK4, h = 0.5 s)")
    print(" " + "-" * 62)
    print(f" {'t [s]':>7} {'A analítica':>16} {'A RK4':>16} {'diferencia':>14}")
    t_rk, A_rk = runge_kutta4(derivada, 0, A0, 200, 400, Q, P)
    for i in range(0, 401, 50):
        exacta = float(solucion_analitica(t_rk[i], Q, P, A0))
        print(f" {t_rk[i]:>7.0f} {exacta:>16.6f} {A_rk[i]:>16.6f}"
              f" {abs(exacta - A_rk[i]):>14.2e}")
    error = np.max(np.abs(solucion_analitica(t_rk, Q, P, A0) - A_rk))
    todo_bien &= error < 1e-3
    print(f"\n Error máximo entre ambas soluciones: {error:.2e} MB")

    # Además: la solución debe cumplir la EDO  dA/dt = Q - P·A
    t = np.linspace(0, 200, 2001)
    A = solucion_analitica(t, Q, P, A0)
    residuo = np.max(np.abs(np.gradient(A, t)[5:-5] - derivada(t, A, Q, P)[5:-5]))
    print(f" Residuo de la EDO (derivada numérica):  {residuo:.2e} MB/s")
    print("\n RESULTADO:", "TODO CORRECTO" if todo_bien else "HAY DIFERENCIAS")
    print("=" * 66)
    return todo_bien


# ============================================================================
# 7. PUNTO DE ENTRADA
# ============================================================================
if __name__ == "__main__":
    if MODO_VERIFICAR:
        sys.exit(0 if verificar() else 1)
    elif MODO_EXPORTAR:
        print("Exportando gráficas en alta resolución...")
        exportar_graficas()
    else:
        aplicacion = SimuladorMemoria()
        try:  # intenta abrir la ventana maximizada (en Windows funciona así)
            aplicacion.fig.canvas.manager.window.state("zoomed")
        except Exception:
            pass
        plt.show()
