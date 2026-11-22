# -*- coding: utf-8 -*-
"""
==============================================================================
 SIMULADOR: Acumulación y Depuración Dinámica de Memoria en un Servidor
            (Búfer de Datos)  -  Proyecto de Ecuaciones Diferenciales
            VERSIÓN 2: modelo GENERALIZADO y RAM de hasta 128 GB
==============================================================================

 ¿QUÉ HACE ESTE PROGRAMA?
 ------------------------
 Modela la memoria RAM ocupada A(t) [MB] en el búfer de un servidor con la
 EDO lineal de primer orden en su FORMA GENERAL (la misma forma estándar que
 aparece en el documento del proyecto):

        dA/dt + P(t)·A = Q(t) ,     A(0) = A0

   Q(t) = tasa de entrada de datos (tráfico)               [MB/s]
   P(t) = coeficiente de depuración del Garbage Collector  [1/s]
   A0   = memoria ocupada en el instante inicial           [MB]

 MÉTODO DEL FACTOR INTEGRANTE (caso general):

        mu(t) = e^( ∫0^t P(s) ds )

        A(t)  = (1 / mu(t)) · [ A0 + ∫0^t mu(s)·Q(s) ds ]

 Con limpieza constante P(t) = k y tráfico constante Q se obtiene la FORMA
 GENERALIZADA del documento y su solución:

        dA/dt + k·A = Q   ->   A(t) = Q/k + (A0 - Q/k)·e^(-k·t)

 Con los datos del documento (Q = 180, k = 0.04, A0 = 512):

        A(t) = 4500 - 3988·e^(-0.04·t)   [MB]

 Esta versión permite elegir CINCO escenarios:
   1. Constante (el del documento) ......... Q y k fijos (k puede ser < 0)
   2. Ráfagas periódicas (senoidal) ........ Q(t) = Q0 + Qa·sen(2πt/Tp)
   3. Pico temporal (pulso) ................ Q(t) sube ΔQ durante d segundos
   4. Tráfico creciente (lineal) ........... Q(t) = Q0 + g·t
   5. Fuga de memoria (GC se degrada) ...... P(t) = k·e^(-r·t)
 En los cuatro primeros la integral tiene solución exacta (fórmula cerrada);
 en el quinto no existe primitiva elemental y la fórmula general se evalúa
 numéricamente. En todos se compara con Runge-Kutta 4 como comprobación.

 Además, el límite de RAM es ajustable de 1 GB a 128 GB.

 CÓMO SE EJECUTA
 ---------------
   python simulador_memoria.py              -> abre la aplicación interactiva
   python simulador_memoria.py --verificar  -> comprueba los resultados
   python simulador_memoria.py --exportar   -> guarda las gráficas en /graficas

 Requisitos: Python 3.8 o superior, numpy y matplotlib
   pip install numpy matplotlib
==============================================================================
"""

import os
import sys

# Si solo vamos a exportar o verificar no hace falta abrir ventanas: se usa el
# motor de dibujo "Agg" (sin interfaz). Debe elegirse ANTES de importar pyplot.
MODO_EXPORTAR = "--exportar" in sys.argv
MODO_VERIFICAR = "--verificar" in sys.argv

import matplotlib

if MODO_EXPORTAR or MODO_VERIFICAR:
    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.widgets import Button, RadioButtons, Slider

# ============================================================================
# 1. DATOS DEL PROBLEMA (los del documento) Y CONFIGURACIÓN
# ============================================================================
Q_INICIAL = 180.0       # Tasa de entrada de datos            [MB/s]
P_INICIAL = 0.04        # Coeficiente de depuración del GC    [1/s]
A0_INICIAL = 512.0      # Memoria ocupada en t = 0            [MB]
T_INICIAL = 200.0       # Tiempo que se simula                [s]

MB_POR_GB = 1024
RAM_GB_INICIAL = 8      # Límite de RAM del documento: 8 GB
RAM_GB_MAXIMA = 128     # El slider permite llegar hasta 128 GB
A_MAX = RAM_GB_INICIAL * MB_POR_GB      # 8192 MB

Q_MAXIMO_SLIDER = 2000  # Con 128 GB hace falta más tráfico para saturar [MB/s]
K_MINIMO_SLIDER = -0.05  # k < 0 solo tiene sentido en el escenario constante
K_MINIMO_OTROS = 0.005   # mínimo de k en los demás escenarios
INICIO_PULSO = 40.0     # Instante en que empieza el pico temporal [s]
UMBRAL_RIESGO = 85.0    # Por encima de este % de RAM el semáforo es amarillo

# Paleta de colores
AZUL = "#1f5fbf"
NARANJA = "#e07b00"
VERDE = "#1a9850"
ROJO = "#d62728"
AMARILLO = "#f2b600"
GRIS = "#6b7280"
FONDO = "#f6f8fb"

# ----------------------------------------------------------------------------
# Escenarios disponibles. Para cada uno se indica su nombre y la configuración
# de sus sliders extra: (etiqueta, mínimo, máximo, inicial, paso, formato).
# None significa que ese escenario no usa ese slider.
# ----------------------------------------------------------------------------
ESCENARIOS = {
    "constante": {
        "nombre": "Constante (documento)",
        "param1": None,
        "param2": None,
    },
    "senoidal": {
        "nombre": "Ráfagas periódicas (senoidal)",
        "param1": ("Amplitud de la ráfaga  Qa  [MB/s]", 0, 600, 80, 5, "%.0f"),
        "param2": ("Periodo entre ráfagas  Tp  [s]", 20, 300, 60, 5, "%.0f"),
    },
    "pulso": {
        "nombre": "Pico temporal (pulso)",
        "param1": ("Aumento del tráfico  ΔQ  [MB/s]", 0, 1500, 150, 10, "%.0f"),
        "param2": (f"Duración del pico  d  [s]   (inicia en t = {INICIO_PULSO:.0f} s)",
                   5, 150, 40, 1, "%.0f"),
    },
    "lineal": {
        "nombre": "Tráfico creciente (lineal)",
        "param1": ("Crecimiento del tráfico  g  [MB/s²]", 0, 5, 1, 0.05, "%.2f"),
        "param2": None,
    },
    "fuga": {
        "nombre": "Fuga de memoria (GC se degrada)",
        "param1": ("Degradación del GC  r  [1/s]", 0, 0.05, 0.01, 0.0005, "%.4f"),
        "param2": None,
    },
}
ORDEN_ESCENARIOS = list(ESCENARIOS)


def valores_por_defecto(clave):
    """Devuelve (p1, p2) iniciales del escenario (0 si no los usa)."""
    e = ESCENARIOS[clave]
    p1 = e["param1"][3] if e["param1"] else 0.0
    p2 = e["param2"][3] if e["param2"] else 60.0
    return float(p1), float(p2)


# ============================================================================
# 2. MODELO MATEMÁTICO GENERAL
#    La clase Modelo reúne TODO lo matemático de un escenario. No dibuja nada.
# ============================================================================
class Modelo:
    """
    Representa el problema de valor inicial

        dA/dt + P(t)·A = Q(t) ,   A(0) = A0

    para un escenario concreto. Sus métodos calculan Q(t), P(t), la solución
    (exacta o numérica), el techo de memoria y el instante de colapso.

    Parámetros:
      clave  : nombre del escenario ("constante", "senoidal", ...)
      Q0, P0 : tráfico base Q [MB/s] y limpieza base k [1/s]
               (en el código se llama P0 al coeficiente k de la forma generalizada)
      A0     : memoria inicial [MB]
      A_max  : límite físico de la RAM [MB]
      p1, p2 : parámetros propios del escenario (amplitud, periodo, etc.)
    """

    def __init__(self, clave, Q0, P0, A0, A_max, p1=0.0, p2=60.0):
        self.clave_elegida = clave          # lo que eligió el usuario
        self.clave = clave                  # la que se usa en los cálculos
        self.Q0, self.P0, self.A0 = float(Q0), float(P0), float(A0)
        self.A_max = float(A_max)
        self.p1, self.p2 = float(p1), float(p2)
        self.aviso = ""

        # En el escenario constante k puede ser negativo o cero (casos teóricos
        # de la forma generalizada). En los demás se exige k > 0.
        if clave != "constante" and self.P0 < K_MINIMO_OTROS:
            self.P0 = K_MINIMO_OTROS
            self.aviso = f"En este escenario k mínimo = {K_MINIMO_OTROS}"

        if clave == "senoidal":
            # El tráfico no puede ser negativo: la amplitud no puede pasar de Q0
            if self.p1 > self.Q0:
                self.p1 = self.Q0
                self.aviso = "Amplitud limitada a Q0 (el tráfico no puede ser negativo)"
            self.omega = 2 * np.pi / self.p2          # frecuencia angular [rad/s]
        if clave == "pulso":
            self.t1 = INICIO_PULSO                    # inicio del pico
            self.t2 = INICIO_PULSO + self.p2          # fin del pico
        if clave == "fuga" and self.p1 <= 0:
            self.clave = "constante"   # si el GC no se degrada, es el caso del documento

    @property
    def nombre(self):
        return ESCENARIOS[self.clave_elegida]["nombre"]

    # ------------------------------------------------------------------
    # Coeficientes de la EDO
    # ------------------------------------------------------------------
    def Q(self, t):
        """Tasa de entrada Q(t) [MB/s] según el escenario."""
        t = np.asarray(t, dtype=float)
        if self.clave == "senoidal":
            return self.Q0 + self.p1 * np.sin(self.omega * t)
        if self.clave == "pulso":
            dentro_del_pico = (t >= self.t1) & (t < self.t2)
            return self.Q0 + self.p1 * dentro_del_pico
        if self.clave == "lineal":
            return self.Q0 + self.p1 * t
        return self.Q0 + 0.0 * t                       # constante

    def P(self, t):
        """Coeficiente de limpieza P(t) [1/s] según el escenario."""
        t = np.asarray(t, dtype=float)
        if self.clave == "fuga":
            return self.P0 * np.exp(-self.p1 * t)      # el GC pierde eficacia
        return self.P0 + 0.0 * t

    def derivada(self, t, A):
        """
        Lado derecho de la EDO escrita como  dA/dt = Q(t) - P(t)·A.
        Es la ley de balance de flujo: lo que entra menos lo que se limpia.
        """
        return self.Q(t) - self.P(t) * A

    # ------------------------------------------------------------------
    # Soluciones
    # ------------------------------------------------------------------
    def tiene_formula_cerrada(self):
        """La fuga de memoria no tiene primitiva elemental; las demás sí."""
        return self.clave != "fuga"

    def coeficientes_senoidal(self):
        """
        Para Q(t) = Q0 + Qa·sen(ωt) la integral ∫ e^(Ps)·sen(ωs) ds se resuelve
        por partes dos veces y deja una solución de la forma
            A(t) = Q0/P + a·sen(ωt) + b·cos(ωt) + C·e^(-Pt)
        con  a = Qa·P/(P²+ω²),  b = -Qa·ω/(P²+ω²)  y  C de la condición inicial.
        """
        P, w, Qa = self.P0, self.omega, self.p1
        den = P ** 2 + w ** 2
        a = Qa * P / den
        b = -Qa * w / den
        C = self.A0 - self.Q0 / P - b          # porque A(0) = Q0/P + b + C
        return a, b, C

    def coeficientes_lineal(self):
        """
        Para Q(t) = Q0 + k·t la integral ∫ e^(Ps)·s ds se resuelve por partes:
            A(t) = α + β·t + C·e^(-Pt),  con  β = k/P  y  α = Q0/P - k/P²
        """
        P, k = self.P0, self.p1
        beta = k / P
        alfa = self.Q0 / P - k / P ** 2
        C = self.A0 - alfa
        return alfa, beta, C

    def valores_pulso(self):
        """
        El pulso se resuelve por TRAMOS: en cada tramo Q es constante y vale la
        fórmula del documento, tomando como condición inicial el valor final del
        tramo anterior. Devuelve A(t1) (inicio del pico) y A(t2) (fin del pico).
        """
        P = self.P0
        nivel_base = self.Q0 / P
        nivel_pico = (self.Q0 + self.p1) / P
        A1 = nivel_base + (self.A0 - nivel_base) * np.exp(-P * self.t1)
        A2 = nivel_pico + (A1 - nivel_pico) * np.exp(-P * self.p2)
        return A1, A2

    def solucion_cerrada(self, t):
        """
        Solución EXACTA obtenida a mano con el factor integrante. Devuelve None
        si el escenario no tiene fórmula cerrada (fuga de memoria).
        """
        t = np.asarray(t, dtype=float)
        P, A0 = self.P0, self.A0
        if self.clave == "constante":
            if abs(P) < 1e-12:                 # k = 0: sin limpieza, A = A0 + Q·t
                return A0 + self.Q0 * t
            A_eq = self.Q0 / P                 # A(t) = Q/k + (A0 - Q/k)·e^(-kt)
            return A_eq + (A0 - A_eq) * np.exp(-P * t)
        if self.clave == "senoidal":
            a, b, C = self.coeficientes_senoidal()
            w = self.omega
            return self.Q0 / P + a * np.sin(w * t) + b * np.cos(w * t) + C * np.exp(-P * t)
        if self.clave == "lineal":
            alfa, beta, C = self.coeficientes_lineal()
            return alfa + beta * t + C * np.exp(-P * t)
        if self.clave == "pulso":
            A1, A2 = self.valores_pulso()
            base, pico = self.Q0 / P, (self.Q0 + self.p1) / P
            # np.maximum evita exponentes positivos en los tramos que no aplican
            tramo1 = base + (A0 - base) * np.exp(-P * t)
            tramo2 = pico + (A1 - pico) * np.exp(-P * np.maximum(t - self.t1, 0))
            tramo3 = base + (A2 - base) * np.exp(-P * np.maximum(t - self.t2, 0))
            return np.where(t < self.t1, tramo1, np.where(t < self.t2, tramo2, tramo3))
        return None

    def solucion_general(self, t):
        """
        Evalúa NUMÉRICAMENTE la fórmula general del factor integrante

            A(t) = (1/mu(t))·[ A0 + ∫0^t mu(s)·Q(s) ds ] ,  mu(t) = e^(∫0^t P)

        Sirve para cualquier Q(t) y P(t), aunque la integral no tenga primitiva.

        Se aplica la fórmula intervalo por intervalo: en cada intervalo pequeño
        [t_n, t_n + h] se toman P y Q en el punto medio (constantes) y la fórmula
        general da exactamente la solución del documento en ese tramo:

            A_(n+1) = A_n·e^(-P·h) + (Q/P)·(1 - e^(-P·h))

        Trabajar por tramos evita calcular e^(P·t) para t grandes (números
        enormes que la computadora no puede guardar).
        """
        t = np.asarray(t, dtype=float)
        A = np.empty_like(t)
        A[0] = self.A0
        h = np.diff(t)                         # tamaño de cada intervalo
        t_medio = t[:-1] + h / 2
        P_m = self.P(t_medio)
        Q_m = self.Q(t_medio)
        decaimiento = np.exp(-P_m * h)         # = mu(t_n) / mu(t_n + h)
        # (1 - e^(-P·h)) / P   calculado de forma estable (si P ≈ 0 vale h)
        with np.errstate(divide="ignore", invalid="ignore"):
            ganancia = np.where(np.abs(P_m) > 1e-12, -np.expm1(-P_m * h) / P_m, h)
        for n in range(len(h)):
            A[n + 1] = A[n] * decaimiento[n] + Q_m[n] * ganancia[n]
        return A

    def solucion(self, t):
        """Usa la fórmula cerrada si existe; si no, la fórmula general numérica."""
        exacta = self.solucion_cerrada(t)
        return exacta if exacta is not None else self.solucion_general(t)

    def runge_kutta4(self, t_final, pasos):
        """
        Método numérico de Runge-Kutta de orden 4 (RK4). Resuelve la EDO usando
        SOLO la derivada, sin ninguna fórmula. Sirve para comprobar las otras
        soluciones: si coinciden, el desarrollo matemático es correcto.

        En cada paso de tamaño h se calculan 4 pendientes y se promedian:
            k1 = f(t, A)
            k2 = f(t + h/2, A + h·k1/2)
            k3 = f(t + h/2, A + h·k2/2)
            k4 = f(t + h,   A + h·k3)
            A_siguiente = A + (h/6)·(k1 + 2·k2 + 2·k3 + k4)
        """
        f = self.derivada
        h = t_final / pasos
        t = np.linspace(0, t_final, pasos + 1)
        A = np.zeros(pasos + 1)
        A[0] = self.A0
        for i in range(pasos):
            k1 = f(t[i], A[i])
            k2 = f(t[i] + h / 2, A[i] + h * k1 / 2)
            k3 = f(t[i] + h / 2, A[i] + h * k2 / 2)
            k4 = f(t[i] + h, A[i] + h * k3)
            A[i + 1] = A[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return t, A

    def malla(self, T, n=1200):
        """
        Instantes de tiempo donde se evalúa la solución. En el pulso se añaden
        los instantes exactos donde empieza y termina el pico (saltos de Q).
        """
        t = np.linspace(0, T, n)
        if self.clave == "pulso":
            saltos = [x for x in (self.t1, self.t2) if 0 < x < T]
            t = np.union1d(t, saltos)
        return t

    # ------------------------------------------------------------------
    # Análisis del sistema
    # ------------------------------------------------------------------
    def techo(self):
        """
        Nivel máximo que alcanza la memoria a largo plazo (o en el pico).
          constante : equilibrio Q/P (límite cuando t -> infinito)
          senoidal  : Q0/P + Qa/sqrt(P² + ω²)  (cresta de la oscilación final)
          pulso     : el mayor entre el equilibrio Q0/P y el valor al final del pico
          lineal    : infinito si k > 0 (el tráfico crece sin parar)
          fuga      : infinito si r > 0 (al final el GC deja de limpiar)
        """
        if self.P0 <= 0:
            return np.inf        # k <= 0: no hay equilibrio, la memoria crece sin límite
        base = self.Q0 / self.P0
        if self.clave == "senoidal":
            return base + self.p1 / np.sqrt(self.P0 ** 2 + self.omega ** 2)
        if self.clave == "pulso":
            return max(base, self.valores_pulso()[1])
        if self.clave == "lineal" and self.p1 > 0:
            return np.inf
        if self.clave == "fuga" and self.Q0 > 0:
            return np.inf
        return base

    def tiempo_de_colapso(self, T):
        """
        Instante t* en el que la memoria alcanza el límite físico (error OOM).
        Devuelve None si NUNCA lo alcanza.

        Caso constante: se despeja de A(t*) = A_max:
            t* = -(1/P)·ln[ (A_max - A_eq) / (A0 - A_eq) ]
        Demás casos: se recorre la solución numérica y se busca el primer
        instante en que A(t) >= A_max (si la memoria crece sin límite se va
        ampliando el horizonte de búsqueda hasta encontrarlo).
        """
        if self.A0 >= self.A_max:
            return 0.0                                   # arranca saturado
        if self.clave == "constante":
            k, Q = self.P0, self.Q0
            if abs(k) < 1e-12:                    # k = 0: recta A0 + Q·t
                return (self.A_max - self.A0) / Q if Q > 0 else None
            A_eq = Q / k
            if k > 0 and A_eq <= self.A_max:
                return None
            # También vale para k < 0 (A_eq < 0): el logaritmo es positivo
            return -(1.0 / k) * np.log((self.A_max - A_eq) / (self.A0 - A_eq))

        diverge = np.isinf(self.techo())
        horizonte = max(T, 8.0 / self.P0)
        if self.clave == "senoidal":
            horizonte += 3 * self.p2
        if self.clave == "pulso":
            horizonte += self.t2
        for _ in range(16):
            t = self.malla(horizonte, 8000)
            A = self.solucion_general(t)
            cruces = np.nonzero(A >= self.A_max)[0]
            if cruces.size:
                i = cruces[0]
                # interpolación lineal entre los dos puntos que rodean el cruce
                fraccion = (self.A_max - A[i - 1]) / (A[i] - A[i - 1])
                return t[i - 1] + fraccion * (t[i] - t[i - 1])
            if not diverge:
                return None
            horizonte *= 2
        return None

    def diagnostico(self, T):
        """Reúne los resultados importantes y decide el color del semáforo."""
        techo = self.techo()
        porcentaje = techo / self.A_max * 100 if np.isfinite(techo) else np.inf
        t_col = self.tiempo_de_colapso(T)

        if t_col is not None or np.isinf(techo):
            estado, color = "COLAPSO (OOM)", ROJO
        elif porcentaje >= UMBRAL_RIESGO:
            estado, color = "EN RIESGO", AMARILLO
        else:
            estado, color = "SISTEMA SEGURO", VERDE

        return {
            "techo": techo,
            "porcentaje": porcentaje,
            "libre": self.A_max - techo,
            "t_colapso": t_col,
            "diverge": np.isinf(techo),
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


def termino(coeficiente, cuerpo="", decimales=2):
    """Escribe ' + 12.5·cuerpo' o ' - 12.5·cuerpo' según el signo."""
    signo = "-" if coeficiente < 0 else "+"
    return f" {signo} {num(abs(coeficiente), decimales)}{cuerpo}"


def texto_gb(mb):
    """8192 -> '8 GB'"""
    return f"{num(mb / MB_POR_GB, 1)} GB"


def pasos_constante(m, d):
    """
    Caso constante en FORMA GENERALIZADA: primero se resuelve con los símbolos
    k, Q y A0 (la teoría) y después se sustituyen los valores elegidos.
    Contempla los tres casos del factor integrante: k > 0, k = 0 y k < 0.
    """
    k, Q, A0, A_max = m.P0, m.Q0, m.A0, m.A_max
    kk, q, a0 = num(k, 4), num(Q), num(A0)
    pasos = [
        ("1. Forma generalizada (P(t) = k constante)",
         r"$\frac{dA}{dt} + k\,A = Q, \quad A(0) = A_0$"),
        ("2. Valores elegidos",
         rf"$k = {kk},\;\; Q = {q},\;\; A_0 = {a0},\;\; A_{{máx}} = {num(A_max)}$"),
    ]
    t_col = d["t_colapso"]

    if abs(k) < 1e-12:                                    # caso k = 0
        pasos += [
            ("3. Factor integrante (k = 0)", r"$\mu(t) = e^{\int 0\,dt} = e^{0} = 1$"),
            ("4. Integrar directamente", r"$A(t) = A_0 + Q\,t$"),
            ("5. Solución particular", rf"$A(t) = {a0} + {q}\,t$  [MB]"),
            ("6. Límite: sin depuración no hay equilibrio", r"$\lim_{t\to\infty} A(t) = \infty$"),
            ("7. Colapso:  A(t*) = A_max",
             rf"$t^* = \frac{{A_{{máx}} - A_0}}{{Q}} = {num(t_col, 1)}$ s" if t_col is not None
             else "no colapsa (Q = 0)"),
        ]
        return pasos

    Aeq = Q / k
    C = A0 - Aeq
    # e^(-kt): si k < 0 el exponente queda positivo (evita escribir "--")
    exponente = (rf"\,e^{{-{kk}t}}" if k > 0 else rf"\,e^{{{num(-k, 4)}t}}")
    pasos += [
        ("3. Factor integrante",
         rf"$\mu(t) = e^{{\int k\,dt}} = e^{{kt}} = e^{{{kk}t}}$"),
        ("4. Agrupar e integrar",
         r"$\frac{d}{dt}\left[e^{kt}A\right] = Q\,e^{kt} \;\Rightarrow\; e^{kt}A = \frac{Q}{k}e^{kt} + C$"),
        ("5. Solución generalizada (con A(0) = A0)",
         r"$A(t) = \frac{Q}{k} + \left(A_0 - \frac{Q}{k}\right)e^{-kt}$"),
        ("6. Sustituir los valores",
         rf"$\frac{{Q}}{{k}} = {num(Aeq)}, \quad A_0 - \frac{{Q}}{{k}} = {num(C)}$"),
        ("7. Solución particular",
         rf"$A(t) = {num(Aeq)}{termino(C, exponente)}$  [MB]"),
    ]
    if k > 0:
        pasos += [
            ("8. Límite (k > 0: e^(-kt) → 0)",
             rf"$\lim_{{t\to\infty}} A(t) = \frac{{Q}}{{k}} = {num(Aeq)}$ MB"),
            ("9. Porcentaje generalizado",
             rf"$\frac{{Q}}{{k\,A_{{máx}}}} \times 100 = {num(d['porcentaje'])}$ %"),
            ("10. Condición de no colapso:  Q < k·A_max",
             rf"$Q_{{máx}} = k\,A_{{máx}} = {num(k * A_max)}$ MB/s  "
             + ("(cumple)" if Q < k * A_max else "(no cumple)")),
        ]
    else:
        pasos += [
            ("8. Límite (k < 0: e^(-kt) → ∞)",
             r"$\lim_{t\to\infty} A(t) = \infty$  (crece exponencialmente)"),
            ("9. Colapso:  A(t*) = A_max",
             rf"$t^* = -\frac{{1}}{{k}}\ln\frac{{A_{{máx}} - Q/k}}{{A_0 - Q/k}} = {num(t_col, 1)}$ s"),
        ]
    return pasos


def pasos_demostracion(m, d):
    """
    Genera la resolución por FACTOR INTEGRANTE con los valores actuales.
    Devuelve una lista de pares (título del paso, fórmula en notación
    matemática de matplotlib).
    """
    if m.clave == "constante":
        return pasos_constante(m, d)

    p, q = num(m.P0, 4), num(m.Q0)

    # ---- Coeficientes y factor integrante de cada escenario ----
    if m.clave == "senoidal":
        Q_tex = rf"{q} + {num(m.p1)}\sin({num(m.omega, 4)}t)"
    elif m.clave == "pulso":
        Q_tex = (rf"{q}\;\;(\mathrm{{y}}\; {num(m.Q0 + m.p1)}\;\mathrm{{si}}\; "
                 rf"{num(m.t1)} \leq t < {num(m.t2)})")
    elif m.clave == "lineal":
        Q_tex = rf"{q} + {num(m.p1)}\,t"
    else:
        Q_tex = q
    if m.clave == "fuga":
        r = num(m.p1, 4)
        P_tex = rf"{p}\,e^{{-{r}t}}"
        mu_tex = rf"e^{{{num(m.P0 / m.p1, 3)}(1 - e^{{-{r}t}})}}"
    else:
        P_tex = rf"k = {p}"
        mu_tex = rf"e^{{{p}t}}"

    pasos = [
        ("1. Forma estándar (caso general, P(t) variable)",
         r"$\frac{dA}{dt} + P(t)\,A = Q(t)$"),
        ("2. Coeficientes de este escenario",
         rf"$P(t) = {P_tex},\quad Q(t) = {Q_tex}$"),
        ("3. Factor integrante",
         rf"$\mu(t) = e^{{\int_0^t P(s)\,ds}} = {mu_tex}$"),
        ("4. Fórmula general de la solución",
         r"$A(t) = \frac{1}{\mu(t)}\left[A_0 + \int_0^t \mu(s)\,Q(s)\,ds\right]$"),
    ]

    t_col = d["t_colapso"]
    texto_colapso = (rf"$t^* \approx {num(t_col, 1)}$ s" if t_col is not None
                     else "no alcanza el límite")

    # ---- Pasos propios de cada escenario ----
    if m.clave == "senoidal":
        a, b, C = m.coeficientes_senoidal()
        amp = m.p1 / np.sqrt(m.P0 ** 2 + m.omega ** 2)
        pasos += [
            ("5. Integral por partes (dos veces)",
             r"$A(t) = \frac{Q_0}{k} + a\,\sin(\omega t) + b\,\cos(\omega t) + C\,e^{-kt}$"),
            ("6. Coeficientes y condición inicial A(0) = A0",
             rf"$a = \frac{{Q_a k}}{{k^2+\omega^2}} = {num(a)},\;\; "
             rf"b = {num(b)},\;\; C = {num(C)}$"),
            ("7. Régimen permanente (t → ∞)",
             rf"$A(t) \to {num(m.Q0 / m.P0)} \pm {num(amp)}$  (oscila)"),
            ("8. Techo de la memoria",
             rf"$\frac{{Q_0}}{{k}} + \frac{{Q_a}}{{\sqrt{{k^2+\omega^2}}}} = {num(d['techo'])}$"
             rf" MB  $\to$  {num(d['porcentaje'])} %"),
        ]
    elif m.clave == "pulso":
        A1, A2 = m.valores_pulso()
        base, pico = m.Q0 / m.P0, (m.Q0 + m.p1) / m.P0
        t1, t2 = num(m.t1), num(m.t2)
        pasos += [
            (f"5. Tramo 1  (0 ≤ t < {t1})",
             rf"$A(t) = {num(base)}{termino(m.A0 - base, rf'\,e^{{-{p}t}}')}$"),
            (f"6. Tramo 2: durante el pico  ({t1} ≤ t < {t2})",
             rf"$A(t) = {num(pico)}{termino(A1 - pico, rf'\,e^{{-{p}(t-{t1})}}')}$"),
            (f"7. Tramo 3: después del pico  (t ≥ {t2})",
             rf"$A(t) = {num(base)}{termino(A2 - base, rf'\,e^{{-{p}(t-{t2})}}')}$"),
            ("8. Pico máximo (al terminar el pulso)",
             rf"$A({t2}) = {num(A2)}$ MB  $\to$  {num(A2 / m.A_max * 100)} %"),
        ]
    elif m.clave == "lineal":
        alfa, beta, C = m.coeficientes_lineal()
        largo_plazo = (rf"$A(t) \approx {num(alfa)} + {num(beta)}\,t \to \infty$"
                       if m.p1 > 0 else rf"$A(t) \to {num(alfa)}$ MB")
        pasos += [
            ("5. Integral por partes",
             rf"$A(t) = {num(alfa)} + {num(beta)}\,t + C\,e^{{-{p}t}}$"),
            ("6. Condición inicial  A(0) = A0",
             rf"$C = A_0 - \alpha = {num(C)}$"),
            ("7. Comportamiento a largo plazo", largo_plazo),
            ("8. Colapso:  A(t*) = A_max", texto_colapso),
        ]
    else:  # fuga
        limite_int = m.P0 / m.p1
        pasos += [
            ("5. La integral no tiene primitiva elemental",
             r"$\int_0^t \mu(s)\,Q\,ds$  se evalúa numéricamente"),
            ("6. El factor integrante se estanca",
             rf"$\int_0^\infty P\,dt = \frac{{k}}{{r}} = {num(limite_int)}"
             rf" \;\Rightarrow\; \mu \to e^{{{num(limite_int)}}}$"),
            ("7. Comportamiento a largo plazo",
             rf"$P(t) \to 0 \;\Rightarrow\; \frac{{dA}}{{dt}} \to {q}$ MB/s $\Rightarrow A \to \infty$"),
            ("8. Colapso:  A(t*) = A_max", texto_colapso),
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


def escala_vertical(pico_curva, A_max):
    """
    Decide hasta dónde llega el eje vertical.
    Si el límite de RAM está cerca de la curva, se muestran los dos. Si está
    muy por encima (por ejemplo 128 GB con una curva de 4 GB), dibujarlo
    aplastaría la curva: se acerca la vista a la curva y el límite se indica
    con un letrero. Devuelve (altura del eje, ¿se ve el límite?).
    """
    pico_curva = max(pico_curva, 100.0)
    if A_max <= 2.5 * pico_curva:
        return 1.15 * max(A_max, pico_curva), True
    return 1.3 * pico_curva, False


def dibujar_memoria(ax, m, T, t_marcador=None):
    """
    GRÁFICA PRINCIPAL: memoria ocupada A(t) frente al límite de RAM.
      - Línea azul      : solución (fórmula cerrada o fórmula general numérica).
      - Puntos naranjas : solución numérica RK4 (comprobación).
      - Línea roja      : límite físico de la RAM (A_max).
      - Línea verde     : isoclina Q(t)/P(t) (donde dA/dt = 0). En el caso
                          constante es el equilibrio Q/P.
    """
    # En versiones viejas de matplotlib, ax.clear() no borra el eje secundario
    # de porcentaje: se quita a mano para que no se acumulen uno encima de otro
    for hijo in list(getattr(ax, "child_axes", [])):
        hijo.remove()
    estilo_ejes(ax)
    d = m.diagnostico(T)
    t = m.malla(T)
    A = m.solucion(t)
    A_max = m.A_max

    # Escala: se mira la curva (sin dejar que la proyección tras el colapso
    # la estire demasiado)
    pico = min(np.max(A), 1.25 * A_max)
    y_tope, limite_visible = escala_vertical(pico, A_max)

    if limite_visible:
        ax.axhspan(A_max, y_tope, color=ROJO, alpha=0.10)
        ax.axhline(A_max, color=ROJO, linestyle="--", linewidth=1.8,
                   label=f"Límite de RAM = {num(A_max)} MB ({texto_gb(A_max)})")
    else:
        ax.text(0.99, 0.98, f"▲ Límite de RAM: {num(A_max)} MB ({texto_gb(A_max)})\n"
                            "muy por encima de la curva",
                ha="right", va="top", transform=ax.transAxes, fontsize=8.5,
                color=ROJO, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ROJO))

    # Isoclina nula: si la memoria estuviera en esta curva, no cambiaría
    with np.errstate(divide="ignore", invalid="ignore"):
        isoclina = np.where(m.P(t) > 0, m.Q(t) / m.P(t), np.nan)
    if m.clave == "constante":
        etiqueta_iso = (f"Equilibrio Q/k = {num(m.Q0 / m.P0)} MB" if m.P0 > 0
                        else "Sin equilibrio (k ≤ 0)")
    else:
        etiqueta_iso = "Isoclina Q(t)/P(t)  (dA/dt = 0)"
    ax.plot(t, np.minimum(isoclina, y_tope * 1.5), color=VERDE, linestyle=":",
            linewidth=1.8, label=etiqueta_iso)

    etiqueta = ("A(t) analítica (factor integrante)" if m.tiene_formula_cerrada()
                else "A(t) fórmula general (integral numérica)")
    t_col = d["t_colapso"]
    if t_col is not None and t_col <= T:
        # Hasta el colapso la curva es real; después es solo una proyección
        antes = t <= t_col
        ax.plot(t[antes], A[antes], color=AZUL, linewidth=2.8, label=etiqueta)
        ax.plot(t[~antes], A[~antes], color=AZUL, linewidth=1.5, linestyle="--",
                alpha=0.55, label="Proyección sin límite físico")
        ax.plot([t_col], [A_max], marker="X", color=ROJO, markersize=13,
                markeredgecolor="white", zorder=6, linestyle="none",
                label=f"Colapso OOM en t = {num(t_col, 1)} s")
        ax.axvline(t_col, color=ROJO, linewidth=1, alpha=0.5)
    else:
        ax.plot(t, A, color=AZUL, linewidth=2.8, label=etiqueta)

    # Comprobación numérica con RK4
    t_rk, A_rk = m.runge_kutta4(T, 400)
    ax.plot(t_rk[::16], A_rk[::16], "o", color=NARANJA, markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.6,
            label="Numérica RK4 (comprobación)")
    ax.plot([0], [m.A0], "s", color=AZUL, markersize=7, markeredgecolor="white",
            zorder=6)

    # Marcador móvil de la animación
    if t_marcador is not None:
        A_m = float(np.interp(t_marcador, t, A))
        y_m = min(A_m, y_tope)
        ax.plot([t_marcador], [y_m], "o", color="black",
                markersize=9, markeredgecolor="white", zorder=11)
        # El letrero se pone del lado donde hay espacio: a la izquierda del
        # punto cuando está en la mitad derecha, y arriba cuando el punto está
        # abajo. Así nunca se sale de la gráfica ni queda tapado por la leyenda.
        a_la_derecha = t_marcador < 0.55 * T
        arriba = y_m < 0.45 * y_tope
        texto = (f"t = {num(t_marcador, 1)} s\nA = {num(A_m, 1)} MB\n"
                 f"{num(A_m / A_max * 100, 1)} % RAM")
        ax.annotate(texto, xy=(t_marcador, y_m),
                    xytext=(14 if a_la_derecha else -14, 22 if arriba else -22),
                    textcoords="offset points", fontsize=9, fontweight="bold",
                    ha="left" if a_la_derecha else "right",
                    va="bottom" if arriba else "top", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1),
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
        # Lectura fija en la esquina superior izquierda (siempre visible)
        ax.text(0.01, 0.98, f"Memoria usada: {num(A_m, 1)} MB  "
                            f"({num(A_m / A_max * 100, 1)} % de {texto_gb(A_max)})",
                transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
                fontweight="bold", color="#17202e", zorder=12,
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff8d6", ec=AMARILLO))

    ax.set_xlim(0, T)
    ax.set_ylim(0, y_tope)
    ax.set_xlabel("Tiempo  t  [s]", fontsize=10)
    ax.set_ylabel("Memoria ocupada  A(t)  [MB]", fontsize=10)
    ax.set_title(f"Memoria RAM ocupada  —  {m.nombre}",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="lower right" if m.clave == "constante" else "best",
              fontsize=7.8, framealpha=0.95)

    # Eje derecho: la misma altura expresada en % de la RAM total
    eje_pct = ax.secondary_yaxis(
        "right", functions=(lambda y: y / A_max * 100, lambda p: p * A_max / 100))
    eje_pct.set_ylabel(f"% de la RAM ({texto_gb(A_max)})", fontsize=9)
    eje_pct.tick_params(labelsize=8.5)


def dibujar_campo(ax, m, T):
    """
    CAMPO DE DIRECCIONES de la EDO.
    En cada punto (t, A) se dibuja una flechita con pendiente
    dA/dt = Q(t) - P(t)·A. Toda solución debe "seguir" las flechas.
    Las curvas grises son soluciones con otras memorias iniciales.
    """
    estilo_ejes(ax)
    t = m.malla(T, 300)
    A_nuestra = m.solucion(t)
    y_tope, limite_visible = escala_vertical(min(np.max(A_nuestra), 1.25 * m.A_max),
                                             m.A_max)

    tt, aa = np.meshgrid(np.linspace(0, T, 17), np.linspace(0, y_tope, 13))
    pendiente = m.derivada(tt, aa)
    # Flechas normalizadas (misma longitud), en proporciones de la gráfica
    u = np.ones_like(pendiente) / T
    v = pendiente / y_tope
    largo = np.hypot(u, v)
    ax.quiver(tt, aa, u / largo, v / largo, angles="uv", pivot="mid",
              color="#9aa7b8", scale=32, width=0.0035, headwidth=3.2)

    # Familia de soluciones con distintas condiciones iniciales
    for a_ini in np.linspace(0, y_tope, 7):
        otra = Modelo(m.clave_elegida, m.Q0, m.P0, a_ini, m.A_max, m.p1, m.p2)
        ax.plot(t, otra.solucion_general(t), color=GRIS, linewidth=0.9, alpha=0.65)

    ax.plot(t, A_nuestra, color=AZUL, linewidth=2.4,
            label=f"Solución con A(0) = {num(m.A0)}")
    with np.errstate(divide="ignore", invalid="ignore"):
        isoclina = np.where(m.P(t) > 0, m.Q(t) / m.P(t), np.nan)
    ax.plot(t, np.minimum(isoclina, y_tope * 1.5), color=VERDE,
            linestyle=":", linewidth=1.6, label="dA/dt = 0")
    if limite_visible:
        ax.axhline(m.A_max, color=ROJO, linestyle="--", linewidth=1.3)

    ax.set_xlim(0, T)
    ax.set_ylim(0, y_tope)
    ax.set_xlabel("t  [s]", fontsize=9)
    ax.set_ylabel("A  [MB]", fontsize=9)
    ax.set_title("Campo de direcciones de la EDO", fontsize=10.5,
                 fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=7.3, framealpha=0.95)


def dibujar_flujos(ax, m, T):
    """
    BALANCE DE FLUJOS: tasa de entrada Q(t) contra tasa de salida P(t)·A(t).
    La distancia entre ambas curvas es justamente dA/dt. Donde se cruzan,
    entra lo mismo que sale y la memoria deja de cambiar en ese instante.
    """
    estilo_ejes(ax)
    t = m.malla(T, 400)
    entrada = m.Q(t)
    salida = m.P(t) * m.solucion(t)

    ax.plot(t, entrada, color=NARANJA, linewidth=2.2, label="Entrada  Q(t)")
    ax.plot(t, salida, color=VERDE, linewidth=2.2, label="Salida  P(t)·A(t)")
    ax.fill_between(t, salida, entrada, color=AZUL, alpha=0.13,
                    label="Diferencia = dA/dt")

    ax.set_xlim(0, T)
    ax.set_ylim(0, max(np.max(entrada), np.max(salida), 1) * 1.3)
    ax.set_xlabel("t  [s]", fontsize=9)
    ax.set_ylabel("Tasa  [MB/s]", fontsize=9)
    ax.set_title("Balance de flujos: entrada vs salida", fontsize=10.5,
                 fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=7.3, framealpha=0.95)


def dibujar_semaforo(ax, d):
    """SEMÁFORO de seguridad: enciende la luz que corresponde al estado."""
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.2, 0.9), 5.0, 2.2,
                                boxstyle="round,pad=0.15,rounding_size=0.5",
                                fc="#20242c", ec="none"))
    for x, color in [(1.2, VERDE), (2.7, AMARILLO), (4.2, ROJO)]:
        encendida = (color == d["color"])
        ax.add_patch(Circle((x, 2.0), 0.62, fc=color if encendida else "#4a4f5a",
                            ec="white" if encendida else "none", lw=1.5))

    ax.text(5.9, 2.55, d["estado"], fontsize=11.5, fontweight="bold",
            color=d["color"] if d["color"] != AMARILLO else "#a87f00", va="center")
    if np.isfinite(d["porcentaje"]):
        detalle = f"Techo de memoria: {num(d['porcentaje'])} %"
    else:
        detalle = "La memoria crece sin límite"
    ax.text(5.9, 1.45, detalle, fontsize=9, color="#30343b", va="center")


def texto_resultados(m, d):
    """Resumen numérico que se muestra debajo del semáforo."""
    lineas = [f"Límite de RAM: {num(m.A_max)} MB ({texto_gb(m.A_max)})"]
    if m.clave == "constante" and m.P0 > 0:
        A_eq = m.Q0 / m.P0
        lineas.append(f"Equilibrio  Q/k = {num(A_eq)} MB")
        lineas.append(f"Constante C = A0 - Q/k = {num(m.A0 - A_eq)}")
        lineas.append(f"Tráfico máx. k·A_max = {num(m.P0 * m.A_max)} MB/s")
        lineas.append(f"Depuración mín. Q/A_max = {num(m.Q0 / m.A_max, 4)} 1/s")
    elif m.clave == "constante":
        lineas.append("k ≤ 0: no hay equilibrio")
    elif np.isfinite(d["techo"]):
        lineas.append(f"Techo de memoria = {num(d['techo'])} MB")
    else:
        lineas.append("Techo de memoria = infinito")
    if np.isfinite(d["techo"]):
        if d["libre"] >= 0:
            lineas.append(f"Libre en el techo: {num(d['libre'])} MB "
                          f"({num(100 - d['porcentaje'])} %)")
        else:
            lineas.append(f"Faltarían {num(-d['libre'])} MB de RAM")
    if d["t_colapso"] is not None:
        lineas.append(f"Colapso OOM en  t = {num(d['t_colapso'], 1)} s")
    elif d["diverge"]:
        lineas.append("Colapsa, pero muy tarde")
    else:
        lineas.append(f"Nunca alcanza el límite de {texto_gb(m.A_max)}")
    if m.aviso:
        lineas.append(m.aviso)
    return "\n".join(lineas)


# ============================================================================
# 5. APLICACIÓN INTERACTIVA
# ============================================================================
class SimuladorMemoria:
    """
    Ventana principal, organizada en tres columnas:
      IZQUIERDA : escenario, sliders y botones.
      CENTRO    : gráfica principal + campo de direcciones + balance de flujos.
      DERECHA   : semáforo, resultados y demostración paso a paso.
    Cada vez que se mueve un slider o se cambia de escenario se llama a
    self.actualizar(), que recalcula el modelo y vuelve a dibujar todo.
    """

    def __init__(self):
        self.t_marcador = None      # posición del punto animado (None = oculto)
        self.animando = False
        self.ocupado = False        # evita redibujar mientras se reconfigura
        self.escenario = "constante"

        self.fig = plt.figure(figsize=(15.5, 8.7), facecolor=FONDO)
        try:
            self.fig.canvas.manager.set_window_title(
                "Simulador de memoria del servidor - Ecuaciones Diferenciales")
        except Exception:
            pass

        # ---- Encabezado ----------------------------------------------------
        self.fig.text(0.015, 0.958,
                      "Modelo de Acumulación y Depuración Dinámica de Memoria en un Servidor",
                      fontsize=15, fontweight="bold", color="#17202e")
        self.fig.text(0.015, 0.927,
                      r"Caso general:  $\frac{dA}{dt} + P(t)\,A = Q(t)$,   $A(0)=A_0$"
                      r"   (con limpieza constante, $P(t) = k$)  —  resuelto por factor integrante",
                      fontsize=10.5, color="#3b4658")

        # ---- Columna central: las tres gráficas -----------------------------
        self.ax_mem = self.fig.add_axes([0.268, 0.465, 0.375, 0.41])
        self.ax_campo = self.fig.add_axes([0.268, 0.07, 0.17, 0.29])
        self.ax_flujos = self.fig.add_axes([0.493, 0.07, 0.17, 0.29])

        # ---- Columna derecha: semáforo, resultados y demostración -----------
        self.fig.text(0.715, 0.885, "SEMÁFORO DE SEGURIDAD", fontsize=10,
                      fontweight="bold", color=GRIS)
        self.ax_semaforo = self.fig.add_axes([0.715, 0.775, 0.27, 0.105])
        self.fig.text(0.715, 0.765, "RESULTADOS", fontsize=10, fontweight="bold",
                      color=GRIS)
        self.txt_resultados = self.fig.text(0.715, 0.745, "", fontsize=8.6,
                                            va="top", family="monospace",
                                            color="#17202e", linespacing=1.45)
        self.ax_pasos = self.fig.add_axes([0.715, 0.025, 0.27, 0.565])

        # ---- Columna izquierda: escenario -----------------------------------
        self.fig.text(0.015, 0.878, "ESCENARIO", fontsize=10, fontweight="bold",
                      color=GRIS)
        ax_radio = self.fig.add_axes([0.012, 0.705, 0.21, 0.168])
        ax_radio.set_facecolor(FONDO)
        for lado in ax_radio.spines.values():
            lado.set_visible(False)
        nombres = [ESCENARIOS[c]["nombre"] for c in ORDEN_ESCENARIOS]
        self.radio = RadioButtons(ax_radio, nombres, active=0, activecolor=AZUL,
                                  label_props={"fontsize": [9.2] * len(nombres)})
        self.radio.on_clicked(self.cambiar_escenario)

        # ---- Columna izquierda: sliders -------------------------------------
        self.fig.text(0.015, 0.682, "CONTROLES", fontsize=10, fontweight="bold",
                      color=GRIS)
        self.y_sliders = [0.632, 0.569, 0.506, 0.443, 0.380, 0.317, 0.254]
        y = self.y_sliders
        self.s_Q = self._crear_slider(y[0], "Tráfico de entrada  Q  [MB/s]",
                                      0, Q_MAXIMO_SLIDER, Q_INICIAL, 1, NARANJA, "%.0f")
        self.s_P = self._crear_slider(y[1], "Limpieza del Garbage Collector  k  [1/s]",
                                      K_MINIMO_SLIDER, 0.15, P_INICIAL, 0.001, VERDE, "%.3f")
        self.s_RAM = self._crear_slider(y[3], "Límite de RAM  A_max  [GB]",
                                        1, RAM_GB_MAXIMA, RAM_GB_INICIAL, 1, ROJO,
                                        "%d GB")
        self.s_A0 = self._crear_slider(y[2], "Memoria inicial  A0  [MB]",
                                       0, A_MAX, A0_INICIAL, 16, AZUL, "%.0f")
        self.s_T = self._crear_slider(y[4], "Tiempo simulado  [s]",
                                      50, 600, T_INICIAL, 10, GRIS, "%.0f")
        self.s_p1 = None
        self.s_p2 = None
        self._sliders_del_escenario()

        # ---- Botones --------------------------------------------------------
        self.b_animar = self._crear_boton([0.015, 0.105, 0.066, 0.048], "Animar",
                                          self.animar)
        self.b_reset = self._crear_boton([0.087, 0.105, 0.066, 0.048],
                                         "Datos del\ndocumento", self.restablecer)
        self.b_guardar = self._crear_boton([0.159, 0.105, 0.066, 0.048],
                                           "Guardar\nimagen", self.guardar)
        self.txt_aviso = self.fig.text(0.015, 0.07, "", fontsize=8, color=GRIS)

        # Temporizador que mueve el punto durante la animación
        self.temporizador = self.fig.canvas.new_timer(interval=40)
        self.temporizador.add_callback(self._paso_animacion)

        self.actualizar()

    # ------------------------------------------------------------------ util
    def _crear_slider(self, y, etiqueta, minimo, maximo, inicial, paso, color, formato):
        """
        Crea un slider con su título encima y conecta el evento de cambio.
        Devuelve el slider; su título queda guardado en slider.titulo.
        """
        titulo = self.fig.text(0.015, y + 0.03, etiqueta, fontsize=8.8,
                               color="#17202e")
        ax = self.fig.add_axes([0.015, y, 0.165, 0.024])
        slider = Slider(ax, "", minimo, maximo, valinit=inicial, valstep=paso,
                        color=color, initcolor="none", valfmt=formato)
        slider.valtext.set_fontsize(9.2)
        slider.valtext.set_fontweight("bold")
        slider.titulo = titulo
        slider.on_changed(self._al_mover_slider)
        return slider

    def _quitar_slider(self, slider):
        """Borra un slider y su título de la ventana."""
        if slider is not None:
            slider.titulo.remove()
            slider.ax.remove()

    def _crear_boton(self, posicion, texto, funcion):
        ax = self.fig.add_axes(posicion)
        boton = Button(ax, texto, color="white", hovercolor="#dfe8f7")
        boton.label.set_fontsize(8.3)
        boton.on_clicked(funcion)
        return boton

    def _sliders_del_escenario(self):
        """
        Crea (o vuelve a crear) los dos sliders extra según el escenario
        elegido: por ejemplo amplitud y periodo en el caso senoidal.
        """
        self._quitar_slider(self.s_p1)
        self._quitar_slider(self.s_p2)
        self.s_p1 = self.s_p2 = None
        e = ESCENARIOS[self.escenario]
        for nombre_atributo, spec, y in (("s_p1", e["param1"], self.y_sliders[5]),
                                         ("s_p2", e["param2"], self.y_sliders[6])):
            if spec is None:
                continue
            etiqueta, minimo, maximo, inicial, paso, formato = spec
            slider = self._crear_slider(y, etiqueta, minimo, maximo, inicial, paso,
                                        "#7b4fb8", formato)
            setattr(self, nombre_atributo, slider)

    def _ajustar_slider_A0(self):
        """La memoria inicial no puede pasar del límite de RAM elegido."""
        A_max = self.s_RAM.val * MB_POR_GB
        if abs(self.s_A0.valmax - A_max) < 1e-9:
            return
        valor = min(self.s_A0.val, A_max)
        self._quitar_slider(self.s_A0)
        self.s_A0 = self._crear_slider(self.y_sliders[2], "Memoria inicial  A0  [MB]",
                                       0, A_max, valor, 16, AZUL, "%.0f")
        self.s_A0.valinit = A0_INICIAL if A0_INICIAL <= A_max else valor

    def modelo(self):
        """Construye el Modelo con los valores actuales de los controles."""
        p1_def, p2_def = valores_por_defecto(self.escenario)
        return Modelo(self.escenario,
                      Q0=self.s_Q.val, P0=self.s_P.val, A0=self.s_A0.val,
                      A_max=self.s_RAM.val * MB_POR_GB,
                      p1=self.s_p1.val if self.s_p1 else p1_def,
                      p2=self.s_p2.val if self.s_p2 else p2_def)

    # ------------------------------------------------------------ eventos
    def _al_mover_slider(self, _valor):
        if self.ocupado:
            return
        self.ocupado = True
        self._ajustar_slider_A0()
        self.ocupado = False
        self.actualizar()

    def cambiar_escenario(self, etiqueta):
        """Se llama al elegir otro escenario en los botones de opción."""
        for clave in ORDEN_ESCENARIOS:
            if ESCENARIOS[clave]["nombre"] == etiqueta:
                self.escenario = clave
        self._detener_animacion()       # si estaba animando, se detiene
        self.ocupado = True
        self._sliders_del_escenario()
        self.ocupado = False
        self.actualizar()

    def _detener_animacion(self):
        """Para el temporizador y oculta el punto de la animación."""
        self.temporizador.stop()
        self.animando = False
        self.t_marcador = None
        self.b_animar.label.set_text("Animar")

    # ------------------------------------------------------------ redibujado
    def actualizar(self):
        """Recalcula el modelo y vuelve a dibujar TODO con los valores actuales."""
        if self.ocupado:
            return
        m = self.modelo()
        T = float(self.s_T.val)
        d = m.diagnostico(T)

        for ax in (self.ax_mem, self.ax_campo, self.ax_flujos):
            ax.clear()
        dibujar_memoria(self.ax_mem, m, T, t_marcador=self.t_marcador)
        dibujar_campo(self.ax_campo, m, T)
        dibujar_flujos(self.ax_flujos, m, T)
        dibujar_semaforo(self.ax_semaforo, d)
        self.txt_resultados.set_text(texto_resultados(m, d))
        self._dibujar_pasos(m, d)
        self.fig.canvas.draw_idle()

    def _dibujar_pasos(self, m, d):
        """Escribe la demostración paso a paso en el panel de la derecha."""
        ax = self.ax_pasos
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc="white", ec="#d5dbe5", lw=1.2))
        ax.text(0.05, 0.955, "Demostración paso a paso", fontsize=11,
                fontweight="bold", color="#17202e", va="center")
        ax.text(0.05, 0.917, "Método del factor integrante", fontsize=8.6,
                color=GRIS, va="center")

        pasos = pasos_demostracion(m, d)
        y = 0.862
        salto = 0.73 / len(pasos)
        for titulo, formula in pasos:
            ax.text(0.05, y, titulo, fontsize=8.1, color=AZUL, fontweight="bold",
                    va="center")
            ax.text(0.07, y - salto * 0.47, formula, fontsize=9.6, color="#17202e",
                    va="center")
            y -= salto

        # Conclusión, con el color del semáforo
        t_col = d["t_colapso"]
        if t_col is not None:
            conclusion = (f"CONCLUSIÓN: el búfer se llena en t = {num(t_col, 1)} s.\n"
                          "El servidor colapsa por OOM.")
        elif d["diverge"]:
            conclusion = ("CONCLUSIÓN: la memoria crece sin límite.\n"
                          "Tarde o temprano el servidor colapsa.")
        elif d["estado"] == "EN RIESGO":
            conclusion = ("CONCLUSIÓN: no colapsa, pero el techo\n"
                          "queda muy cerca del límite (poco margen).")
        else:
            conclusion = (f"CONCLUSIÓN: la memoria se mantiene por\n"
                          f"debajo de {texto_gb(m.A_max)}. El servidor nunca colapsa.")
        color_texto = "#a87f00" if d["color"] == AMARILLO else d["color"]
        ax.text(0.05, 0.06, conclusion, fontsize=8.6, fontweight="bold",
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
        try:
            self._avanzar_animacion()
        except Exception as error:      # si algo falla, se muestra en la ventana
            self.temporizador.stop()
            self.animando = False
            self.b_animar.label.set_text("Animar")
            self.txt_aviso.set_text(f"Error en la animación: {error}")
            self.fig.canvas.draw_idle()

    def _avanzar_animacion(self):
        if self.t_marcador is None:     # p. ej. se cambió de escenario a mitad
            self.t_marcador = 0.0
        m = self.modelo()
        T = float(self.s_T.val)
        fin = T
        t_col = m.tiempo_de_colapso(T)
        if t_col is not None:
            fin = min(T, t_col)         # el servidor "muere" al colapsar
        self.t_marcador += T / 120.0
        if self.t_marcador >= fin:
            self.t_marcador = fin
            self.temporizador.stop()
            self.animando = False
            self.b_animar.label.set_text("Animar")
        self.ax_mem.clear()
        dibujar_memoria(self.ax_mem, m, T, t_marcador=self.t_marcador)
        self.fig.canvas.draw_idle()

    def restablecer(self, _evento=None):
        """Vuelve a los datos del documento (Q=180, k=0.04, A0=512, 8 GB)."""
        self._detener_animacion()
        self.ocupado = True
        self.radio.set_active(0)          # escenario constante
        self.escenario = "constante"
        self._sliders_del_escenario()
        self.s_RAM.set_val(RAM_GB_INICIAL)
        self._ajustar_slider_A0()
        for slider, valor in ((self.s_Q, Q_INICIAL), (self.s_P, P_INICIAL),
                              (self.s_A0, A0_INICIAL), (self.s_T, T_INICIAL)):
            slider.set_val(valor)
        self.ocupado = False
        self.actualizar()

    def guardar(self, _evento=None):
        """Guarda la ventana completa como imagen PNG de alta resolución."""
        carpeta = carpeta_graficas()
        m = self.modelo()
        nombre = (f"simulador_{m.clave_elegida}_Q{num(m.Q0)}_k{num(m.P0, 3)}"
                  f"_{num(m.A_max / MB_POR_GB)}GB.png")
        self.fig.savefig(os.path.join(carpeta, nombre), dpi=250,
                         facecolor=self.fig.get_facecolor())
        self.txt_aviso.set_text(f"Guardada en graficas/{nombre}")
        self.fig.canvas.draw_idle()


# ============================================================================
# 6. MODOS SIN VENTANA: exportar gráficas y verificar resultados
# ============================================================================
def carpeta_graficas():
    """Carpeta 'graficas' junto a este archivo (se crea si no existe)."""
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficas")
    os.makedirs(carpeta, exist_ok=True)
    return carpeta


def modelo_documento(clave="constante", A_max=A_MAX):
    """Modelo con los datos del documento y los parámetros por defecto."""
    p1, p2 = valores_por_defecto(clave)
    return Modelo(clave, Q_INICIAL, P_INICIAL, A0_INICIAL, A_max, p1, p2)


def exportar_graficas():
    """
    Genera imágenes PNG en alta resolución (300 dpi) listas para pegar en el
    documento o en las diapositivas.
    """
    carpeta = carpeta_graficas()
    m = modelo_documento()
    T = T_INICIAL
    trabajos = [
        ("1_memoria_vs_tiempo.png", dibujar_memoria, (9, 5.2)),
        ("2_campo_de_direcciones.png", dibujar_campo, (7.5, 5)),
        ("3_balance_de_flujos.png", dibujar_flujos, (7.5, 5)),
    ]
    for nombre, funcion, medida in trabajos:
        fig, ax = plt.subplots(figsize=medida)
        funcion(ax, m, T)
        fig.tight_layout()
        fig.savefig(os.path.join(carpeta, nombre), dpi=300)
        plt.close(fig)
        print("  guardada:", nombre)

    # Comparación de niveles de tráfico constante (caso del documento)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    estilo_ejes(ax)
    t = np.linspace(0, 300, 600)
    for q, color, etiqueta in [
        (180, AZUL, "Documento: Q=180, k=0.04  ->  equilibrio 4500 MB (seguro)"),
        (300, AMARILLO, "Hora pico: Q=300, k=0.04  ->  equilibrio 7500 MB (en riesgo)"),
        (400, ROJO, "Sobrecarga: Q=400, k=0.04  ->  equilibrio 10000 MB (colapsa)"),
    ]:
        mod = Modelo("constante", q, P_INICIAL, A0_INICIAL, A_MAX)
        ax.plot(t, mod.solucion(t), color=color, linewidth=2.5, label=etiqueta)
    ax.axhline(A_MAX, color=ROJO, linestyle="--", linewidth=1.6,
               label="Límite de RAM = 8192 MB")
    ax.axhspan(A_MAX, A_MAX * 1.3, color=ROJO, alpha=0.10)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, A_MAX * 1.3)
    ax.set_xlabel("Tiempo  t  [s]")
    ax.set_ylabel("Memoria ocupada  A(t)  [MB]")
    ax.set_title("Comparación de niveles de tráfico", fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "4_comparacion_escenarios.png"), dpi=300)
    plt.close(fig)
    print("  guardada: 4_comparacion_escenarios.png")

    # Los cinco escenarios generalizados con los datos por defecto
    fig, ax = plt.subplots(figsize=(9, 5.2))
    estilo_ejes(ax)
    T6 = 300
    colores = [AZUL, VERDE, NARANJA, "#7b4fb8", ROJO]
    for clave, color in zip(ORDEN_ESCENARIOS, colores):
        mod = modelo_documento(clave)
        tt = mod.malla(T6, 1500)
        ax.plot(tt, np.minimum(mod.solucion(tt), A_MAX * 1.4), color=color,
                linewidth=2.3, label=mod.nombre)
    ax.axhline(A_MAX, color=ROJO, linestyle="--", linewidth=1.6,
               label="Límite de RAM = 8192 MB (8 GB)")
    ax.axhspan(A_MAX, A_MAX * 1.3, color=ROJO, alpha=0.10)
    ax.set_xlim(0, T6)
    ax.set_ylim(0, A_MAX * 1.3)
    ax.set_xlabel("Tiempo  t  [s]")
    ax.set_ylabel("Memoria ocupada  A(t)  [MB]")
    ax.set_title("Modelo generalizado: los cinco escenarios (Q0=180, k=0.04, A0=512)",
                 fontweight="bold", loc="left", fontsize=11)
    ax.legend(fontsize=8.3, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "6_escenarios_generalizados.png"), dpi=300)
    plt.close(fig)
    print("  guardada: 6_escenarios_generalizados.png")

    # Capturas de la aplicación completa
    app = SimuladorMemoria()
    app.fig.savefig(os.path.join(carpeta, "5_aplicacion_completa.png"), dpi=200,
                    facecolor=app.fig.get_facecolor())
    print("  guardada: 5_aplicacion_completa.png")
    app.radio.set_active(ORDEN_ESCENARIOS.index("senoidal"))
    app.fig.savefig(os.path.join(carpeta, "7_aplicacion_rafagas.png"), dpi=200,
                    facecolor=app.fig.get_facecolor())
    print("  guardada: 7_aplicacion_rafagas.png")
    plt.close(app.fig)
    print("\nListo. Las imágenes están en:", carpeta)


def verificar():
    """
    Comprueba que:
      1. Con los datos del documento se obtienen sus mismos resultados.
      2. En cada escenario, la fórmula cerrada, la fórmula general numérica y
         RK4 dan la misma curva.
      3. El límite de 128 GB funciona.
    """
    todo_bien = True
    m = modelo_documento()
    d = m.diagnostico(T_INICIAL)
    print("=" * 70)
    print(" 1. DATOS DEL DOCUMENTO (Q=180, k=0.04, A0=512, RAM = 8 GB)")
    print("=" * 70)
    comprobaciones = [
        ("Memoria de equilibrio Q/P [MB]", d["techo"], 4500.0),
        ("Constante de integración C", m.A0 - m.Q0 / m.P0, -3988.0),
        ("Porcentaje de uso de RAM [%]", d["porcentaje"], 54.931640625),
        ("Memoria libre [MB]", d["libre"], 3692.0),
        ("A(0) [MB]", float(m.solucion(0.0)), 512.0),
    ]
    for nombre, obtenido, esperado in comprobaciones:
        correcto = abs(obtenido - esperado) < 1e-6
        todo_bien &= correcto
        print(f" {nombre:<34} {obtenido:>14.6f}   esperado {esperado:>12.6f}"
              f"   {'OK' if correcto else 'ERROR'}")
    sin_colapso = d["t_colapso"] is None
    todo_bien &= sin_colapso
    print(f" {'¿Colapsa?':<34} {'No' if sin_colapso else 'Sí':>14}   esperado "
          f"{'No':>12}   {'OK' if sin_colapso else 'ERROR'}")

    print("\n" + "=" * 70)
    print(" 2. ESCENARIOS: fórmula cerrada vs fórmula general vs RK4 (0-200 s)")
    print("=" * 70)
    print(f" {'Escenario':<33}{'cerrada-general':>16}{'general-RK4':>14}"
          f"{'colapso [s]':>12}")
    for clave in ORDEN_ESCENARIOS:
        mod = modelo_documento(clave)
        t_rk, A_rk = mod.runge_kutta4(200, 4000)          # h = 0.05 s
        A_gen = mod.solucion_general(mod.malla(200, 4001))
        A_gen = np.interp(t_rk, mod.malla(200, 4001), A_gen)
        exacta = mod.solucion_cerrada(t_rk)
        e1 = np.max(np.abs(exacta - A_gen)) if exacta is not None else None
        e2 = np.max(np.abs(A_gen - A_rk))
        # Tolerancia: 0.05 % de la escala de la curva. (En el pulso, Q da saltos
        # y RK4 pierde algo de precisión justo en ellos; por eso es el mayor.)
        tolerancia = 5e-4 * np.max(np.abs(A_rk))
        correcto = (e1 is None or e1 < 0.05) and e2 < tolerancia
        todo_bien &= correcto
        t_col = mod.tiempo_de_colapso(200)
        print(f" {mod.nombre:<33}"
              f"{('%.2e MB' % e1) if e1 is not None else 'sin fórmula':>16}"
              f"{'%.2e MB' % e2:>14}"
              f"{(num(t_col, 1) if t_col is not None else 'no'):>12}"
              f"   {'OK' if correcto else 'ERROR'}")

    print("\n" + "=" * 70)
    print(" 3. LÍMITE DE RAM DE 128 GB (131072 MB)")
    print("=" * 70)
    m128 = Modelo("constante", 180, 0.04, 512, 128 * MB_POR_GB)
    d128 = m128.diagnostico(200)
    print(f" Datos del documento con 128 GB: techo {num(d128['techo'])} MB = "
          f"{num(d128['porcentaje'], 3)} %  -> {d128['estado']}")
    todo_bien &= d128["estado"] == "SISTEMA SEGURO"
    m_sat = Modelo("constante", 1500, 0.01, 512, 128 * MB_POR_GB)
    t_sat = m_sat.tiempo_de_colapso(600)
    esperado = -100 * np.log((131072 - 150000) / (512 - 150000))
    correcto = abs(t_sat - esperado) < 1e-9
    todo_bien &= correcto
    print(f" Q=1500, k=0.01 con 128 GB: equilibrio 150000 MB, colapso en "
          f"t = {num(t_sat, 2)} s   {'OK' if correcto else 'ERROR'}")

    print("\n" + "=" * 70)
    print(" 4. FORMA GENERALIZADA  dA/dt + kA = Q  (casos k > 0, k = 0, k < 0)")
    print("=" * 70)
    q_max = m.P0 * m.A_max
    correcto = abs(q_max - 327.68) < 1e-9
    todo_bien &= correcto
    print(f" Tráfico máximo soportado k·A_max = {num(q_max)} MB/s (esperado 327.68)"
          f"   {'OK' if correcto else 'ERROR'}")
    for k in (0.04, 0.0, -0.01):
        mk = Modelo("constante", 180, k, 512, A_MAX)
        t_rk, A_rk = mk.runge_kutta4(60, 1200)
        e = np.max(np.abs(mk.solucion_cerrada(t_rk) - A_rk))
        t_exacto = mk.tiempo_de_colapso(200)
        # el instante de colapso también se busca recorriendo la solución numérica
        tt = np.linspace(0, 400, 400001)
        AA = mk.solucion_general(tt)
        cruce = np.nonzero(AA >= A_MAX)[0]
        t_num = tt[cruce[0]] if cruce.size else None
        iguales = (t_exacto is None and t_num is None) or (
            t_exacto is not None and t_num is not None and abs(t_exacto - t_num) < 0.01)
        correcto = e < 1e-4 and iguales
        todo_bien &= correcto
        print(f" k = {num(k, 3):>5}: fórmula vs RK4 {e:.1e} MB, colapso exacto "
              f"{(num(t_exacto, 2) + ' s') if t_exacto is not None else 'no':>9}, numérico "
              f"{(num(t_num, 2) + ' s') if t_num is not None else 'no':>9}   {'OK' if correcto else 'ERROR'}")

    print("\n RESULTADO:", "TODO CORRECTO" if todo_bien else "HAY DIFERENCIAS")
    print("=" * 70)
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
