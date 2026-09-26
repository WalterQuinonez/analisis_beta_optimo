#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera las figuras (heatmaps accuracy(beta_hidden, beta_output)) y el PDF
LaTeX con las conclusiones de `estudiar_beta_2layers.py`.

Correr DESPUES de que `resultados_beta_2layers.csv` este completo (o
parcialmente completo: solo grafica/reporta lo que haya).
"""

import os
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(THIS_DIR, "resultados_beta_2layers.csv")
CSV_PATH_TANH = os.path.join(THIS_DIR, "resultados_beta_2layers_tanh.csv")

NOMBRE_CURVA_LEGIBLE = {"lineal": "Curva P/D lineal (a=4567.9)", "no_lineal": "Curva P/D no lineal (a=98.55)"}


def cargar(path=CSV_PATH):
    df = pd.read_csv(path)
    return df


def hacer_heatmap(df_curva, curva_nombre, sufijo="", titulo_extra=""):
    betas_h = np.sort(df_curva["beta_hidden"].unique())
    betas_o = np.sort(df_curva["beta_output"].unique())
    mat = np.full((len(betas_h), len(betas_o)), np.nan)
    idx_h = {b: i for i, b in enumerate(betas_h)}
    idx_o = {b: i for i, b in enumerate(betas_o)}
    for _, row in df_curva.iterrows():
        mat[idx_h[row["beta_hidden"]], idx_o[row["beta_output"]]] = row["acc_mean"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(betas_o)))
    ax.set_xticklabels([f"{b:.0f}" for b in betas_o], rotation=90, fontsize=7)
    ax.set_yticks(range(len(betas_h)))
    ax.set_yticklabels([f"{b:.0f}" for b in betas_h], fontsize=7)
    ax.set_xlabel(r"$\beta_{salida}$")
    ax.set_ylabel(r"$\beta_{oculta}$")
    ax.set_title(NOMBRE_CURVA_LEGIBLE.get(curva_nombre, curva_nombre) + titulo_extra)

    # marcar la diagonal (beta_hidden == beta_output, "un unico beta global")
    n = min(len(betas_h), len(betas_o))
    diag_mask_h, diag_mask_o = [], []
    for i, bh in enumerate(betas_h):
        for j, bo in enumerate(betas_o):
            if abs(bh - bo) < 1e-6:
                diag_mask_h.append(i)
                diag_mask_o.append(j)
    ax.scatter(diag_mask_o, diag_mask_h, marker="x", color="red", s=25, label=r"$\beta_{oculta}=\beta_{salida}$")

    # marcar el maximo global
    i_max, j_max = np.unravel_index(np.nanargmax(mat), mat.shape)
    ax.scatter([j_max], [i_max], marker="*", color="white", edgecolor="black", s=180, label="Máximo global")

    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
    fig.colorbar(im, ax=ax, label="accuracy (media, 3 folds)")
    fig.tight_layout()
    out_path = os.path.join(THIS_DIR, f"heatmap_{curva_nombre}{sufijo}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, betas_h, betas_o, mat


def analizar_curva(df, curva_nombre, sufijo="", titulo_extra=""):
    df_c = df[df["curva"] == curva_nombre].copy()
    out_path, betas_h, betas_o, mat = hacer_heatmap(df_c, curva_nombre, sufijo=sufijo, titulo_extra=titulo_extra)

    # diagonal = "beta unico global"
    diag = df_c[np.isclose(df_c["beta_hidden"], df_c["beta_output"], rtol=1e-6)]
    if len(diag) == 0:
        best_diag = None
    else:
        best_diag = diag.loc[diag["acc_mean"].idxmax()]

    best_global = df_c.loc[df_c["acc_mean"].idxmax()]

    ganancia = None
    if best_diag is not None:
        ganancia = (best_global["acc_mean"] - best_diag["acc_mean"]) * 100  # en puntos porcentuales

    return dict(
        curva=curva_nombre,
        n_combos=len(df_c),
        heatmap_path=out_path,
        best_diag=best_diag,
        best_global=best_global,
        ganancia_pp=ganancia,
    )


LATEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-noshorthands]{babel}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue]{hyperref}
\usepackage{caption}
\usepackage{parskip}
\usepackage[expansion=false]{microtype}
\sloppy

\title{\textbf{$\beta$ por capa vs. $\beta$ global en un crossbar\\
con capa oculta de 100 neuronas (ReLU)}}
\author{Walter Quiñonez}
\date{Reporte generado con Claude Code \textperiodcentered\ 13 de septiembre de 2026}

\begin{document}
\maketitle

\begin{quote}\itshape
Exploración pedida en \texttt{prompt.txt}: para una arquitectura
\texttt{[784, 100, 10]} (capa oculta de 100 neuronas con ReLU, capa de
salida lineal), ¿conviene usar un $\beta$ distinto para la capa oculta y la
de salida, o alcanza con un único $\beta$ global? Se mapeó el accuracy en
una grilla log-espaciada de $(\beta_{oculta}, \beta_{salida})$, para una
curva P/D lineal y una no lineal, con el protocolo mínimo validado en
\texttt{analisis\_epochs\_series\_minimos/} ($k$=3 folds, 8 épocas,
\emph{common random numbers} entre puntos de la grilla). No se modificó
ningún código existente del proyecto.
\end{quote}

\tableofcontents
"""


def generar_tex(resultados, resultados_tanh=None):
    inv_rows_tanh = [
        ("lineal", 10.00, 18087.16, 180871.6, 0.853333, 0.019508),
        ("lineal", 50.00, 3617.43, 180871.6, 0.905517, 0.002710),
        ("lineal", 100.00, 1808.72, 180871.6, 0.897883, 0.005764),
        ("lineal", 425.29, 425.29, 180871.6, 0.947783, 0.002289),
        ("lineal", 2000.00, 90.44, 180871.6, 0.928733, 0.001515),
        ("lineal", 5000.00, 36.17, 180871.6, 0.896950, 0.001398),
        ("no\\_lineal", 10.00, 14785.56, 147855.6, 0.887400, 0.004456),
        ("no\\_lineal", 50.00, 2957.11, 147855.6, 0.875800, 0.025828),
        ("no\\_lineal", 100.00, 1478.56, 147855.6, 0.890233, 0.008304),
        ("no\\_lineal", 384.52, 384.52, 147855.6, 0.933167, 0.002499),
        ("no\\_lineal", 2000.00, 73.93, 147855.6, 0.893983, 0.010480),
        ("no\\_lineal", 5000.00, 29.57, 147855.6, 0.880017, 0.001429),
    ]
    partes = [LATEX_PREAMBLE]

    partes.append(r"\section{Resumen ejecutivo}")
    partes.append(r"\begin{table}[htbp]\centering")
    partes.append(r"\begin{tabular}{@{}lrrr@{}}\toprule")
    partes.append(r"Curva P/D & mejor acc. (beta único) & mejor acc. (beta por capa) & ganancia [pp] \\ \midrule")
    for r in resultados:
        if r["best_diag"] is None:
            continue
        partes.append(
            f"{NOMBRE_CURVA_LEGIBLE.get(r['curva'], r['curva'])} & "
            f"{r['best_diag']['acc_mean']*100:.2f}\\% "
            f"($\\beta$={r['best_diag']['beta_hidden']:.0f}) & "
            f"{r['best_global']['acc_mean']*100:.2f}\\% "
            f"($\\beta_o$={r['best_global']['beta_hidden']:.0f}, $\\beta_s$={r['best_global']['beta_output']:.0f}) & "
            f"{r['ganancia_pp']:.2f} \\\\"
        )
    partes.append(r"\bottomrule\end{tabular}")
    partes.append(r"\caption{Comparación entre el mejor punto sobre la diagonal "
                   r"($\beta_{oculta}=\beta_{salida}$, equivalente a un único $\beta$ global) "
                   r"y el mejor punto de toda la grilla ($\beta$ por capa libre).}")
    partes.append(r"\end{table}")

    for r in resultados:
        nombre = NOMBRE_CURVA_LEGIBLE.get(r["curva"], r["curva"])
        partes.append(f"\\section{{Curva: {nombre}}}")
        partes.append(r"\begin{figure}[htbp]\centering")
        partes.append(f"\\includegraphics[width=0.75\\textwidth]{{{os.path.basename(r['heatmap_path'])}}}")
        partes.append(f"\\caption{{Accuracy medio (3 folds) en función de $(\\beta_{{oculta}}, \\beta_{{salida}})$, "
                       f"{nombre}. La cruz roja marca la diagonal (beta único); la estrella blanca, el máximo global.}}")
        partes.append(r"\end{figure}")

        if r["best_diag"] is not None:
            partes.append(
                f"Mejor punto sobre la diagonal (beta único global): "
                f"$\\beta$={r['best_diag']['beta_hidden']:.1f}, "
                f"acc={r['best_diag']['acc_mean']*100:.2f}\\% "
                f"(SEM {r['best_diag']['acc_sem']*100:.2f}pp).\\\\"
            )
        partes.append(
            f"Mejor punto global (beta por capa libre): "
            f"$\\beta_{{oculta}}$={r['best_global']['beta_hidden']:.1f}, "
            f"$\\beta_{{salida}}$={r['best_global']['beta_output']:.1f}, "
            f"acc={r['best_global']['acc_mean']*100:.2f}\\% "
            f"(SEM {r['best_global']['acc_sem']*100:.2f}pp).\\\\"
        )
        if r["ganancia_pp"] is not None:
            partes.append(f"Ganancia del mejor beta-por-capa sobre el mejor beta-único: "
                           f"\\textbf{{{r['ganancia_pp']:.2f} puntos porcentuales}}.")

    partes.append(r"\section{Por qué la grilla es (casi) simétrica en el producto $\beta_{oculta}\cdot\beta_{salida}$}")
    partes.append(
        r"En los heatmaps de las Secciones 2 y 3 se observa que puntos muy alejados de la diagonal "
        r"(splits muy desparejos entre las 2 capas) pueden dar el mismo accuracy que un punto cercano "
        r"a la diagonal, siempre que el \emph{producto} $\beta_{oculta}\cdot\beta_{salida}$ sea "
        r"parecido. Esto no es casualidad: tiene una razón matemática exacta."
    )
    partes.append(
        r"El forward pass de este modelo (\texttt{MemDNN.forward}, sin bias) es "
        r"$h=\mathrm{ReLU}(\beta_{oculta}\,(x W_1))$, $\mathrm{logits}=\beta_{salida}\,(h W_2)$. "
        r"La ReLU es homogénea de grado 1 para escalares positivos: $\mathrm{ReLU}(c\,z)=c\,\mathrm{ReLU}(z)$ "
        r"para todo $c>0$. Por lo tanto, para cualquier $c>0$, reemplazar "
        r"$(\beta_{oculta},\beta_{salida}) \to (c\,\beta_{oculta},\, \beta_{salida}/c)$ deja "
        r"\textbf{exactamente} el mismo $h$ escalado por $c$ y el mismo logit final "
        r"(el $c$ se cancela): $h'=c\,h$, $\mathrm{logits}'=(\beta_{salida}/c)(c\,h\,W_2)=\mathrm{logits}$. "
        r"Como la regla de actualización física de este proyecto "
        r"(\texttt{Nmanhattan\_vectorizado}) sólo usa el \emph{signo} del gradiente para decidir "
        r"potenciación/depresión (nunca su magnitud), y ese signo se deriva de los logits/softmax "
        r"—que son idénticos bajo el reemplazo anterior—, la trayectoria completa de entrenamiento "
        r"(qué pulso se aplica en cada paso) debería depender, en aritmética exacta, "
        r"\textbf{sólo del producto $\beta_{oculta}\cdot\beta_{salida}$, no de cómo se reparte entre las 2 capas}."
    )
    partes.append(
        r"\subsection{Verificación empírica: ¿la invariancia se sostiene en la práctica?}"
    )
    partes.append(
        r"Se corrieron splits muy distintos con el producto fijado en $\beta_{pred}^2$ de cada curva "
        r"(fuera de la grilla 10$\times$10 principal, para no reciclar los mismos resultados):"
    )
    partes.append(r"\begin{table}[htbp]\centering\begin{tabular}{@{}lrrrr@{}}\toprule")
    partes.append(r"Curva & $\beta_{oculta}$ & $\beta_{salida}$ & producto & accuracy (SEM) \\ \midrule")
    inv_rows = [
        ("lineal", 425.29, 425.29, 180871.6, 0.945633, 0.002087),
        ("lineal", 50.00, 3617.43, 180871.6, 0.927033, 0.013608),
        ("lineal", 100.00, 1808.72, 180871.6, 0.927033, 0.013608),
        ("lineal", 2000.00, 90.44, 180871.6, 0.920317, 0.015890),
        ("no\\_lineal", 384.52, 384.52, 147855.6, 0.742033, 0.026692),
        ("no\\_lineal", 2000.00, 73.93, 147855.6, 0.729683, 0.021020),
        ("no\\_lineal", 50.00, 2957.11, 147855.6, 0.740250, 0.028299),
        ("no\\_lineal", 100.00, 1478.56, 147855.6, 0.740250, 0.028299),
    ]
    for c, bh, bo, prod, acc, sem in inv_rows:
        partes.append(f"{c} & {bh:.2f} & {bo:.2f} & {prod:.1f} & {acc*100:.2f}\\% ({sem*100:.2f}pp) \\\\")
    partes.append(r"\bottomrule\end{tabular}")
    partes.append(r"\caption{Splits muy distintos con el mismo producto $\beta_{oculta}\cdot\beta_{salida}$. "
                   r"Los pares (50, prod/50) y (100, prod/100) dan accuracy \emph{idéntico} (hasta el 6.º "
                   r"decimal, SEM incluido) en ambas curvas; el split ``natural'' "
                   r"($\beta_{oculta}=\beta_{salida}=\sqrt{\text{prod}}$) y el split extremo (2000, prod/2000) "
                   r"difieren de esos hasta $\sim$2.5pp.}")
    partes.append(r"\end{table}")
    partes.append(
        r"Estas diferencias de hasta $\sim$2.5pp son del mismo orden que el propio SEM de la medición "
        r"(hasta 2.83pp con sólo 3 folds), así que \textbf{no hay evidencia concluyente de que la "
        r"invariancia se rompa de forma sistemática} más allá del ruido estadístico normal del "
        r"protocolo mínimo usado. Es plausible que la causa de que algunos pares SÍ coincidan "
        r"exactamente y otros no sea la naturaleza discreta de la actualización física (Manhattan, "
        r"basada en el signo del gradiente): pequeñas diferencias de redondeo en float32 al computar "
        r"$c\cdot\beta_{oculta}\cdot(xW_1)$ para $c$ muy distintos podrían, en principio, voltear el "
        r"signo de algún pulso cerca de un umbral y desviar la trayectoria — pero con sólo 8 épocas y "
        r"3 folds esta hipótesis no se puede distinguir del ruido de muestreo con este experimento."
    )

    partes.append(r"\section{Extensión analítica: ¿vale lo mismo con tangente hiperbólica?}")
    partes.append(
        r"Todo lo anterior depende de una propiedad muy específica de la ReLU: ser homogénea de "
        r"grado 1 ($\mathrm{ReLU}(cz)=c\,\mathrm{ReLU}(z)$ para $c>0$). La tangente hiperbólica NO "
        r"tiene esa propiedad, así que vale la pena rehacer la cuenta para "
        r"$h=\tanh(\beta_{oculta}(xW_1))$ en vez de $h=\mathrm{ReLU}(\beta_{oculta}(xW_1))$, "
        r"manteniendo $\mathrm{logits}=\beta_{salida}(hW_2)$ y sin bias, igual que antes. "
        r"\textbf{Esta sección es puramente analítica: no se corrió ninguna simulación nueva con "
        r"tanh}, sólo se repite el argumento matemático de la Sección 4 para ver si sigue valiendo."
    )
    partes.append(r"\subsection{La reparametrización $(\beta_{oculta},\beta_{salida})\to(c\,\beta_{oculta},\beta_{salida}/c)$ ya no es exacta}")
    partes.append(
        r"Sea $z_1=\beta_{oculta}(xW_1)$ (pre-activación de la capa oculta). Con ReLU, escalar "
        r"$\beta_{oculta}\to c\,\beta_{oculta}$ escala $z_1\to c\,z_1$ y, por homogeneidad, "
        r"$h\to c\,h$ exactamente — ese factor $c$ se cancela después al dividir "
        r"$\beta_{salida}$ por $c$. Con tanh, en cambio:"
    )
    partes.append(r"$$h' = \tanh(c\,z_1) \quad\text{mientras que}\quad c\cdot\tanh(z_1) = c\cdot h,$$")
    partes.append(
        r"y en general $\tanh(cz)\neq c\,\tanh(z)$ para $c\neq 1$ (la única función homogénea de "
        r"grado 1 y acotada es la nula). Por lo tanto "
        r"$\mathrm{logits}' = (\beta_{salida}/c)\,\tanh(c z_1)\,W_2 \neq \beta_{salida}\,\tanh(z_1)\,W_2 "
        r"= \mathrm{logits}$ salvo en el caso trivial $c=1$: \textbf{la invariancia exacta de la "
        r"Sección 4 no se sostiene con tanh}. Con tanh, el resultado de la red depende en general de "
        r"$\beta_{oculta}$ y $\beta_{salida}$ por separado, no sólo de su producto."
    )
    partes.append(r"\subsection{Dos regímenes límite}")
    partes.append(
        r"El tamaño del error de la aproximación se entiende mejor mirando los dos extremos de "
        r"$z_1=\beta_{oculta}(xW_1)$:"
    )
    partes.append(r"\begin{itemize}")
    partes.append(
        r"\item \textbf{$\beta_{oculta}$ chico (régimen lineal, $|z_1|\ll 1$):} usando "
        r"$\tanh(z)=z-\tfrac{z^3}{3}+O(z^5)$,"
    )
    partes.append(r"$$\frac{\tanh(cz_1)}{c} = z_1 - \frac{c^2 z_1^3}{3} + O(z_1^5) "
                   r"\approx \tanh(z_1) - \frac{(c^2-1)}{3}z_1^3,$$")
    partes.append(
        r"es decir, la invariancia exacta de ReLU se recupera de forma \emph{aproximada}, con un "
        r"error relativo de orden $|c^2-1|\,z_1^2/3$: chico si $z_1$ es chico o si $c\approx 1$ "
        r"(cerca de la diagonal), y creciente a medida que $\beta_{oculta}$ empuja $z_1$ lejos de 0. "
        r"Esto es consistente con que la ReLU y la tanh son parecidas cerca del origen "
        r"($\tanh(z)\approx z\approx\mathrm{ReLU}(z)$ para $z>0$ chico)."
    )
    partes.append(
        r"\item \textbf{$\beta_{oculta}$ grande (régimen saturado, $|z_1|\gg 1$):} "
        r"$\tanh(z_1)\to\mathrm{sign}(z_1)\in\{-1,+1\}$, una función de grado \emph{0} "
        r"(no de grado 1): al seguir aumentando $\beta_{oculta}$, $h$ deja de crecer y se estanca en "
        r"$\pm 1$ salvo muy cerca del borde de decisión $xW_1=0$. En este régimen $\beta_{oculta}$ ya "
        r"no actúa como una ganancia multiplicativa de $h$ (como en ReLU), sino como un "
        r"\emph{umbral/nitidez} de la frontera de decisión de cada neurona oculta, mientras que "
        r"$\beta_{salida}$ sigue siendo la única ganancia lineal de los logits. Son dos roles "
        r"cualitativamente distintos y, por construcción, ya no intercambiables vía ningún $c$."
    )
    partes.append(r"\end{itemize}")
    partes.append(r"\subsection{Conclusión analítica}")
    partes.append(
        r"A diferencia de ReLU, con tanh en la capa oculta \textbf{no hay razón matemática para "
        r"esperar que un único $\beta$ global sea tan bueno como optimizar $\beta_{oculta}$ y "
        r"$\beta_{salida}$ por separado}. La aproximación por producto sigue siendo razonable sólo "
        r"mientras $\beta_{oculta}$ mantenga la capa oculta en su tramo aproximadamente lineal "
        r"($|z_1|\lesssim 1$); apenas $\beta_{oculta}$ crece lo suficiente para saturar la tanh, el "
        r"argumento se rompe cualitativamente y $\beta_{oculta}$ pasa a controlar algo distinto "
        r"(la nitidez del umbral de cada neurona) de lo que controla $\beta_{salida}$ (la ganancia de "
        r"los logits). Esto sugiere que, si se repitiera el experimento numérico de las Secciones "
        r"2--3 con tanh en vez de ReLU, sería esperable ver una ganancia real (no sólo ruido "
        r"estadístico) de usar $\beta$ por capa, sobre todo en la zona de la grilla donde "
        r"$\beta_{oculta}$ es grande — algo que este reporte no verifica numéricamente, sólo lo "
        r"predice a partir de la forma funcional de la tanh."
    )

    # ==========================================================================
    # Verificacion numerica con tanh (usa `resultados` de la activacion tanh,
    # pasado en resultados_tanh; ver generar_tex())
    # ==========================================================================
    if resultados_tanh is not None:
        partes.append(r"\section{Verificación numérica con tanh}")
        partes.append(
            r"Se repitió \textbf{exactamente} el experimento de las Secciones 2--3 (mismas curvas "
            r"P/D, misma grilla log 10$\times$10, mismo protocolo mínimo, mismas semillas) "
            r"reemplazando \texttt{MemDNN} por \texttt{MemDNNTanh} (subclase nueva en "
            r"\texttt{mem\_dnn\_tanh.py} que sólo cambia \texttt{ReLU}$\to$\texttt{tanh} en la capa "
            r"oculta; no se modificó \texttt{MemCrossbarClass\_beta\_por\_capa.py})."
        )

        partes.append(r"\subsection{Grilla completa: diagonal vs. máximo global}")
        partes.append(r"\begin{table}[htbp]\centering\begin{tabular}{@{}lrrr@{}}\toprule")
        partes.append(r"Curva P/D & mejor acc. (beta único) & mejor acc. (beta por capa) & ganancia [pp] \\ \midrule")
        for r in resultados_tanh:
            if r["best_diag"] is None:
                continue
            partes.append(
                f"{NOMBRE_CURVA_LEGIBLE.get(r['curva'], r['curva'])} & "
                f"{r['best_diag']['acc_mean']*100:.2f}\\% "
                f"($\\beta$={r['best_diag']['beta_hidden']:.0f}) & "
                f"{r['best_global']['acc_mean']*100:.2f}\\% "
                f"($\\beta_o$={r['best_global']['beta_hidden']:.0f}, $\\beta_s$={r['best_global']['beta_output']:.0f}) & "
                f"{r['ganancia_pp']:.2f} \\\\"
            )
        partes.append(r"\bottomrule\end{tabular}")
        partes.append(r"\caption{Con tanh, misma comparación que el Cuadro 1 (ReLU).}")
        partes.append(r"\end{table}")
        partes.append(
            r"A primera vista esta ``ganancia'' (0.06pp lineal, 0.67pp no lineal) es \textbf{parecida "
            r"o incluso menor} que la de ReLU (0.09pp y 0.87pp) — contrario a lo que predice la "
            r"Sección 5. Esto es un artefacto de comparar sólo 2 puntos (el mejor de la diagonal y el "
            r"mejor de toda la grilla): si el óptimo verdadero ya cae cerca de la diagonal para ambas "
            r"activaciones, esa comparación puntual no alcanza a ver cuánto varía el accuracy con el "
            r"split \emph{a igual producto}. La Sección 6.2 hace esa prueba directamente."
        )

        for r in resultados_tanh:
            nombre = NOMBRE_CURVA_LEGIBLE.get(r["curva"], r["curva"])
            partes.append(r"\begin{figure}[htbp]\centering")
            partes.append(f"\\includegraphics[width=0.7\\textwidth]{{{os.path.basename(r['heatmap_path'])}}}")
            partes.append(f"\\caption{{Accuracy medio (3 folds) con tanh en la capa oculta, {nombre}.}}")
            partes.append(r"\end{figure}")

        partes.append(r"\subsection{Prueba directa: mismo producto, splits muy distintos (incluye saturación)}")
        partes.append(
            r"Se fijó el producto $\beta_{oculta}\cdot\beta_{salida}=\beta_{pred}^2$ de cada curva "
            r"(igual que en la Sección 4.1 para ReLU) y se probaron splits desde muy chico "
            r"($\beta_{oculta}=10$, capa oculta casi lineal) hasta muy grande "
            r"($\beta_{oculta}=5000$, capa oculta muy saturada), pasando por el split ``natural'' "
            r"$\beta_{oculta}=\beta_{salida}=\sqrt{\text{prod}}$:"
        )
        partes.append(r"\begin{table}[htbp]\centering\begin{tabular}{@{}lrrrr@{}}\toprule")
        partes.append(r"Curva & $\beta_{oculta}$ & $\beta_{salida}$ & producto & accuracy (SEM) \\ \midrule")
        for c, bh, bo, prod, acc, sem in inv_rows_tanh:
            partes.append(f"{c} & {bh:.2f} & {bo:.2f} & {prod:.1f} & {acc*100:.2f}\\% ({sem*100:.2f}pp) \\\\")
        partes.append(r"\bottomrule\end{tabular}")
        partes.append(r"\caption{Con tanh, a producto fijo el accuracy varía fuertemente con el split "
                       r"(spread $\sim$9.4pp en la curva lineal, $\sim$5.7pp en la no lineal), muy por "
                       r"encima del SEM de cada punto (0.14--2.6pp) y del spread visto con ReLU a "
                       r"producto fijo ($\le$2.5pp, dentro del ruido).}")
        partes.append(r"\end{table}")
        partes.append(
            r"El patrón es consistente en las 2 curvas: el accuracy es máximo cerca del split "
            r"``natural'' ($\beta_{oculta}=\beta_{salida}=\sqrt{\text{prod}}$) y \textbf{cae en ambos "
            r"extremos} — tanto cuando $\beta_{oculta}$ es demasiado chico (capa oculta casi lineal, "
            r"$\beta_{salida}$ enorme) como cuando es demasiado grande (capa oculta saturada, "
            r"$\beta_{salida}$ minúsculo). Esto es exactamente lo que predice la Sección 5.2: en "
            r"ninguno de los 2 extremos el par $(\beta_{oculta},\beta_{salida})$ es intercambiable "
            r"con el del centro, porque $\tanh$ deja de comportarse como una ganancia lineal en "
            r"cuanto $|z_1|$ se aleja de 0."
        )
        partes.append(
            r"\textbf{Esta es la evidencia numérica que confirma la predicción analítica de la "
            r"Sección 5}: a diferencia de ReLU (donde el mismo tipo de prueba dio diferencias "
            r"$\le$2.5pp, indistinguibles del ruido), con tanh el split SÍ importa, con una magnitud "
            r"de varios puntos porcentuales — comparable al efecto de acertar el orden de magnitud "
            r"del producto (Sección 3 de este documento)."
        )

    partes.append(r"\section{Conclusiones}")
    partes.append(r"\begin{enumerate}")
    partes.append(
        r"\item \textbf{La ganancia de usar $\beta$ distinto por capa es chica en las dos curvas, y "
        r"comparable al ruido de la medición.} Curva lineal: 0.09pp (mejor global "
        r"vs. mejor con $\beta$ único); curva no lineal: 0.87pp. Ambos valores son del orden del SEM "
        r"típico de la grilla (0.1--3pp con el protocolo mínimo de 3 folds)."
    )
    partes.append(
        r"\item \textbf{Esto tiene una razón matemática, no es sólo un accidente de estas 2 curvas.} "
        r"Con una capa oculta ReLU y sin bias, el forward pass (y por lo tanto, en aritmética exacta, "
        r"toda la dinámica de entrenamiento con la regla Manhattan basada en el signo del gradiente) "
        r"depende sólo del producto $\beta_{oculta}\cdot\beta_{salida}$, nunca de cómo se reparte entre "
        r"las 2 capas. Un único $\beta$ global $B$ ya cubre, vía $B^2$, el mismo rango de productos que "
        r"cualquier búsqueda independiente por capa con el mismo rango de búsqueda por eje — por eso "
        r"no hay casi nada que ganar buscando 2 valores en vez de 1."
    )
    partes.append(
        r"\item \textbf{Lo que sí importa mucho es acertar el orden de magnitud del producto (el "
        r"``$\beta$ efectivo'' de la red).} Dentro de la misma grilla, mover el producto lejos del "
        r"óptimo cuesta decenas de puntos porcentuales (p.ej. curva no lineal: 92.5\% en el óptimo vs. "
        r"61.6\% en el extremo superior de la grilla) — un efecto un orden de magnitud mayor que el de "
        r"la partición entre capas."
    )
    partes.append(
        r"\item \textbf{Recomendación práctica:} para esta arquitectura (una capa oculta ReLU, sin "
        r"bias) alcanza con optimizar un único $\beta$ global (p.ej. con "
        r"\texttt{search\_best\_beta\_mejorado}, ya existente en el proyecto) en vez de agregar una "
        r"segunda dimensión de búsqueda; el costo de no hacerlo (quedarse cerca de la diagonal en vez "
        r"del óptimo global de la grilla) es, en el peor caso medido acá, de menos de 1 punto "
        r"porcentual de accuracy."
    )
    partes.append(
        r"\item \textbf{Limitación:} esta conclusión es específica de una arquitectura de \emph{una "
        r"sola} capa oculta ReLU sin bias. Con más de una capa oculta, o con bias, la invariancia "
        r"exacta del punto 2 ya no se cumple igual (agregar bias rompe la homogeneidad de la ReLU), y "
        r"el resultado podría no generalizar sin repetir este experimento."
    )
    partes.append(
        r"\item \textbf{Con tanh en vez de ReLU, la conclusión SÍ cambia — y esto ya está verificado "
        r"numéricamente (Sección 6), no sólo predicho.} $\tanh$ no es homogénea de grado 1, así que la "
        r"invariancia exacta por producto de ReLU no se cumple. La comparación cruda diagonal-vs-global "
        r"sobre la grilla 10$\times$10 (Sección 6.1) no lo deja ver (da una ganancia parecida a la de "
        r"ReLU, 0.06--0.67pp), pero la prueba directa a producto fijo (Sección 6.2) sí: variar el split "
        r"entre $\beta_{oculta}=10$ y $\beta_{oculta}=5000$ manteniendo el producto constante cambia el "
        r"accuracy en $\sim$9.4pp (curva lineal) y $\sim$5.7pp (curva no lineal) — muy por encima del "
        r"SEM de cada punto y del $\le$2.5pp (ruido) visto con la misma prueba en ReLU. El patrón "
        r"coincide con la predicción: el accuracy es máximo cerca del split ``natural'' "
        r"($\beta_{oculta}\approx\beta_{salida}$) y cae hacia ambos extremos (capa oculta demasiado "
        r"lineal o demasiado saturada). \textbf{Conclusión: con tanh en la capa oculta sí conviene "
        r"buscar $\beta$ por capa, a diferencia de ReLU.}"
    )
    partes.append(r"\end{enumerate}")

    partes.append(r"\end{document}")
    return "\n\n".join(partes)


def compilar_pdf(tex_path):
    for _ in range(2):  # 2 pasadas por la tabla de contenidos
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(tex_path)],
            cwd=THIS_DIR, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )


def main():
    df = cargar()
    resultados = [analizar_curva(df, c) for c in sorted(df["curva"].unique())]

    resultados_tanh = None
    if os.path.exists(CSV_PATH_TANH):
        df_tanh = cargar(CSV_PATH_TANH)
        resultados_tanh = [
            analizar_curva(df_tanh, c, sufijo="_tanh", titulo_extra=" (tanh)")
            for c in sorted(df_tanh["curva"].unique())
        ]

    tex = generar_tex(resultados, resultados_tanh=resultados_tanh)
    tex_path = os.path.join(THIS_DIR, "reporte_beta_capa_oculta.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"TeX escrito en {tex_path}")

    compilar_pdf(tex_path)
    print("PDF compilado.")


if __name__ == "__main__":
    main()
