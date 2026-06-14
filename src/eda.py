"""Notebook 02 — análisis exploratorio de datos (EDA).

Implementa los 7 pasos del análisis exploratorio:
    1. Tasas de no-asistencia globales y por variable clave.
    2. Matriz de correlación (vs Showed_up y vs SMS_received).
    3. Análisis de la asignación de SMS por subgrupo.
    4. Inversión de signo del SMS sin ajustar — núcleo motivacional de IPW.
    5. Distribución de lead_time y valores extremos.
    6. Análisis a nivel paciente (primeras vs repetidas, número de citas).
    7. Sidecar de metadatos con los números clave para defensa.

Convenciones:
    - Funciones y nombres locales en español por coherencia con NB01.
    - Narrativa de los plots en español (títulos, ejes, leyendas).
    - Identificadores de pandas/numpy/seaborn permanecen en inglés.
    - Las figuras se persisten en outputs/figuras/ con prefijo nb02_*.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


# Variables que entran a la matriz de correlación. Se excluyen identificadores
# (PatientId, AppointmentID), columnas de fechas (ScheduledDay, AppointmentDay)
# y la categórica cruda Neighbourhood (representada por neighbourhood_encoded).
COLS_CORRELACION_NUMERICAS = [
    "Age", "lead_time", "prior_appointment_count", "prior_noshow_rate",
    "neighbourhood_encoded", "scheduled_weekday", "scheduled_month",
    "appointment_weekday", "appointment_month", "comorbidity_count",
    "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap",
    "chronic_flag", "is_first_visit",
    "SMS_received", "Showed_up",
]

# Etiquetas legibles para los plots (ejes y leyendas)
ETIQUETAS_VARIABLES: dict[str, str] = {
    "Age": "edad",
    "lead_time": "días de antelación",
    "lead_time_bin": "antelación (bin)",
    "age_band": "banda de edad",
    "appointment_weekday": "día semana cita",
    "scheduled_weekday": "día semana programación",
    "scheduled_month": "mes programación",
    "appointment_month": "mes cita",
    "comorbidity_count": "nº comorbilidades",
    "chronic_flag": "cualquier comorbilidad",
    "Scholarship": "beca social (Bolsa Família)",
    "Hipertension": "hipertensión",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholismo",
    "Handcap": "discapacidad",
    "SMS_received": "recibió SMS",
    "Showed_up": "asistió",
    "prior_appointment_count": "nº citas previas",
    "prior_noshow_rate": "tasa no-show previa",
    "neighbourhood_encoded": "barrio (frecuencia)",
    "is_first_visit": "primera visita",
    "Gender": "género",
}

# Mapeo numérico → texto para días de la semana (Python: lunes=0)
NOMBRES_DIA_SEMANA = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


# Columnas binarias que `pd.read_csv` interpreta como `bool` al releer
# el output de NB01. Para que groupbys y comparaciones (.loc[0], .loc[1])
# funcionen igual que en el dataframe original de NB01, las casteamos a int.
COLS_BINARIAS = [
    "SMS_received", "Scholarship",
    "Hipertension", "Diabetes", "Alcoholism", "Handcap",
    # Derivadas binarias de NB01 que también queremos como int por
    # simetría — defienden frente a `pd.read_csv` interpretándolas como bool
    # si en algún momento se persisten con tipos distintos.
    "is_first_visit", "chronic_flag",
]


# --------------------------------------------------------------------------
# Estilo y utilidades de figuras
# --------------------------------------------------------------------------


def normalizar_binarios(df: pd.DataFrame) -> pd.DataFrame:
    """Castea las columnas binarias del dataset a int (0/1).

    `pd.read_csv` interpreta valores `True`/`False` como `bool` al recargar
    el output de NB01. Para que groupbys, máscaras y `.loc[0]` / `.loc[1]`
    funcionen como en el dataframe canónico de NB01, las normalizamos a
    enteros. La copia evita mutar el dataframe que el llamador ya tiene
    en memoria.
    """
    df = df.copy()
    convertidas = []
    for c in COLS_BINARIAS:
        if c in df.columns and df[c].dtype == bool:
            df[c] = df[c].astype(int)
            convertidas.append(c)
    if convertidas:
        logger.info("Columnas binarias normalizadas a int: %s", convertidas)
    return df


def configurar_estilo() -> None:
    """Fija un estilo seaborn consistente para todas las figuras del NB02.

    Se llama una vez al inicio del notebook. Tamaño y paleta elegidos para
    que las figuras impresas en el TFG sean legibles a tamaño media página.
    """
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="colorblind",
        font_scale=1.0,
    )
    plt.rcParams.update({
        "figure.dpi": 110,
        # 200 DPI para que las figuras impresas en el TFG y en slides de defensa
        # se vean nítidas a media página (150 es borderline a tamaño impreso).
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })


def guardar_figura(fig: plt.Figure, ruta: str | Path) -> Path:
    """Persiste la figura, garantizando que el directorio existe."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    logger.info("Figura guardada: %s", ruta)
    return ruta


# --------------------------------------------------------------------------
# Tasas de no-asistencia (paso 1 del plan)
# --------------------------------------------------------------------------

def tasa_noshow_global(df: pd.DataFrame) -> dict[str, float]:
    """Devuelve la tasa de asistencia y de no-show del dataset completo."""
    n = len(df)
    n_asistio = int((df["Showed_up"] == 1).sum())
    n_noshow = n - n_asistio
    pct_asistio = n_asistio / n
    pct_noshow = 1 - pct_asistio
    logger.info(
        "Tasa global: %d filas → %d asistieron (%.2f%%), %d no-show (%.2f%%)",
        n, n_asistio, 100 * pct_asistio, n_noshow, 100 * pct_noshow,
    )
    return {
        "n_total": n,
        "n_asistio": n_asistio,
        "n_noshow": n_noshow,
        "pct_asistio": pct_asistio,
        "pct_noshow": pct_noshow,
    }


def tasa_noshow_por_variable(
    df: pd.DataFrame,
    columna: str,
    *,
    orden: Iterable | None = None,
    min_n: int = 0,
) -> pd.DataFrame:
    """Tabla de tasa de no-show por nivel de `columna`.

    Devuelve un DataFrame ordenado con n, pct_asistio, pct_noshow para cada
    valor único de la columna. Útil para inspección numérica antes de los
    plots y para alimentar las celdas de narrativa del notebook.
    """
    grupo = df.groupby(columna, dropna=False, observed=True)
    tabla = pd.DataFrame({
        "n": grupo.size(),
        "pct_asistio": grupo["Showed_up"].mean(),
    })
    tabla["pct_noshow"] = 1 - tabla["pct_asistio"]
    tabla = tabla.loc[tabla["n"] >= min_n]
    if orden is not None:
        tabla = tabla.reindex([v for v in orden if v in tabla.index])
    return tabla


def plot_tasa_noshow_panel(
    df: pd.DataFrame,
    ruta_salida: str | Path,
    *,
    variables: list[str] | None = None,
) -> Path:
    """Panel 2x3 con tasa de no-show por las variables clave del plan §1.

    Variables por defecto: lead_time_bin, age_band, appointment_weekday,
    SMS_received, comorbidity_count, Scholarship.
    """
    if variables is None:
        variables = [
            "lead_time_bin", "age_band", "appointment_weekday",
            "SMS_received", "comorbidity_count", "Scholarship",
        ]

    ordenes: dict[str, list] = {
        "lead_time_bin": ["mismo_dia", "1-7d", "8-14d", "15+d"],
        "age_band": ["0-18", "19-35", "36-55", "56-70", "70+"],
    }
    tasa_global = (1 - df["Showed_up"].mean())

    n_var = len(variables)
    n_cols = 3
    n_rows = (n_var + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, var in zip(axes, variables):
        tabla = tasa_noshow_por_variable(df, var, orden=ordenes.get(var))
        x = [str(v) for v in tabla.index]
        if var == "appointment_weekday":
            x = [NOMBRES_DIA_SEMANA[int(v)] for v in tabla.index]
        bars = ax.bar(x, 100 * tabla["pct_noshow"], color=sns.color_palette()[0])
        ax.axhline(100 * tasa_global, color="grey", linestyle="--", linewidth=1,
                   label=f"global ({100 * tasa_global:.1f}%)")
        ax.set_title(f"No-show por {ETIQUETAS_VARIABLES.get(var, var)}")
        ax.set_ylabel("tasa de no-show (%)")
        # Anotar n para que el lector vea soporte muestral
        for rect, n in zip(bars, tabla["n"]):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.4,
                    f"n={n:,}".replace(",", "."),
                    ha="center", va="bottom", fontsize=8, color="dimgray")
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis="x", rotation=0)

    # Apagar ejes sobrantes si el grid no se llena por completo
    for ax in axes[len(variables):]:
        ax.set_visible(False)

    fig.suptitle("Tasa de no-asistencia por variable (descriptivo, no causal)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return guardar_figura(fig, ruta_salida)


def plot_noshow_top_barrios(
    df: pd.DataFrame, ruta_salida: str | Path, n_top: int = 20,
) -> Path:
    """Barras horizontales con tasa de no-show de los `n_top` barrios más frecuentes."""
    tabla = tasa_noshow_por_variable(df, "Neighbourhood")
    tabla = tabla.sort_values("n", ascending=False).head(n_top)
    tabla = tabla.sort_values("pct_noshow")  # para que las barras suban

    fig, ax = plt.subplots(figsize=(9, 0.35 * len(tabla) + 1.5))
    ax.barh(tabla.index, 100 * tabla["pct_noshow"], color=sns.color_palette()[0])
    ax.axvline(100 * (1 - df["Showed_up"].mean()), color="grey",
               linestyle="--", linewidth=1, label="global")
    ax.set_xlabel("tasa de no-show (%)")
    ax.set_title(f"Tasa de no-asistencia — top {n_top} barrios por volumen")
    for i, (_, fila) in enumerate(tabla.iterrows()):
        ax.text(100 * fila["pct_noshow"] + 0.2, i,
                f"n={int(fila['n']):,}".replace(",", "."),
                va="center", fontsize=8, color="dimgray")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return guardar_figura(fig, ruta_salida)


# --------------------------------------------------------------------------
# Matriz de correlación (paso 2 del plan)
# --------------------------------------------------------------------------

def matriz_correlacion(
    df: pd.DataFrame, columnas: list[str] | None = None,
) -> pd.DataFrame:
    """Matriz de correlación de Pearson entre las variables modelables.

    Para variables binarias la correlación de Pearson coincide con el
    coeficiente phi (continua para Showed_up=1 vs 0 cuando se cruza con
    una variable continua, equivalente a un point-biserial). pandas maneja
    NaN por defecto en modo pairwise — relevante porque prior_noshow_rate
    es NaN para primeras visitas.
    """
    columnas = columnas or COLS_CORRELACION_NUMERICAS
    cols_validas = [c for c in columnas if c in df.columns]
    faltantes = set(columnas) - set(cols_validas)
    if faltantes:
        logger.warning("Columnas ausentes en la matriz de correlación: %s", faltantes)

    sub = df[cols_validas].copy()
    # Handcap en algunas versiones del dataset llega como bool — forzar numérico
    for c in cols_validas:
        if sub[c].dtype == bool:
            sub[c] = sub[c].astype(int)
    return sub.corr(method="pearson")


def plot_correlacion_heatmap(
    corr: pd.DataFrame, ruta_salida: str | Path,
) -> Path:
    """Heatmap completo de la matriz de correlación."""
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        cbar_kws={"label": "Pearson r"},
        linewidths=0.5, linecolor="white",
    )
    ax.set_title("Matriz de correlación — variables candidatas para IPW y XGBoost")
    fig.tight_layout()
    return guardar_figura(fig, ruta_salida)


def plot_correlaciones_dual(
    corr: pd.DataFrame, ruta_salida: str | Path, *, top_n: int = 15,
) -> Path:
    """Comparación lado-a-lado: correlaciones con Showed_up y con SMS_received.

    Justifica la selección de covariables del propensity score: las variables
    asociadas a *ambas* (outcome y tratamiento) son los confounders críticos.
    """
    if "Showed_up" not in corr.index or "SMS_received" not in corr.index:
        raise ValueError(
            "La matriz de correlación debe incluir Showed_up y SMS_received."
        )

    excluidas = {"Showed_up", "SMS_received"}
    candidatos = [c for c in corr.index if c not in excluidas]
    serie_outcome = corr.loc[candidatos, "Showed_up"].sort_values()
    serie_trat = corr.loc[candidatos, "SMS_received"].sort_values()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4, 0.32 * len(candidatos))))
    colores_a = ["tab:red" if v < 0 else "tab:blue" for v in serie_outcome.values]
    ax1.barh(serie_outcome.index, serie_outcome.values, color=colores_a)
    ax1.axvline(0, color="black", linewidth=0.7)
    ax1.set_title("Correlación con Showed_up (asistió=1)")
    ax1.set_xlabel("Pearson r")

    colores_b = ["tab:red" if v < 0 else "tab:blue" for v in serie_trat.values]
    ax2.barh(serie_trat.index, serie_trat.values, color=colores_b)
    ax2.axvline(0, color="black", linewidth=0.7)
    ax2.set_title("Correlación con SMS_received (tratamiento)")
    ax2.set_xlabel("Pearson r")

    fig.suptitle("Selección de confounders: variables asociadas a outcome y tratamiento",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return guardar_figura(fig, ruta_salida)


# --------------------------------------------------------------------------
# Análisis de la asignación de SMS (paso 3 del plan)
# --------------------------------------------------------------------------

def tasa_sms_por_variable(df: pd.DataFrame, columna: str,
                          *, orden: Iterable | None = None) -> pd.DataFrame:
    """Proporción de SMS=1 por nivel de `columna`."""
    grupo = df.groupby(columna, dropna=False, observed=True)
    tabla = pd.DataFrame({
        "n": grupo.size(),
        "pct_sms": grupo["SMS_received"].mean(),
    })
    if orden is not None:
        tabla = tabla.reindex([v for v in orden if v in tabla.index])
    return tabla


def plot_distribucion_sms(df: pd.DataFrame, ruta_salida: str | Path) -> Path:
    """Panel 2x2 mostrando quién recibe SMS por age_band, lead_time_bin,
    Scholarship y top-10 barrios. Motiva por qué el análisis ingenuo está sesgado.
    """
    ordenes = {
        "lead_time_bin": ["mismo_dia", "1-7d", "8-14d", "15+d"],
        "age_band": ["0-18", "19-35", "36-55", "56-70", "70+"],
    }
    variables = ["age_band", "lead_time_bin", "Scholarship"]
    tasa_global_sms = df["SMS_received"].mean()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, var in zip(axes[:3], variables):
        tabla = tasa_sms_por_variable(df, var, orden=ordenes.get(var))
        x = [str(v) for v in tabla.index]
        bars = ax.bar(x, 100 * tabla["pct_sms"], color=sns.color_palette()[1])
        ax.axhline(100 * tasa_global_sms, color="grey", linestyle="--",
                   linewidth=1, label=f"global ({100 * tasa_global_sms:.1f}%)")
        ax.set_title(f"% recepción SMS por {ETIQUETAS_VARIABLES.get(var, var)}")
        ax.set_ylabel("% SMS = 1")
        for rect, n in zip(bars, tabla["n"]):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.4,
                    f"n={n:,}".replace(",", "."),
                    ha="center", va="bottom", fontsize=8, color="dimgray")
        ax.legend(loc="upper right", fontsize=8)

    # Top-10 barrios por volumen
    ax = axes[3]
    tabla = tasa_sms_por_variable(df, "Neighbourhood")
    tabla = tabla.sort_values("n", ascending=False).head(10).sort_values("pct_sms")
    ax.barh(tabla.index, 100 * tabla["pct_sms"], color=sns.color_palette()[1])
    ax.axvline(100 * tasa_global_sms, color="grey", linestyle="--",
               linewidth=1, label="global")
    ax.set_xlabel("% SMS = 1")
    ax.set_title("% recepción SMS — top 10 barrios por volumen")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("¿Quién recibe SMS? Distribución por subgrupo (motiva IPW)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    return guardar_figura(fig, ruta_salida)


# --------------------------------------------------------------------------
# Inversión de signo del SMS (paso 4 del plan — núcleo de la motivación)
# --------------------------------------------------------------------------

def inversion_signo_sms(df: pd.DataFrame) -> dict[str, float]:
    """Calcula las tasas de no-show por brazo de SMS sin ningún ajuste.

    En este dataset se espera que SMS=1 muestre *más* no-show que SMS=0
    (selección: el SMS se mandaba a citas de mayor antelación y mayor
    riesgo basal). NO es un efecto perjudicial: es la motivación literal
    para el IPW estimado en NB04.
    """
    grupo = df.groupby("SMS_received", observed=True)
    serie_noshow = 1 - grupo["Showed_up"].mean()
    n_por_brazo = grupo.size()

    pct_noshow_sms0 = float(serie_noshow.loc[0])
    pct_noshow_sms1 = float(serie_noshow.loc[1])
    diff_pp = pct_noshow_sms1 - pct_noshow_sms0  # positivo → reversal observada

    pct_sms_global = float(df["SMS_received"].mean())

    logger.info(
        "Tasas no-show sin ajustar: SMS=0 → %.2f%%, SMS=1 → %.2f%% (Δ = %+.2f pp)",
        100 * pct_noshow_sms0, 100 * pct_noshow_sms1, 100 * diff_pp,
    )
    if diff_pp > 0:
        logger.info(
            "Inversión de signo CONFIRMADA: los receptores de SMS no asisten "
            "MÁS sin ajustar. Esto es sesgo de selección, no efecto causal — "
            "documentado en NB02 §4 como motivación cualitativa de IPW."
        )
    else:
        logger.warning(
            "Inversión NO observada en este split (Δ = %+.2f pp). Revisar "
            "la sección 4 del EDA antes de redactar la narrativa.",
            100 * diff_pp,
        )

    return {
        "n_sms0": int(n_por_brazo.loc[0]),
        "n_sms1": int(n_por_brazo.loc[1]),
        "pct_sms_global": pct_sms_global,
        "pct_noshow_sms0": pct_noshow_sms0,
        "pct_noshow_sms1": pct_noshow_sms1,
        "diff_pp_sms1_menos_sms0": diff_pp,
    }


def plot_inversion_signo_sms(df: pd.DataFrame, ruta_salida: str | Path) -> Path:
    """Gráfico de barras explicativo con anotaciones de N y % por brazo."""
    info = inversion_signo_sms(df)
    fig, ax = plt.subplots(figsize=(7, 5))

    etiquetas = ["sin SMS (SMS=0)", "con SMS (SMS=1)"]
    valores = [100 * info["pct_noshow_sms0"], 100 * info["pct_noshow_sms1"]]
    ns = [info["n_sms0"], info["n_sms1"]]
    colores = [sns.color_palette()[0], sns.color_palette()[3]]

    bars = ax.bar(etiquetas, valores, color=colores)
    for rect, val, n in zip(bars, valores, ns):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.3,
                f"{val:.2f}%\n(n={n:,})".replace(",", "."),
                ha="center", va="bottom", fontsize=10)

    diff_pp = 100 * info["diff_pp_sms1_menos_sms0"]
    ax.set_ylim(0, max(valores) + 5)
    ax.set_ylabel("tasa de no-show (%)")
    ax.set_title(
        "Comparación SIN AJUSTAR: tasa de no-show por brazo de SMS\n"
        f"Δ = {diff_pp:+.2f} pp — selección, no efecto causal"
    )
    ax.text(0.5, -0.18,
            "Los receptores de SMS son sistemáticamente diferentes "
            "(mayor antelación, mayor riesgo basal).\n"
            "La estimación causal se realiza en NB04 mediante IPW estabilizado.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="dimgray", style="italic")
    fig.tight_layout()
    return guardar_figura(fig, ruta_salida)


# --------------------------------------------------------------------------
# Distribución de lead_time (paso 5 del plan)
# --------------------------------------------------------------------------

def plot_distribucion_lead_time(df: pd.DataFrame, ruta_salida: str | Path) -> Path:
    """Histograma del lead_time + superposición de tasa de no-show por bin diario.

    Recorta visualmente la cola larga a P99 para que el histograma sea legible
    (los valores extremos se documentan numéricamente).
    """
    p99 = float(np.percentile(df["lead_time"], 99))
    # Histograma sólo de la masa principal (lead_time entre 0 y P99). El
    # número exacto de filas con lead_time<0 y >P99 se documenta en el
    # título para que ningún lector piense que el plot oculta filas.
    sub = df.loc[(df["lead_time"] >= 0) & (df["lead_time"] <= p99)].copy()
    n_neg = int((df["lead_time"] < 0).sum())
    n_cola = int((df["lead_time"] > p99).sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel izquierdo: histograma del lead_time
    ax1.hist(sub["lead_time"], bins=range(0, int(p99) + 2), color=sns.color_palette()[0],
             edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("lead_time (días)")
    ax1.set_ylabel("nº de citas")
    ax1.set_title(
        f"Distribución de lead_time (0 ≤ x ≤ P99 = {p99:.0f} días)\n"
        f"n graficado = {len(sub):,}; descartado: {n_neg} con lead_time<0, "
        f"{n_cola} en la cola >P99".replace(",", ".")
    )
    ax1.axvline(0, color="grey", linestyle="--", linewidth=0.8)

    # Panel derecho: tasa de no-show por día de antelación (sólo días con n>=200)
    por_dia = sub.groupby("lead_time").agg(
        n=("Showed_up", "size"), pct_noshow=("Showed_up", lambda s: 1 - s.mean()),
    )
    por_dia = por_dia.loc[por_dia["n"] >= 200]
    ax2.plot(por_dia.index, 100 * por_dia["pct_noshow"],
             marker="o", markersize=3, color=sns.color_palette()[3])
    ax2.axhline(100 * (1 - df["Showed_up"].mean()),
                color="grey", linestyle="--", linewidth=1, label="global")
    ax2.set_xlabel("lead_time (días)")
    ax2.set_ylabel("tasa de no-show (%)")
    ax2.set_title("Tasa de no-show por día de antelación (n ≥ 200)")
    ax2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    return guardar_figura(fig, ruta_salida)


def resumen_lead_time(df: pd.DataFrame) -> dict[str, float]:
    """Estadísticas clave del lead_time: cola larga y bookings del mismo día."""
    s = df["lead_time"]
    return {
        "n_negativo": int((s < 0).sum()),
        "n_mismo_dia": int((s == 0).sum()),
        "pct_mismo_dia": float((s == 0).mean()),
        "mediana": float(s.median()),
        "media": float(s.mean()),
        "p90": float(np.percentile(s, 90)),
        "p99": float(np.percentile(s, 99)),
        "maximo": int(s.max()),
    }


# --------------------------------------------------------------------------
# Análisis a nivel paciente (paso 6 del plan)
# --------------------------------------------------------------------------

def estadisticas_pacientes(df: pd.DataFrame) -> dict[str, Any]:
    """Distribución de número de citas por paciente y no-show primera vs repetida."""
    citas_por_paciente = df.groupby("PatientId").size()
    n_pacientes = int(citas_por_paciente.size)

    noshow_primeras = float(
        1 - df.loc[df["is_first_visit"] == 1, "Showed_up"].mean()
    )
    noshow_repetidas = float(
        1 - df.loc[df["is_first_visit"] == 0, "Showed_up"].mean()
    )

    return {
        "n_pacientes_unicos": n_pacientes,
        "media_citas_por_paciente": float(citas_por_paciente.mean()),
        "mediana_citas_por_paciente": float(citas_por_paciente.median()),
        "p95_citas_por_paciente": float(np.percentile(citas_por_paciente, 95)),
        "max_citas_por_paciente": int(citas_por_paciente.max()),
        "pct_pacientes_una_sola_cita": float((citas_por_paciente == 1).mean()),
        "noshow_primeras_visitas": noshow_primeras,
        "noshow_repetidas": noshow_repetidas,
        "diff_pp_primera_menos_repetida": noshow_primeras - noshow_repetidas,
    }


def plot_analisis_pacientes(df: pd.DataFrame, ruta_salida: str | Path) -> Path:
    """Dos paneles: distribución de citas/paciente y no-show primera vs repetida."""
    citas_por_paciente = df.groupby("PatientId").size()
    info = estadisticas_pacientes(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel izquierdo: distribución del nº de citas por paciente (eje x recortado)
    p95 = int(np.percentile(citas_por_paciente, 95))
    bins = np.arange(0.5, p95 + 1.5, 1)
    ax1.hist(citas_por_paciente, bins=bins, color=sns.color_palette()[0],
             edgecolor="white", linewidth=0.4)
    ax1.set_xlabel("nº de citas por paciente")
    ax1.set_ylabel("nº de pacientes")
    ax1.set_title(
        f"Distribución de citas por paciente (eje recortado a P95={p95})\n"
        f"{info['n_pacientes_unicos']:,} pacientes únicos, "
        f"máx = {info['max_citas_por_paciente']}".replace(",", ".")
    )

    # Panel derecho: no-show primera vs repetida
    etiquetas = ["primera visita", "visita repetida"]
    valores = [100 * info["noshow_primeras_visitas"], 100 * info["noshow_repetidas"]]
    ns = [
        int((df["is_first_visit"] == 1).sum()),
        int((df["is_first_visit"] == 0).sum()),
    ]
    bars = ax2.bar(etiquetas, valores,
                   color=[sns.color_palette()[0], sns.color_palette()[3]])
    for rect, val, n in zip(bars, valores, ns):
        ax2.text(rect.get_x() + rect.get_width() / 2,
                 rect.get_height() + 0.3,
                 f"{val:.2f}%\n(n={n:,})".replace(",", "."),
                 ha="center", va="bottom", fontsize=10)
    diff_pp = 100 * info["diff_pp_primera_menos_repetida"]
    ax2.set_ylabel("tasa de no-show (%)")
    ax2.set_title(f"No-show: primera vs repetida (Δ = {diff_pp:+.2f} pp)")
    ax2.set_ylim(0, max(valores) + 5)

    fig.tight_layout()
    return guardar_figura(fig, ruta_salida)


# --------------------------------------------------------------------------
# Sidecar de metadatos NB02 (paso 7 del plan)
# --------------------------------------------------------------------------

def escribir_metadatos_nb02(
    ruta_json: str | Path,
    *,
    df: pd.DataFrame,
    info_global: Mapping[str, Any],
    info_inversion: Mapping[str, Any],
    info_lead_time: Mapping[str, Any],
    info_pacientes: Mapping[str, Any],
    top_corr_outcome: Mapping[str, float],
    top_corr_tratamiento: Mapping[str, float],
) -> dict[str, Any]:
    """Persiste los números clave de NB02 a JSON para defensa y reproducibilidad.

    Incluye los hallazgos numéricos que la sección de EDA del TFG escrito
    citará textualmente: tasa global, inversión de signo del SMS, mediana
    de lead_time, perfil de pacientes y top correlaciones.
    """
    metadatos: dict[str, Any] = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "n_total": info_global["n_total"],
        "pct_atendieron": info_global["pct_asistio"],
        "pct_noshow_global": info_global["pct_noshow"],

        # Inversión de signo del SMS (hallazgo clave de NB02 §4)
        "sms_pct_global": info_inversion["pct_sms_global"],
        "n_sms0": info_inversion["n_sms0"],
        "n_sms1": info_inversion["n_sms1"],
        "pct_noshow_sms0": info_inversion["pct_noshow_sms0"],
        "pct_noshow_sms1": info_inversion["pct_noshow_sms1"],
        "diff_pp_sms1_menos_sms0": info_inversion["diff_pp_sms1_menos_sms0"],
        "inversion_signo_confirmada": bool(
            info_inversion["diff_pp_sms1_menos_sms0"] > 0
        ),

        # Lead time
        "lead_time_mediana": info_lead_time["mediana"],
        "lead_time_p90": info_lead_time["p90"],
        "lead_time_p99": info_lead_time["p99"],
        "lead_time_maximo": info_lead_time["maximo"],
        "n_lead_time_negativo": info_lead_time["n_negativo"],
        "n_lead_time_mismo_dia": info_lead_time["n_mismo_dia"],
        "pct_mismo_dia": info_lead_time["pct_mismo_dia"],

        # Pacientes
        "n_pacientes_unicos": info_pacientes["n_pacientes_unicos"],
        "media_citas_por_paciente": info_pacientes["media_citas_por_paciente"],
        "p95_citas_por_paciente": info_pacientes["p95_citas_por_paciente"],
        "max_citas_por_paciente": info_pacientes["max_citas_por_paciente"],
        "pct_pacientes_una_sola_cita": info_pacientes["pct_pacientes_una_sola_cita"],
        "noshow_primeras_visitas": info_pacientes["noshow_primeras_visitas"],
        "noshow_repetidas": info_pacientes["noshow_repetidas"],

        # Top correlaciones (informativo). prior_noshow_rate sólo está
        # definido para visitas no-primeras (~28%); pandas usa pairwise
        # deletion en .corr() — el caveat queda persistido para citarse
        # en la sección de metodología/limitaciones del TFG escrito.
        "top_corr_con_showed_up": dict(top_corr_outcome),
        "top_corr_con_sms_received": dict(top_corr_tratamiento),
        "correlaciones_metodo": "pearson",
        "correlaciones_nan_handling": "pairwise (pandas .corr() por defecto)",
        "n_prior_noshow_rate_no_nulos": int(df["prior_noshow_rate"].notna().sum()),

        # Cross-check con NB01 (sidecar auto-contenido para el TFG escrito)
        "n_anomalia_appt_lt_sched": int(
            (df["AppointmentDay"] < df["ScheduledDay"]).sum()
        ),
        "dias_semana_presentes": sorted(
            int(d) for d in df["appointment_weekday"].unique()
        ),
    }

    ruta_json = Path(ruta_json)
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    with ruta_json.open("w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=2, ensure_ascii=False)
    logger.info("Metadatos NB02 guardados en %s", ruta_json)
    return metadatos
