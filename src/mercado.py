"""Parámetros de mercado español que alimentan el Monte Carlo (Notebook 06).

Formaliza, en una tabla versionada y con fuente, los parámetros económicos del
modelo de pricing: valor de la consulta, coste del SMS, contexto de inasistencia
en España y los anclajes del efecto. Cada cifra verificable se traza a su fuente,
URL y fecha de acceso; los parámetros sin fuente citable se marcan como supuesto.

La recogida es manual y documental: los precios publicados de clínicas privadas y
de pasarelas de SMS se consultaron directamente en sus tarifas públicas (no se
extraen de forma programática), lo que mantiene el notebook reproducible aunque
las webs cambien. La triangulación de varias fuentes por parámetro acota rangos
en lugar de fijar un único valor puntual.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

FECHA_ACCESO: str = "2026-06-14"
DPI_FIGURAS: int = 200

# --- Parámetros del Monte Carlo: distribución, rango y trazabilidad ----------
# Una fila por parámetro del modelo del Notebook 06. `tipo` distingue las cifras
# verificadas en fuente de los supuestos declarados (sin fuente citable).
PARAMETROS_MONTECARLO: list[dict[str, Any]] = [
    {
        "parametro": "reduccion_relativa_noshow",
        "descripcion": "Reducción relativa del no-show por recordatorio",
        "distribucion": "triangular",
        "min": 0.18, "moda": 0.25, "max": 0.335,
        "unidad": "proporción",
        "tipo": "verificado",
        "fuente": "Robotham 2016 (RR no-show 0,75 [0,68–0,82]); Gurol-Urganci 2013 pub3 (32,2%→21,4%)",
    },
    {
        "parametro": "tasa_basal_noshow",
        "descripcion": "Tasa basal de inasistencia de la clínica",
        "distribucion": "triangular",
        "min": 0.10, "moda": 0.125, "max": 0.20,
        "unidad": "proporción",
        "tipo": "verificado",
        "fuente": "Hernández-García 2018 (12,5%, referencia contextual); rango sectorial",
    },
    {
        "parametro": "valor_consulta",
        "descripcion": "Valor económico de una consulta privada",
        "distribucion": "triangular",
        "min": 60.0, "moda": 100.0, "max": 160.0,
        "unidad": "€",
        "tipo": "verificado",
        "fuente": "IMDA (160 €); Clínica Buenavista (60/100 €)",
    },
    {
        "parametro": "coste_sms",
        "descripcion": "Coste unitario del SMS",
        "distribucion": "triangular",
        "min": 0.034, "moda": 0.045, "max": 0.096,
        "unidad": "€",
        "tipo": "verificado",
        "fuente": "Esendex (0,034–0,096 €); LabsMobile (0,045 €)",
    },
    {
        "parametro": "volumen_mensual",
        "descripcion": "Volumen mensual de citas de la clínica",
        "distribucion": "uniforme",
        "min": 200.0, "moda": np.nan, "max": 800.0,
        "unidad": "citas/mes",
        "tipo": "supuesto",
        "fuente": "Supuesto declarado (sensibilidad sobre el tamaño de la clínica)",
    },
    {
        "parametro": "margen_contribucion",
        "descripcion": "Margen de contribución por consulta recuperada",
        "distribucion": "triangular",
        "min": 0.60, "moda": 0.75, "max": 0.90,
        "unidad": "proporción",
        "tipo": "supuesto",
        "fuente": "Supuesto declarado (estructura de coste de servicios sanitarios)",
    },
]

# --- Evidencia de precios: observaciones puntuales con fuente ----------------
# Precios publicados que sustentan el rango triangular del valor de consulta y
# del coste del SMS. Sirven para el gráfico y para la tabla de mercado del TFG.
EVIDENCIA_PRECIOS: list[dict[str, Any]] = [
    {
        "categoria": "consulta", "proveedor": "Clínica Buenavista (Madrid)",
        "servicio": "Fisioterapia (valoración)", "precio_eur": 60.0,
        "url": "https://clinicabuenavista.com/servicios/precios/",
    },
    {
        "categoria": "consulta", "proveedor": "Clínica Buenavista (Madrid)",
        "servicio": "Ginecología", "precio_eur": 100.0,
        "url": "https://clinicabuenavista.com/servicios/precios/",
    },
    {
        "categoria": "consulta", "proveedor": "Clínica Buenavista (Madrid)",
        "servicio": "Urología", "precio_eur": 100.0,
        "url": "https://clinicabuenavista.com/servicios/precios/",
    },
    {
        "categoria": "consulta", "proveedor": "IMDA",
        "servicio": "Dermatología (primera consulta)", "precio_eur": 160.0,
        "url": "https://www.imda.es/tarifas/",
    },
    {
        "categoria": "sms", "proveedor": "Esendex",
        "servicio": "Pack prepago 50.000 SMS", "precio_eur": 0.034,
        "url": "https://www.esendex.es/precios/",
    },
    {
        "categoria": "sms", "proveedor": "LabsMobile",
        "servicio": "25.000 SMS a España", "precio_eur": 0.045,
        "url": "https://www.labsmobile.com/es/blog/cuanto-puede-costar-una-campana-de-sms-marketing",
    },
    {
        "categoria": "sms", "proveedor": "Esendex",
        "servicio": "Pack prepago 500 SMS", "precio_eur": 0.096,
        "url": "https://www.esendex.es/precios/",
    },
]

# --- Anclajes del efecto: literatura experimental ----------------------------
# El efecto del recordatorio se ancla en evidencia publicada, no en la estimación
# causal propia (no identificable de forma estable, Notebook 04).
FUENTES_EFECTO: list[dict[str, Any]] = [
    {
        "fuente": "Robotham et al. (2016), BMJ Open",
        "metrica": "RR no-show 0,75 [0,68–0,82]; −25% de inasistencia",
        "rol": "Ancla MODA (0,25) y MIN (0,18); meta-análisis más reciente sobre no-show",
        "url": "https://doi.org/10.1136/bmjopen-2016-012116",
    },
    {
        "fuente": "Gurol-Urganci et al. (2013), Cochrane CD007458.pub3",
        "metrica": "Asistencia 67,8%→78,6%; no-show 32,2%→21,4%; RR asistencia 1,14",
        "rol": "Ancla MAX (0,335); reducción relativa entre brazos",
        "url": "https://doi.org/10.1002/14651858.CD007458.pub3",
    },
    {
        "fuente": "Guy et al. (2012), Health Services Research",
        "metrica": "OR asistencia 1,48 [1,23–1,72] (8 ECA)",
        "rol": "Comprobación de coherencia; no es ancla de la triangular",
        "url": "https://doi.org/10.1111/j.1475-6773.2011.01342.x",
    },
    {
        "fuente": "Hernández-García et al. (2018), J Healthc Qual Res",
        "metrica": "Absentismo 12,5% (consulta externa, Zaragoza)",
        "rol": "Referencia contextual de la tasa basal española (moda 0,125)",
        "url": "https://doi.org/10.1016/j.cali.2017.12.006",
    },
]


def configurar_estilo() -> None:
    """Estilo de plots coherente con los demás notebooks."""
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": DPI_FIGURAS,
        "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })


def tabla_parametros() -> pd.DataFrame:
    """Tabla versionada de los parámetros del Monte Carlo con su trazabilidad."""
    df = pd.DataFrame(PARAMETROS_MONTECARLO)
    df["fecha_acceso"] = FECHA_ACCESO
    return df


def tabla_evidencia_precios() -> pd.DataFrame:
    """Precios publicados (consulta y SMS) que sustentan los rangos triangulares."""
    df = pd.DataFrame(EVIDENCIA_PRECIOS)
    df["fecha_acceso"] = FECHA_ACCESO
    return df


def tabla_fuentes_efecto() -> pd.DataFrame:
    """Anclajes del efecto del recordatorio (literatura experimental)."""
    df = pd.DataFrame(FUENTES_EFECTO)
    df["fecha_acceso"] = FECHA_ACCESO
    return df


def plot_precios_consulta(
    evidencia: pd.DataFrame, params: pd.DataFrame, ruta_fig: str | Path,
) -> None:
    """Precios de consulta observados con el rango triangular (min/moda/max)."""
    cons = evidencia[evidencia["categoria"] == "consulta"].sort_values("precio_eur")
    fila = params.set_index("parametro").loc["valor_consulta"]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    etiquetas = cons["proveedor"] + "\n" + cons["servicio"]
    ax.bar(etiquetas, cons["precio_eur"], color="#1f77b4", alpha=0.8,
           edgecolor="white")
    for x, (_, r) in enumerate(cons.iterrows()):
        ax.text(x, r["precio_eur"] + 2, f"{r['precio_eur']:.0f} €",
                ha="center", va="bottom", fontsize=9)
    for valor, nombre, color in (
        (fila["min"], "mín.", "#2ca02c"),
        (fila["moda"], "moda", "black"),
        (fila["max"], "máx.", "#d62728"),
    ):
        ax.axhline(valor, color=color, ls="--", lw=1.3,
                   label=f"Triangular — {nombre} {valor:.0f} €")
    ax.set_ylabel("Precio de consulta (€)")
    ax.set_title("Precios de consulta privada publicados y rango del Monte Carlo")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta_fig, dpi=DPI_FIGURAS, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico de precios de consulta guardado en %s.", ruta_fig)


def reconciliar_con_nb06(
    params: pd.DataFrame, ruta_nb06_meta: str | Path,
) -> pd.DataFrame:
    """Comprueba que los rangos documentados coinciden con los que usó el NB06.

    Lee el sidecar del Notebook 06 y contrasta min/moda/max de cada parámetro
    triangular. Devuelve la tabla de comprobación y lanza si hay discrepancia.
    """
    with open(ruta_nb06_meta, encoding="utf-8") as f:
        meta = json.load(f)
    p06 = meta["parametros"]

    # parámetro documentado -> prefijo de las claves en el sidecar de NB06
    mapa = {
        "reduccion_relativa_noshow": "reduccion",
        "tasa_basal_noshow": "basal",
        "valor_consulta": "valor",
        "coste_sms": "sms",
        "margen_contribucion": "margen",
    }
    filas = []
    for parametro, prefijo in mapa.items():
        doc = params.set_index("parametro").loc[parametro]
        for stat in ("min", "moda", "max"):
            clave = f"{prefijo}_{stat}"
            documentado = float(doc[stat])
            usado = float(p06[clave])
            filas.append({
                "parametro": parametro, "estadistico": stat,
                "documentado": documentado, "usado_nb06": usado,
                "coincide": np.isclose(documentado, usado),
            })
    # Volumen es uniforme (min/max)
    for stat, clave in (("min", "volumen_min"), ("max", "volumen_max")):
        doc = params.set_index("parametro").loc["volumen_mensual"]
        filas.append({
            "parametro": "volumen_mensual", "estadistico": stat,
            "documentado": float(doc[stat]), "usado_nb06": float(p06[clave]),
            "coincide": np.isclose(float(doc[stat]), float(p06[clave])),
        })
    tabla = pd.DataFrame(filas)
    if not tabla["coincide"].all():
        discrepantes = tabla[~tabla["coincide"]]
        raise ValueError(
            f"Parámetros de mercado no coinciden con el NB06:\n{discrepantes}")
    logger.info("Reconciliación con NB06: %d parámetros coinciden.", len(tabla))
    return tabla


def persistir_tabla(df: pd.DataFrame, ruta: str | Path) -> None:
    """Guarda una tabla de parámetros/evidencia en CSV."""
    df.to_csv(ruta, index=False)
    logger.info("Tabla guardada en %s (%d filas).", ruta, len(df))


def persistir_sidecar(metadata: dict[str, Any], ruta: str | Path) -> None:
    """Sidecar JSON con el resumen de parámetros y fuentes."""
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Sidecar guardado en %s.", ruta)
