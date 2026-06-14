"""Modelo Monte Carlo del valor económico recuperado y pricing basado en valor.

Simula el valor mensual que una clínica privada española recupera al reducir
las inasistencias con recordatorios, y deriva tres tarifas (fijo + variable)
ancladas a los percentiles P10/P50/P90 de la distribución simulada.

El parámetro de efecto es una **reducción relativa del no-show** informada por
la evidencia experimental (Gurol-Urganci et al., 2013; Robotham et al., 2016),
no la estimación causal propia (no identificable, ver NB04). Por eso la fórmula
multiplica `tasa_noshow_basal × reduccion_relativa`: el parámetro es relativo
por definición, no un efecto ya expresado en puntos porcentuales absolutos.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_ITER: int = 10_000
SEED_MONTECARLO: int = 2026
DPI_FIGURAS: int = 200


@dataclass
class ParametrosMonteCarlo:
    """Rangos de los parámetros del Monte Carlo (triangular: min, moda, max).

    Datos verificados (junio 2026): valor de consulta y coste del SMS.
    Supuestos declarados: volumen mensual, margen de contribución y los
    parámetros comerciales de aiXtensa.
    """
    # Efecto: reducción relativa del no-show (literatura experimental)
    reduccion_min: float = 0.18      # IC sup. RR no-show Robotham 2016 (0,82)
    reduccion_moda: float = 0.25     # Robotham 2016, "25% less likely to no-show"
    reduccion_max: float = 0.335     # Cochrane 2013, 32,2% → 21,4% entre brazos

    # Tasa basal de no-show (Hernández-García 2018 = 12,5%; rango sectorial)
    basal_min: float = 0.10
    basal_moda: float = 0.125
    basal_max: float = 0.20

    # Valor de la consulta en € (tarifas verificadas: 60, 100, 100, 160)
    valor_min: float = 60.0
    valor_moda: float = 100.0
    valor_max: float = 160.0

    # Coste del SMS en € (Esendex / LabsMobile, tramos 5k-25k)
    sms_min: float = 0.034
    sms_moda: float = 0.045
    sms_max: float = 0.096

    # Volumen mensual de citas (SUPUESTO declarado, no citado)
    volumen_min: float = 200.0
    volumen_max: float = 800.0

    # Margen de contribución por consulta recuperada (SUPUESTO declarado)
    margen_min: float = 0.60
    margen_moda: float = 0.75
    margen_max: float = 0.90

    # Fracción de pacientes a los que se envía el recordatorio
    share_targeted: float = 1.0      # 1.0 = SMS uniforme a toda la agenda

    # Parámetros comerciales de aiXtensa (SUPUESTOS declarados)
    fixed_pct: float = 0.20          # fijo = 20% del percentil de valor neto
    operational_floor: float = 50.0  # mínimo mensual (comunicación + soporte)
    variable_conservador: float = 0.10
    variable_recomendado: float = 0.15
    variable_premium: float = 0.20


def simular(
    params: ParametrosMonteCarlo,
    n_iter: int = N_ITER,
    seed: int = SEED_MONTECARLO,
) -> pd.DataFrame:
    """Ejecuta la simulación y devuelve un DataFrame con una fila por iteración.

    Por iteración: citas tratadas = volumen × share; citas recuperadas =
    tratadas × basal × reducción_relativa; valor bruto = recuperadas × valor ×
    margen; coste = tratadas × coste_sms; valor neto = bruto − coste.
    """
    rng = np.random.default_rng(seed)

    reduccion = rng.triangular(params.reduccion_min, params.reduccion_moda,
                               params.reduccion_max, n_iter)
    basal = rng.triangular(params.basal_min, params.basal_moda,
                           params.basal_max, n_iter)
    valor = rng.triangular(params.valor_min, params.valor_moda,
                           params.valor_max, n_iter)
    sms = rng.triangular(params.sms_min, params.sms_moda, params.sms_max, n_iter)
    volumen = rng.uniform(params.volumen_min, params.volumen_max, n_iter)
    margen = rng.triangular(params.margen_min, params.margen_moda,
                            params.margen_max, n_iter)

    tratadas = volumen * params.share_targeted
    # Efecto relativo × basal = reducción absoluta de no-shows. NO es el ATE
    # absoluto de NB04: aquí el parámetro es una reducción relativa de literatura.
    recuperadas = tratadas * basal * reduccion
    valor_bruto = recuperadas * valor * margen
    coste_comunicacion = tratadas * sms
    valor_neto = valor_bruto - coste_comunicacion

    logger.info(
        "Monte Carlo: %d iteraciones. Valor neto mensual P50=%.0f€ "
        "(P10=%.0f€, P90=%.0f€).",
        n_iter, np.median(valor_neto),
        np.quantile(valor_neto, 0.10), np.quantile(valor_neto, 0.90),
    )
    return pd.DataFrame({
        "reduccion_relativa": reduccion,
        "tasa_basal": basal,
        "valor_consulta": valor,
        "coste_sms": sms,
        "volumen_mensual": volumen,
        "margen": margen,
        "citas_recuperadas": recuperadas,
        "valor_bruto": valor_bruto,
        "coste_comunicacion": coste_comunicacion,
        "valor_neto": valor_neto,
    })


def resumen_percentiles(valor_neto: np.ndarray) -> dict[str, float]:
    """Percentiles y media de la distribución de valor neto mensual (€)."""
    return {
        "p10": float(np.quantile(valor_neto, 0.10)),
        "p25": float(np.quantile(valor_neto, 0.25)),
        "p50": float(np.quantile(valor_neto, 0.50)),
        "p75": float(np.quantile(valor_neto, 0.75)),
        "p90": float(np.quantile(valor_neto, 0.90)),
        "media": float(np.mean(valor_neto)),
        "pct_negativo": float((valor_neto < 0).mean()),
    }


def derivar_tarifas(
    percentiles: dict[str, float], params: ParametrosMonteCarlo,
) -> pd.DataFrame:
    """Tres planes (fijo + variable) anclados a P10/P50/P90 del valor neto.

    Fijo = fixed_pct × percentil (con suelo operativo para el Conservador).
    Variable = % del valor recuperado realizado.
    """
    filas = [
        ("Conservador", percentiles["p10"], params.variable_conservador,
         max(params.fixed_pct * percentiles["p10"], params.operational_floor)),
        ("Recomendado", percentiles["p50"], params.variable_recomendado,
         params.fixed_pct * percentiles["p50"]),
        ("Premium", percentiles["p90"], params.variable_premium,
         params.fixed_pct * percentiles["p90"]),
    ]
    return pd.DataFrame([
        {
            "plan": nombre,
            "percentil_valor_neto_eur": round(percentil, 2),
            "cuota_fija_eur_mes": round(fijo, 2),
            "componente_variable_pct": int(var * 100),
        }
        for nombre, percentil, var, fijo in filas
    ])


def plot_distribucion_valor(
    valor_neto: np.ndarray, percentiles: dict[str, float], ruta_fig: str | Path,
) -> None:
    """Histograma del valor neto mensual con P10/P50/P90 marcados."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(valor_neto, bins=60, color="#1f77b4", alpha=0.75, edgecolor="white")
    for etiqueta, clave, color in (
        ("P10", "p10", "#d62728"), ("P50 (mediana)", "p50", "black"),
        ("P90", "p90", "#2ca02c"),
    ):
        ax.axvline(percentiles[clave], color=color, ls="--", lw=1.6,
                   label=f"{etiqueta} = {percentiles[clave]:,.0f} €")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Valor neto mensual recuperado (€)")
    ax.set_ylabel("Frecuencia (10.000 iteraciones)")
    ax.set_title("Distribución simulada del valor mensual recuperado por clínica")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_fig, dpi=DPI_FIGURAS, bbox_inches="tight")
    plt.close(fig)
    logger.info("Histograma de valor guardado en %s.", ruta_fig)


def persistir_sidecar(metadata: dict[str, Any], ruta: str | Path) -> None:
    """Sidecar JSON con parámetros, percentiles y tarifas derivadas."""
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Sidecar guardado en %s.", ruta)
