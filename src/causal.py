"""Inferencia causal del efecto del SMS sobre la asistencia (IPW estabilizado y ATO).

Estima el efecto de `SMS_received` sobre `Showed_up` en puntos porcentuales
absolutos, sobre el dataset completo. El propensity score es una logística con
términos cuadráticos en las continuas; `Neighbourhood` entra con frequency
encoding. Si la positividad falla (ESS/n < 0.5), se pasa a overlap weights /
estimand ATO (Li, Morgan & Zaslavsky, 2018). La incertidumbre se obtiene con
cluster bootstrap por `PatientId` (los pacientes se repiten).

Signo: con `Showed_up=1` (asistió), un efecto positivo = el SMS se asocia con
mayor asistencia.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# =========================================================================
# Constantes
# =========================================================================

# Continuas que entran con expansión cuadrática (z + z²).
COVARIABLES_CONTINUAS: list[str] = [
    "lead_time",
    "Age",
    "prior_noshow_rate",
]

# Covariables que entran linealmente. No incluye el tratamiento ni
# scheduled_hour (constante en el dataset).
COVARIABLES_LINEALES: list[str] = [
    "gender_F",
    "neighbourhood_encoded",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "comorbidity_count",
    "chronic_flag",
    "scheduled_weekday",
    "scheduled_month",
    "appointment_weekday",
    "appointment_month",
    "prior_appointment_count",
    "is_first_visit",
]

COL_TRATAMIENTO: str = "SMS_received"
COL_OUTCOME: str = "Showed_up"
COL_CLUSTER: str = "PatientId"

# Etiquetas legibles para love-plot, overlap y demás plots en español.
ETIQUETAS_COVARIABLES: dict[str, str] = {
    "lead_time": "antelación (días)",
    "lead_time_sq": "antelación² (z²)",
    "Age": "edad",
    "Age_sq": "edad² (z²)",
    "prior_noshow_rate": "tasa no-show previa",
    "prior_noshow_rate_sq": "tasa no-show previa² (z²)",
    "gender_F": "género (F=1)",
    "neighbourhood_encoded": "barrio (frecuencia)",
    "Scholarship": "beca Bolsa Família",
    "Hipertension": "hipertensión",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholismo",
    "Handcap": "discapacidad",
    "comorbidity_count": "nº comorbilidades",
    "chronic_flag": "cualquier comorbilidad",
    "scheduled_weekday": "día semana programación",
    "scheduled_month": "mes programación",
    "appointment_weekday": "día semana cita",
    "appointment_month": "mes cita",
    "prior_appointment_count": "nº citas previas observadas",
    "is_first_visit": "primera visita observada",
    "lead_1_3": "antelación 1-3 días",
    "lead_4_7": "antelación 4-7 días",
    "lead_8_30": "antelación 8-30 días",
    "lead_30plus": "antelación >30 días",
}

# Cluster bootstrap por PatientId.
N_BOOTSTRAP: int = 1000
SEED_BOOTSTRAP: int = 2026

# Trimming pre-especificado: si ESS/n < 0.5 en algún brazo, usar trimmed 1/99.
TRIM_ESS_UMBRAL: float = 0.5
TRIM_ROBUSTEZ: list[tuple[float, float]] = [(1.0, 99.0), (5.0, 95.0)]

# Logística del propensity. C alto = regularización débil (con ~107k filas no
# hace falta más).
LOG_REG_C: float = 1.0
LOG_REG_MAX_ITER: int = 2000
LOG_REG_SOLVER: str = "lbfgs"

DPI_FIGURAS: int = 200


# =========================================================================
# Estilo
# =========================================================================

def configurar_estilo() -> None:
    """Estilo de plots coherente con los demás notebooks."""
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": DPI_FIGURAS,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


# =========================================================================
# Carga y verificación
# =========================================================================

def cargar_dataset_completo(ruta: str | Path) -> pd.DataFrame:
    """Carga el dataset limpio completo (el IPW no usa el split temporal)."""
    df = pd.read_csv(ruta, parse_dates=["ScheduledDay", "AppointmentDay"])
    logger.info(
        "full_clean cargado: %d filas, %d columnas, %d pacientes únicos.",
        len(df), df.shape[1], df[COL_CLUSTER].nunique(),
    )
    return df


def verificar_codificacion_showed_up(df: pd.DataFrame) -> None:
    """Comprueba que Showed_up=1 significa asistió antes de promediar el outcome."""
    pct_atendieron = float((df[COL_OUTCOME] == 1).mean())
    if not 0.65 < pct_atendieron < 0.90:
        raise ValueError(
            f"Showed_up tiene una tasa de asistencia inesperada "
            f"({pct_atendieron:.1%}); verificar codificación antes del IPW."
        )
    logger.info(
        "Showed_up verificado: %.2f%% asistencia (esperado ≈80%%).",
        pct_atendieron * 100,
    )


def verificar_codificacion_tratamiento(df: pd.DataFrame) -> dict[str, Any]:
    """Prevalencia del tratamiento; `p_tratamiento` es P(T=1) para los pesos."""
    sms = df[COL_TRATAMIENTO]
    if sms.dtype == bool:
        sms = sms.astype(int)
    n_trat = int((sms == 1).sum())
    n_ctrl = int((sms == 0).sum())
    p_t = n_trat / (n_trat + n_ctrl)
    logger.info(
        "SMS_received: %d tratados (%.1f%%), %d controles (%.1f%%).",
        n_trat, p_t * 100, n_ctrl, (1 - p_t) * 100,
    )
    return {
        "p_tratamiento": float(p_t),
        "n_tratados": n_trat,
        "n_controles": n_ctrl,
    }


# =========================================================================
# Construcción de la matriz de propensidad
# =========================================================================

@dataclass
class EstadisticasEstandarizacion:
    """Medias y SD de las continuas, calculadas una vez y reusadas en cada
    iteración del bootstrap para que los z² sean comparables entre resamples.
    """
    medias: dict[str, float]
    desviaciones: dict[str, float]

    def aplicar(self, serie: pd.Series, columna: str) -> pd.Series:
        return (serie - self.medias[columna]) / self.desviaciones[columna]


def _calcular_estadisticas_continuas(
    df: pd.DataFrame, continuas: list[str] = COVARIABLES_CONTINUAS
) -> EstadisticasEstandarizacion:
    """Media y SD de las continuas tras imputar prior_noshow_rate=0."""
    medias, desv = {}, {}
    for col in continuas:
        serie = df[col].copy()
        if col == "prior_noshow_rate":
            serie = serie.fillna(0.0)
        medias[col] = float(serie.mean())
        desv[col] = float(serie.std(ddof=0))
    return EstadisticasEstandarizacion(medias=medias, desviaciones=desv)


# Bins de lead_time para la spec de sensibilidad. lead_0 (mismo día) es la
# referencia tras drop_first.
LEAD_TIME_BINS: list[float] = [-0.001, 0.5, 3.5, 7.5, 30.5, np.inf]
LEAD_TIME_BIN_LABELS: list[str] = [
    "lead_0", "lead_1_3", "lead_4_7", "lead_8_30", "lead_30plus",
]


def construir_matriz_propensidad(
    df: pd.DataFrame,
    estadisticas: EstadisticasEstandarizacion | None = None,
    lead_time_spec: str = "z_squared",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, EstadisticasEstandarizacion]:
    """Construye `(X, T, Y, estadisticas)` para el IPW / ATO.

    `lead_time_spec="z_squared"` (default) mete lead_time como z + z²;
    `"categorical"` lo discretiza en 5 bins (para el análisis de sensibilidad).
    En ambos casos se imputa prior_noshow_rate a 0, se estandarizan las
    continuas y se castean las binarias a int.
    """
    if lead_time_spec not in {"z_squared", "categorical"}:
        raise ValueError(f"lead_time_spec desconocido: {lead_time_spec!r}")

    df = df.copy()

    df["gender_F"] = (df["Gender"] == "F").astype(int)

    binarias = [
        "Scholarship", "Hipertension", "Diabetes",
        "Alcoholism", "Handcap", "chronic_flag", "is_first_visit",
        COL_TRATAMIENTO,
    ]
    for col in binarias:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    df["prior_noshow_rate"] = df["prior_noshow_rate"].fillna(0.0)

    # Las medias/SD se calculan siempre sobre las tres continuas, para que
    # cambiar lead_time_spec entre llamadas no las recalcule.
    if estadisticas is None:
        estadisticas = _calcular_estadisticas_continuas(df)

    if lead_time_spec == "z_squared":
        continuas_efectivas = COVARIABLES_CONTINUAS
    else:  # categorical: Age y prior_noshow_rate siguen como z+z²
        continuas_efectivas = ["Age", "prior_noshow_rate"]

    bloques: list[pd.DataFrame] = []
    for col in continuas_efectivas:
        z = estadisticas.aplicar(df[col], col).rename(col)
        z_sq = (z ** 2).rename(f"{col}_sq")
        bloques.append(pd.concat([z, z_sq], axis=1))
    X_continuas = pd.concat(bloques, axis=1) if bloques else pd.DataFrame(index=df.index)

    X_lineales = df[COVARIABLES_LINEALES].copy()
    if lead_time_spec == "categorical":
        lead_bin = pd.cut(
            df["lead_time"], bins=LEAD_TIME_BINS, labels=LEAD_TIME_BIN_LABELS,
        )
        dummies = pd.get_dummies(lead_bin, prefix="", prefix_sep="", drop_first=True)
        dummies = dummies.astype(int)
        X_lineales = pd.concat([X_lineales, dummies], axis=1)

    X = pd.concat([X_continuas, X_lineales], axis=1)

    T = df[COL_TRATAMIENTO].to_numpy(dtype=int)
    Y = df[COL_OUTCOME].to_numpy(dtype=int)

    logger.info(
        "Matriz de propensidad construida (lead_time_spec=%s): "
        "X.shape=%s, %d tratados, %d controles.",
        lead_time_spec, X.shape, int(T.sum()), int((1 - T).sum()),
    )
    return X, T, Y, estadisticas


# =========================================================================
# Estimación del propensity score
# =========================================================================

@dataclass
class ResultadoPropensidad:
    """Resumen del ajuste logístico del propensity score."""
    modelo: LogisticRegression
    e_x: np.ndarray                       # P(T=1 | X) por fila
    auc: float                            # AUC del propensity sobre X→T
    p_tratamiento: float                  # P(T=1) marginal
    columnas: list[str]                   # orden de columnas usado
    estadisticas: EstadisticasEstandarizacion
    diagnostico_distribucion: dict[str, float] = field(default_factory=dict)


def estimar_propensity_score(
    X: pd.DataFrame,
    T: np.ndarray,
    estadisticas: EstadisticasEstandarizacion,
    *,
    C: float = LOG_REG_C,
    max_iter: int = LOG_REG_MAX_ITER,
    solver: str = LOG_REG_SOLVER,
) -> ResultadoPropensidad:
    """Ajusta la logística y devuelve scores + diagnósticos de e(X).

    Sin class_weight: queremos que e(X) quede calibrado a la marginal real,
    que es lo que entra en P(T=1)/e(X).
    """
    modelo = LogisticRegression(C=C, max_iter=max_iter, solver=solver)
    modelo.fit(X, T)
    e_x = modelo.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(T, e_x))
    p_t = float(T.mean())

    diag = {
        "min": float(e_x.min()),
        "p01": float(np.quantile(e_x, 0.01)),
        "p05": float(np.quantile(e_x, 0.05)),
        "p50": float(np.quantile(e_x, 0.50)),
        "p95": float(np.quantile(e_x, 0.95)),
        "p99": float(np.quantile(e_x, 0.99)),
        "max": float(e_x.max()),
        "pct_lt_0_01": float((e_x < 0.01).mean()),
        "pct_gt_0_99": float((e_x > 0.99).mean()),
    }
    logger.info(
        "Propensity ajustada: AUC=%.3f, P(T=1)=%.3f, e(X) ∈ [%.3f, %.3f]; "
        "%.2f%% con e<0.01, %.2f%% con e>0.99.",
        auc, p_t, diag["min"], diag["max"],
        100 * diag["pct_lt_0_01"], 100 * diag["pct_gt_0_99"],
    )
    return ResultadoPropensidad(
        modelo=modelo,
        e_x=e_x,
        auc=auc,
        p_tratamiento=p_t,
        columnas=list(X.columns),
        estadisticas=estadisticas,
        diagnostico_distribucion=diag,
    )


# =========================================================================
# Pesos
# =========================================================================

def pesos_estabilizados(e_x: np.ndarray, T: np.ndarray, p_t: float) -> np.ndarray:
    """Pesos IPW estabilizados: ω = P(T=1)/e(X) tratados, P(T=0)/(1−e(X)) controles."""
    e = np.clip(e_x, 1e-6, 1 - 1e-6)  # evita 0/0 en colas extremas
    return np.where(T == 1, p_t / e, (1 - p_t) / (1 - e))


def pesos_overlap_ato(e_x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Overlap weights (Li, Morgan & Zaslavsky, 2018): ω = 1−e(X) tratados, e(X) controles.

    Acotados en [0, 1] (no requieren trimming) y dan balance exacto en media
    para las covariables de e(X). El estimand resultante (ATO) es el efecto en
    la subpoblación con propensity cercana a 0.5, no el ATE ni el ATT.
    """
    return np.where(T == 1, 1 - e_x, e_x)


def ate_ato(Y: np.ndarray, T: np.ndarray, e_x: np.ndarray) -> float:
    """ATO en puntos porcentuales (media de Y ponderada por overlap weights)."""
    w = pesos_overlap_ato(e_x, T)
    return ate_ponderado(Y, T, w)


def resumen_pesos(w: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    """Estadísticos descriptivos de la distribución de pesos por brazo."""
    out: dict[str, Any] = {}
    for nombre, mask in (("tratados", T == 1), ("controles", T == 0)):
        sub = w[mask]
        out[nombre] = {
            "n": int(mask.sum()),
            "media": float(sub.mean()),
            "sd": float(sub.std(ddof=1)),
            "min": float(sub.min()),
            "p01": float(np.quantile(sub, 0.01)),
            "p50": float(np.quantile(sub, 0.50)),
            "p99": float(np.quantile(sub, 0.99)),
            "max": float(sub.max()),
        }
    return out


# =========================================================================
# Diagnósticos: overlap, SMD/love plot, ESS
# =========================================================================

def plot_overlap_propensidad(
    e_x: np.ndarray, T: np.ndarray, ruta_fig: str | Path
) -> None:
    """Densidades de e(X) para tratados y controles (positividad)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(e_x[T == 1], ax=ax, label="SMS = 1 (tratados)",
                fill=True, alpha=0.4, common_norm=False)
    sns.kdeplot(e_x[T == 0], ax=ax, label="SMS = 0 (controles)",
                fill=True, alpha=0.4, common_norm=False)
    ax.axvline(0.01, ls=":", color="gray", lw=0.8)
    ax.axvline(0.99, ls=":", color="gray", lw=0.8)
    ax.set_xlabel("Propensity score e(X) = P(SMS = 1 | X)")
    ax.set_ylabel("Densidad")
    ax.set_title("Solape del propensity score por brazo (positividad)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_fig, dpi=DPI_FIGURAS, bbox_inches="tight")
    plt.close(fig)
    logger.info("Overlap plot guardado en %s.", ruta_fig)


def _smd(x_t: np.ndarray, x_c: np.ndarray, w_t: np.ndarray | None = None,
         w_c: np.ndarray | None = None) -> float:
    """Standardized Mean Difference (ponderada si se pasan pesos)."""
    if w_t is None:
        mu_t, mu_c = x_t.mean(), x_c.mean()
        v_t = x_t.var(ddof=0)
        v_c = x_c.var(ddof=0)
    else:
        mu_t = np.average(x_t, weights=w_t)
        mu_c = np.average(x_c, weights=w_c)
        v_t = np.average((x_t - mu_t) ** 2, weights=w_t)
        v_c = np.average((x_c - mu_c) ** 2, weights=w_c)
    pooled = np.sqrt((v_t + v_c) / 2)
    if pooled < 1e-12:
        return 0.0
    return float((mu_t - mu_c) / pooled)


def calcular_tabla_smd(
    X: pd.DataFrame, T: np.ndarray, w: np.ndarray,
) -> pd.DataFrame:
    """SMD por covariable antes y después de ponderar (|SMD| < 0.10 = aceptable)."""
    filas = []
    mask_t = T == 1
    mask_c = T == 0
    for col in X.columns:
        x = X[col].to_numpy(dtype=float)
        smd_pre = _smd(x[mask_t], x[mask_c])
        smd_post = _smd(x[mask_t], x[mask_c], w[mask_t], w[mask_c])
        filas.append({
            "covariable": col,
            "etiqueta": ETIQUETAS_COVARIABLES.get(col, col),
            "smd_pre": smd_pre,
            "smd_post": smd_post,
            "abs_smd_pre": abs(smd_pre),
            "abs_smd_post": abs(smd_post),
        })
    tabla = pd.DataFrame(filas).sort_values(
        "abs_smd_post", ascending=False
    ).reset_index(drop=True)
    n_balance = int((tabla["abs_smd_post"] < 0.10).sum())
    logger.info(
        "Balance post-ponderación: %d / %d covariables con |SMD| < 0.10.",
        n_balance, len(tabla),
    )
    return tabla


def plot_love(
    tabla_smd: pd.DataFrame, ruta_fig: str | Path, metodo: str = "IPW",
) -> None:
    """Love plot: |SMD| pre vs post por covariable.

    `metodo` etiqueta el esquema de ponderación ("IPW" o "ATO"), ya que esta
    función se reutiliza para ambos.
    """
    tabla = tabla_smd.sort_values("abs_smd_pre", ascending=True).reset_index(drop=True)
    y = np.arange(len(tabla))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(tabla))))
    ax.scatter(tabla["abs_smd_pre"], y, label="Sin ponderar",
               marker="o", s=40, color="#d62728")
    ax.scatter(tabla["abs_smd_post"], y, label=f"Ponderado ({metodo})",
               marker="s", s=40, color="#1f77b4")
    for i, fila in tabla.iterrows():
        ax.plot([fila["abs_smd_pre"], fila["abs_smd_post"]], [i, i],
                color="gray", lw=0.6, alpha=0.5)
    ax.axvline(0.10, ls="--", color="black", lw=0.8, label="|SMD| = 0.10")
    ax.set_yticks(y)
    ax.set_yticklabels(tabla["etiqueta"])
    ax.set_xlabel("|SMD|")
    ax.set_title(f"Balance de covariables: antes vs después del {metodo}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta_fig, dpi=DPI_FIGURAS, bbox_inches="tight")
    plt.close(fig)
    logger.info("Love plot guardado en %s.", ruta_fig)


def calcular_ess(w: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    """Effective Sample Size por brazo: ESS = (Σω)² / Σω².

    Si ESS/n < 0.5 en algún brazo, se marca trim_recomendado=True.
    """
    out: dict[str, Any] = {}
    for nombre, mask in (("tratados", T == 1), ("controles", T == 0)):
        sub = w[mask]
        n = int(mask.sum())
        ess = float((sub.sum() ** 2) / np.sum(sub ** 2))
        out[nombre] = {
            "n": n,
            "ess": ess,
            "ess_sobre_n": ess / n,
        }
    out["trim_recomendado"] = bool(
        out["tratados"]["ess_sobre_n"] < TRIM_ESS_UMBRAL
        or out["controles"]["ess_sobre_n"] < TRIM_ESS_UMBRAL
    )
    logger.info(
        "ESS tratados=%.0f (%.1f%%); ESS controles=%.0f (%.1f%%); trim=%s.",
        out["tratados"]["ess"], 100 * out["tratados"]["ess_sobre_n"],
        out["controles"]["ess"], 100 * out["controles"]["ess_sobre_n"],
        out["trim_recomendado"],
    )
    return out


def plot_distribucion_pesos(
    w: np.ndarray, T: np.ndarray, ruta_fig: str | Path,
) -> None:
    """Histograma de log-pesos por brazo (log porque las colas son largas)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(np.log10(w).min(), np.log10(w).max(), 60)
    ax.hist(np.log10(w[T == 1]), bins=bins, alpha=0.5,
            label="SMS = 1 (tratados)", color="#1f77b4")
    ax.hist(np.log10(w[T == 0]), bins=bins, alpha=0.5,
            label="SMS = 0 (controles)", color="#d62728")
    ax.set_xlabel("log₁₀(peso estabilizado)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de pesos estabilizados por brazo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_fig, dpi=DPI_FIGURAS, bbox_inches="tight")
    plt.close(fig)
    logger.info("Histograma de pesos guardado en %s.", ruta_fig)


# =========================================================================
# Estimación del ATE
# =========================================================================

def _media_ponderada(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(x, weights=w))


def ate_ponderado(Y: np.ndarray, T: np.ndarray, w: np.ndarray) -> float:
    """ATE = E_w[Y|T=1] − E_w[Y|T=0], en puntos porcentuales.

    Asociacional (ajustado por covariables observadas), no causal en sentido
    estricto.
    """
    att = _media_ponderada(Y[T == 1].astype(float), w[T == 1])
    atc = _media_ponderada(Y[T == 0].astype(float), w[T == 0])
    return (att - atc) * 100.0


def ate_trimmed(
    Y: np.ndarray, T: np.ndarray, w: np.ndarray,
    pct_inferior: float, pct_superior: float,
) -> tuple[float, np.ndarray]:
    """ATE recortando pesos por percentiles. Devuelve (ate_pp, máscara conservada)."""
    lo = np.quantile(w, pct_inferior / 100.0)
    hi = np.quantile(w, pct_superior / 100.0)
    mask = (w >= lo) & (w <= hi)
    ate = ate_ponderado(Y[mask], T[mask], w[mask])
    return ate, mask


def construir_tabla_ate(
    Y: np.ndarray, T: np.ndarray, w: np.ndarray,
    spec_preferida: str,
) -> pd.DataFrame:
    """Tabla con el ATE untrimmed + las dos especificaciones de robustez (1/99, 5/95)."""
    filas: list[dict[str, Any]] = []
    ate_unt = ate_ponderado(Y, T, w)
    filas.append({
        "especificacion": "untrimmed",
        "pct_inferior": np.nan,
        "pct_superior": np.nan,
        "n_filas_usadas": len(w),
        "ate_pp": ate_unt,
        "es_preferida": spec_preferida == "untrimmed",
    })
    for lo, hi in TRIM_ROBUSTEZ:
        ate_t, mask = ate_trimmed(Y, T, w, lo, hi)
        nombre = f"trimmed_{int(lo)}_{int(hi)}"
        filas.append({
            "especificacion": nombre,
            "pct_inferior": lo,
            "pct_superior": hi,
            "n_filas_usadas": int(mask.sum()),
            "ate_pp": ate_t,
            "es_preferida": spec_preferida == nombre,
        })
    return pd.DataFrame(filas)


def decidir_especificacion_preferida(ess: dict[str, Any]) -> str:
    """untrimmed si ambos brazos cumplen ESS/n ≥ 0.5; si no, trimmed_1_99."""
    if ess["trim_recomendado"]:
        return "trimmed_1_99"
    return "untrimmed"


# =========================================================================
# Cluster bootstrap (PatientId)
# =========================================================================

def _construir_indices_por_paciente(patient_ids: pd.Series) -> dict[Any, np.ndarray]:
    """{PatientId → posiciones en el df}. Precomputado una vez para el bootstrap."""
    return {k: np.asarray(v) for k, v in patient_ids.groupby(patient_ids).indices.items()}


def cluster_bootstrap_ato(
    X: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    patient_ids: pd.Series,
    *,
    n_iter: int = N_BOOTSTRAP,
    seed: int = SEED_BOOTSTRAP,
    C: float = LOG_REG_C,
    max_iter: int = LOG_REG_MAX_ITER,
    solver: str = LOG_REG_SOLVER,
    verbose_cada: int = 100,
) -> np.ndarray:
    """Cluster bootstrap del ATO por PatientId.

    En cada iteración: remuestrea PatientIds con reemplazo, refita la logística
    sobre el resample, recalcula los overlap weights y devuelve el ATO en pp.
    No aplica trimming (los pesos ya están acotados en [0, 1]).
    """
    rng = np.random.default_rng(seed)
    pid_to_idx = _construir_indices_por_paciente(patient_ids)
    unique_pids = np.fromiter(pid_to_idx.keys(), dtype=patient_ids.dtype)

    X_np = X.to_numpy(dtype=float)
    atos = np.empty(n_iter, dtype=float)

    logger.info(
        "Cluster bootstrap ATO: %d iteraciones, %d pacientes únicos.",
        n_iter, len(unique_pids),
    )

    for i in range(n_iter):
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([pid_to_idx[pid] for pid in sampled])

        X_b = X_np[idx]
        T_b = T[idx]
        Y_b = Y[idx]

        modelo = LogisticRegression(C=C, max_iter=max_iter, solver=solver)
        modelo.fit(X_b, T_b)
        e_b = modelo.predict_proba(X_b)[:, 1]
        w_b = pesos_overlap_ato(e_b, T_b)

        atos[i] = ate_ponderado(Y_b, T_b, w_b)

        if verbose_cada and (i + 1) % verbose_cada == 0:
            logger.info(
                "  bootstrap %d/%d — ATO acumulado: media=%.3f pp, "
                "IC95%%≈[%.3f, %.3f] pp",
                i + 1, n_iter, atos[:i + 1].mean(),
                np.quantile(atos[:i + 1], 0.025),
                np.quantile(atos[:i + 1], 0.975),
            )

    return atos


def resumen_bootstrap(ates: np.ndarray) -> dict[str, float]:
    """Resumen de la distribución bootstrap: media, mediana, IC95%."""
    return {
        "n_iter": int(len(ates)),
        "media": float(ates.mean()),
        "mediana": float(np.median(ates)),
        "sd": float(ates.std(ddof=1)),
        "ic95_inf": float(np.quantile(ates, 0.025)),
        "ic95_sup": float(np.quantile(ates, 0.975)),
        "min": float(ates.min()),
        "max": float(ates.max()),
    }


def plot_distribucion_ate_bootstrap(
    ates: np.ndarray, ate_punto: float, ruta_fig: str | Path,
    spec: str = "preferida",
) -> None:
    """Histograma del bootstrap con la estimación puntual y el IC95%."""
    ic_inf = float(np.quantile(ates, 0.025))
    ic_sup = float(np.quantile(ates, 0.975))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ates, bins=40, color="#1f77b4", alpha=0.75, edgecolor="white")
    ax.axvline(ate_punto, color="black", lw=2,
               label=f"ATE puntual = {ate_punto:.2f} pp")
    ax.axvline(ic_inf, color="red", ls="--",
               label=f"IC95% inf = {ic_inf:.2f}")
    ax.axvline(ic_sup, color="red", ls="--",
               label=f"IC95% sup = {ic_sup:.2f}")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("ATE en puntos porcentuales (asistencia)")
    ax.set_ylabel("Frecuencia (sobre 1000 bootstraps)")
    ax.set_title(
        f"Distribución bootstrap del ATE (cluster por PatientId, spec: {spec})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_fig, dpi=DPI_FIGURAS, bbox_inches="tight")
    plt.close(fig)
    logger.info("Histograma del bootstrap guardado en %s.", ruta_fig)


# =========================================================================
# E-value (sensibilidad a confusión no observada)
# =========================================================================

def ate_pp_a_rr(
    ate_pp: float, Y: np.ndarray, T: np.ndarray, w: np.ndarray,
) -> tuple[float, float, float]:
    """Risk ratio aproximado (asistencia ponderada T=1 / T=0) para el E-value."""
    att_T = _media_ponderada(Y[T == 1].astype(float), w[T == 1])
    att_C = _media_ponderada(Y[T == 0].astype(float), w[T == 0])
    rr = att_T / att_C if att_C > 0 else float("nan")
    return float(rr), float(att_T), float(att_C)


def evalue(rr: float) -> float:
    """E-value (VanderWeele & Ding, 2017): E = RR + sqrt(RR·(RR−1)), invirtiendo si RR<1.

    Una confusión no observada tendría que asociarse a tratamiento y outcome
    con un RR ≥ E para explicar el efecto observado.
    """
    if rr <= 0 or not np.isfinite(rr):
        return float("nan")
    if rr == 1:
        return 1.0
    rr_eff = rr if rr > 1 else 1 / rr
    return float(rr_eff + np.sqrt(rr_eff * (rr_eff - 1)))


# =========================================================================
# Persistencia
# =========================================================================

def persistir_propensidad(
    resultado: ResultadoPropensidad, ruta: str | Path,
) -> None:
    """Guarda el modelo + columnas + estadísticas de estandarización (pickle)."""
    payload = {
        "modelo": resultado.modelo,
        "columnas": resultado.columnas,
        "estadisticas_medias": resultado.estadisticas.medias,
        "estadisticas_desviaciones": resultado.estadisticas.desviaciones,
        "p_tratamiento": resultado.p_tratamiento,
        "auc": resultado.auc,
    }
    with open(ruta, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Propensidad persistida en %s.", ruta)


def persistir_bootstrap(ates: np.ndarray, ruta: str | Path) -> None:
    """Guarda el array `.npy` del bootstrap."""
    np.save(ruta, ates)
    logger.info("Bootstrap persistido en %s (%d valores).", ruta, len(ates))


def persistir_sidecar(metadata: dict[str, Any], ruta: str | Path) -> None:
    """Sidecar JSON con metadatos del notebook (timestamps + decisiones + KPIs)."""
    metadata = {**metadata, "guardado_en": datetime.now(timezone.utc).isoformat()}
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Sidecar JSON guardado en %s.", ruta)
