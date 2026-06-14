"""Segmentación de pacientes por riesgo y accionabilidad (K-means).

Agrupa las citas del conjunto de test (las que tienen probabilidad calibrada de
no-show del modelo XGBoost) en perfiles operativamente interpretables, para
mostrar que un envío uniforme de SMS es subóptimo y orientar la priorización.

El clustering combina dos tipos de eje: la probabilidad calibrada aporta el
*ranking de riesgo*, y la antelación, el barrio y la condición de primera visita
aportan *accionabilidad operativa* (cuándo, dónde y a quién intervenir). Las
variables de riesgo más finas (edad, tasa de no-show previa, comorbilidades) se
reservan para perfilar los clústeres, no para definirlos.
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SEED: int = 42
DPI_FIGURAS: int = 200

# Variables de clustering (compactas, accionables). El bin de antelación entra
# como ordinal (la antelación sí tiene orden natural, a diferencia del día de
# la semana, que sólo se usa para perfilar).
LEAD_BIN_ORDINAL: dict[str, int] = {"mismo_dia": 0, "1-7d": 1, "8-14d": 2, "15+d": 3}
FEATURES_CLUSTERING: list[str] = [
    "prob_noshow_calibrada",
    "neighbourhood_encoded",
    "lead_bin_ord",
    "is_first_visit",
]
ETIQUETAS_FEATURES: dict[str, str] = {
    "prob_noshow_calibrada": "prob. no-show calibrada",
    "neighbourhood_encoded": "barrio (frecuencia)",
    "lead_bin_ord": "antelación (bin ordinal)",
    "is_first_visit": "primera visita observada",
}


def configurar_estilo() -> None:
    """Estilo de plots coherente con los demás notebooks."""
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": DPI_FIGURAS,
        "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })


def cargar_datos_segmentacion(
    ruta_probs: str | Path, ruta_full_clean: str | Path,
) -> pd.DataFrame:
    """Une la probabilidad calibrada (NB03) con las covariables de la cita (NB01).

    El clustering vive sobre el conjunto de test (las citas con probabilidad
    calibrada fuera de muestra), unidas por `AppointmentID`.
    """
    probs = pd.read_csv(ruta_probs, usecols=[
        "PatientId", "AppointmentID", "Showed_up", "prob_noshow_calibrada"])
    cols_fc = ["AppointmentID", "SMS_received", "lead_time_bin", "is_first_visit",
               "neighbourhood_encoded", "Age", "lead_time", "prior_noshow_rate",
               "comorbidity_count", "chronic_flag", "appointment_weekday"]
    fc = pd.read_csv(ruta_full_clean, usecols=cols_fc)

    df = probs.merge(fc, on="AppointmentID", how="left")
    df["lead_bin_ord"] = df["lead_time_bin"].map(LEAD_BIN_ORDINAL)
    if df["lead_bin_ord"].isna().any():
        raise ValueError("lead_time_bin con valores fuera del mapa ordinal.")
    logger.info("Datos de segmentación: %d citas (test), %d pacientes únicos.",
                len(df), df["PatientId"].nunique())
    return df


def construir_matriz_clustering(
    df: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler]:
    """Estandariza las cuatro variables de clustering (StandardScaler)."""
    X = df[FEATURES_CLUSTERING].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def evaluar_k(
    X_scaled: np.ndarray, k_range: range = range(2, 9), seed: int = SEED,
) -> pd.DataFrame:
    """Inercia (codo) y silueta para cada k. La silueta se calcula sobre una
    submuestra de 5.000 puntos (su coste es cuadrático en n)."""
    filas = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=seed)
        filas.append({"k": k, "inercia": float(km.inertia_), "silueta": float(sil)})
        logger.info("k=%d → inercia=%.0f, silueta=%.4f", k, km.inertia_, sil)
    return pd.DataFrame(filas)


def plot_seleccion_k(tabla: pd.DataFrame, ruta_fig: str | Path) -> None:
    """Curva de codo + silueta frente a k."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(tabla["k"], tabla["inercia"], "o-", color="#1f77b4")
    ax1.set_xlabel("número de clústeres (k)")
    ax1.set_ylabel("inercia (suma de cuadrados intra-clúster)")
    ax1.set_title("Curva de codo")
    ax2.plot(tabla["k"], tabla["silueta"], "o-", color="#d62728")
    ax2.set_xlabel("número de clústeres (k)")
    ax2.set_ylabel("coeficiente de silueta")
    ax2.set_title("Silueta media frente a k")
    fig.tight_layout()
    fig.savefig(ruta_fig, dpi=DPI_FIGURAS, bbox_inches="tight")
    plt.close(fig)
    logger.info("Selección de k guardada en %s.", ruta_fig)


def ajustar_kmeans(
    X_scaled: np.ndarray, k: int, seed: int = SEED,
) -> tuple[np.ndarray, KMeans]:
    """Ajusta K-means con k clústeres y devuelve etiquetas + modelo."""
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def plot_pca(X_scaled: np.ndarray, labels: np.ndarray, ruta_fig: str | Path) -> None:
    """Proyección PCA 2D de las citas, coloreadas por clúster."""
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for c in sorted(np.unique(labels)):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.35, label=f"Clúster {c}")
    ax.set_xlabel(f"PC1 ({100*var[0]:.0f}% var.)")
    ax.set_ylabel(f"PC2 ({100*var[1]:.0f}% var.)")
    ax.set_title("Proyección PCA de los clústeres de pacientes")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(ruta_fig, dpi=DPI_FIGURAS, bbox_inches="tight")
    plt.close(fig)
    logger.info("PCA guardado en %s.", ruta_fig)


def perfilar_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Perfil descriptivo por clúster con variables crudas (no usadas para ajustar)."""
    d = df.copy()
    d["cluster"] = labels
    perfil = d.groupby("cluster").agg(
        n=("AppointmentID", "size"),
        prob_noshow_media=("prob_noshow_calibrada", "mean"),
        noshow_observado=("Showed_up", lambda s: float((s == 0).mean())),
        lead_time_medio=("lead_time", "mean"),
        edad_media=("Age", "mean"),
        pct_primera_visita=("is_first_visit", "mean"),
        prior_noshow_medio=("prior_noshow_rate", "mean"),
        comorbilidades_media=("comorbidity_count", "mean"),
        pct_sms=("SMS_received", "mean"),
    ).reset_index()
    return perfil.round(4)


def resumen_sms_por_cluster(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """No-show observado por brazo de SMS dentro de cada clúster.

    Es la comparación cruda (sin ajustar): igual que en el EDA, está sujeta al
    sesgo de selección del envío de SMS y se interpreta de forma descriptiva,
    no causal.
    """
    d = df.copy()
    d["cluster"] = labels
    d["noshow"] = (d["Showed_up"] == 0).astype(int)
    filas = []
    for c, sub in d.groupby("cluster"):
        s1, s0 = sub[sub["SMS_received"] == 1], sub[sub["SMS_received"] == 0]
        filas.append({
            "cluster": int(c),
            "n_sms": len(s1), "n_no_sms": len(s0),
            "noshow_sms": float(s1["noshow"].mean()) if len(s1) else np.nan,
            "noshow_no_sms": float(s0["noshow"].mean()) if len(s0) else np.nan,
        })
    out = pd.DataFrame(filas)
    out["dif_observada_pp"] = (out["noshow_sms"] - out["noshow_no_sms"]) * 100
    return out.round(4)


def persistir_segmentacion(
    df: pd.DataFrame, labels: np.ndarray, ruta: str | Path,
) -> None:
    """CSV por cita con su clúster asignado (entrada para escenarios de NB06)."""
    out = df[["PatientId", "AppointmentID", "prob_noshow_calibrada",
              "SMS_received", "Showed_up"]].copy()
    out["cluster"] = labels
    out.to_csv(ruta, index=False)
    logger.info("Asignación de clústeres guardada en %s (%d filas).", ruta, len(out))


def persistir_sidecar(metadata: dict[str, Any], ruta: str | Path) -> None:
    """Sidecar JSON con metadatos de la segmentación."""
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Sidecar guardado en %s.", ruta)
