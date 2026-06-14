"""Notebook 03 — modelo predictivo (XGBoost + calibración + SHAP).

Predice la probabilidad de no-asistencia a partir de variables pre-tratamiento
(SMS_received NO entra — es el tratamiento; se estima por separado en NB04).

Decisiones de configuración del modelo:
    1. Target = 1 - Showed_up (no-show). Documentado en `preparar_features`.
    2. Grid de 9 combos (max_depth × learning_rate) con early_stopping=50.
    3. CV walk-forward (3 folds expanding) dentro del 80% inicial del train,
       para no solapar con `train_calib` (último 20%).
    4. Calibración dual Platt/Isotonic en `train_calib`; selección por Brier
       con tiebreak ε=0.001 → Platt.
    5. `scale_pos_weight = neg/pos` para discriminación; la distorsión de la
       escala de probabilidad se corrige con la calibración aguas abajo.
    6. SHAP: TreeExplainer sobre todo el test set; submuestreo a 5k filas
       estratificado SÓLO para visualización.

Convenciones: funciones y nombres locales en español;
identificadores de bibliotecas estándar en inglés; narrativa y plots en
español; figuras a 200 DPI bajo `outputs/figuras/`.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Constantes de configuración
# --------------------------------------------------------------------------

# Features pre-tratamiento (SMS_received y scheduled_hour EXCLUIDOS por
# diseño: SMS_received es el tratamiento, scheduled_hour es constante).
# Las binarias bool del CSV (Scholarship, etc.) se castean a int antes de
# entrenar; XGBoost las admite, pero el tipo explícito evita ambigüedad
# en SHAP y en la matriz de features.
FEATURES_NB03: list[str] = [
    "lead_time",
    "Age",
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
    "prior_noshow_rate",
    "is_first_visit",
]

# Etiquetas legibles en castellano para plots (SHAP, calibración, etc.)
ETIQUETAS_FEATURES: dict[str, str] = {
    "lead_time": "antelación (días)",
    "Age": "edad",
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
    "prior_noshow_rate": "tasa no-show previa observada",
    "is_first_visit": "primera visita observada",
}

# Grid 3×3 (n_estimators sale del cap + early stopping, NO se tunea como
# dimensión: el `best_iteration` resultante se registra como output derivado)
GRID_MAX_DEPTH: list[int] = [4, 6, 8]
GRID_LEARNING_RATE: list[float] = [0.05, 0.1, 0.2]

N_ESTIMATORS_CAP: int = 1000
EARLY_STOPPING_ROUNDS: int = 50

# Hiperparámetros fijos (no tuneados): elección estándar para clasificación
# binaria con muestras > 10k filas. Mover estos rara vez mueve AUC > 0.005.
SUBSAMPLE: float = 0.8
COLSAMPLE_BYTREE: float = 0.8

# Selección de calibrador (PLAN §NB03 + decisión 1)
DELTA_BRIER_TIEBREAK: float = 1e-3   # si |Brier_iso - Brier_platt| < ε → Platt

# Walk-forward: 3 folds expanding window dentro del 80% inicial del train
N_FOLDS_WALK_FORWARD: int = 3
FRAC_TRAIN_XGB: float = 0.80   # primer 80% del periodo de train

# SHAP: tamaño de submuestra estratificada SÓLO para visualización
N_SHAP_VISUALIZACION: int = 5000


# --------------------------------------------------------------------------
# Carga y preparación de features
# --------------------------------------------------------------------------

def cargar_train_test_nb03(
    ruta_train: str | Path, ruta_test: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga `train_v1.csv` y `test_v1.csv` parseando fechas."""
    parse = ["ScheduledDay", "AppointmentDay"]
    train = pd.read_csv(ruta_train, parse_dates=parse)
    test = pd.read_csv(ruta_test, parse_dates=parse)
    logger.info(
        "Train cargado: %d filas, Test: %d filas. Train spans %s → %s.",
        len(train), len(test),
        train["AppointmentDay"].min().date(),
        train["AppointmentDay"].max().date(),
    )
    return train, test


def verificar_codificacion_showed_up_pred(df: pd.DataFrame) -> None:
    """Re-verifica la convención Showed_up=1 → asistió.

    No transforma — sólo valida. La transformación canónica vive en NB01
    (`verificar_codificacion_showed_up` de `preparacion.py`); aquí simplemente
    confirmamos antes de invertir signo a la variable objetivo.
    """
    pct_atendieron = float((df["Showed_up"] == 1).mean())
    if not 0.65 < pct_atendieron < 0.90:
        raise ValueError(
            f"Showed_up tiene una tasa de asistencia inesperada ({pct_atendieron:.1%}); "
            "verificar codificación antes de entrenar."
        )
    logger.info(
        "Showed_up verificado: %.1f%% asistencia. Target del modelo = 1 - Showed_up (no-show).",
        pct_atendieron * 100,
    )


def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Construye la matriz X (features pre-tratamiento) y el target `y_noshow`.

    Convenciones:
        - `y_noshow = 1 - Showed_up` ⇒ positivo = no asistió. Esto facilita
          la interpretación de SHAP (signo positivo = sube el riesgo de
          no-show) y de scale_pos_weight (ratio neg/pos para la clase
          minoritaria, que coincide con el no-show).
        - `Gender` codificado como `gender_F = 1` para F, `0` para M.
          Dirección documentada para interpretar signos de SHAP.
        - Binarias del CSV (`Scholarship`, comorbilidades) casteadas a int.
        - `prior_noshow_rate` NO se imputa: XGBoost gestiona NaN nativamente,
          y `NaN` aquí significa "sin historial observado" (censura por la
          izquierda), no "0 inasistencias previas".
    """
    df = df.copy()

    # gender_F = 1 si F, 0 si M (string 'F'/'M' tras NB01)
    df["gender_F"] = (df["Gender"] == "F").astype(int)

    # Castear binarias bool → int (las trae así pd.read_csv del NB01 output)
    for col in (
        "Scholarship", "Hipertension", "Diabetes",
        "Alcoholism", "Handcap", "chronic_flag", "is_first_visit",
    ):
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    X = df[FEATURES_NB03].copy()
    y = (1 - df["Showed_up"]).astype(int).rename("noshow")
    logger.info(
        "X: %s, prevalencia no-show: %.1f%%",
        X.shape, 100 * y.mean(),
    )
    return X, y


# --------------------------------------------------------------------------
# División cronológica train_xgb / train_calib
# --------------------------------------------------------------------------

def dividir_train_xgb_calib(
    X: pd.DataFrame,
    y: pd.Series,
    fechas: pd.Series,
    frac_xgb: float = FRAC_TRAIN_XGB,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Timestamp]:
    """Split cronológico del train: primer `frac_xgb` → train_xgb, resto → train_calib.

    El corte se define como el cuantil `frac_xgb` de `fechas` (AppointmentDay).
    `train_calib` contiene la última franja temporal del periodo de
    entrenamiento — XGBoost no la verá durante el ajuste final, y la
    calibración se ajustará sobre ella.
    """
    fecha_corte = fechas.quantile(frac_xgb, interpolation="lower")
    mask_xgb = fechas <= fecha_corte
    X_xgb, y_xgb = X.loc[mask_xgb], y.loc[mask_xgb]
    X_cal, y_cal = X.loc[~mask_xgb], y.loc[~mask_xgb]
    logger.info(
        "Train→ train_xgb (≤ %s): %d filas; train_calib (> %s): %d filas.",
        pd.Timestamp(fecha_corte).date(), len(X_xgb),
        pd.Timestamp(fecha_corte).date(), len(X_cal),
    )
    return X_xgb, y_xgb, X_cal, y_cal, pd.Timestamp(fecha_corte)


# --------------------------------------------------------------------------
# Walk-forward CV folds
# --------------------------------------------------------------------------

@dataclass
class FoldWalkForward:
    fold: int
    fecha_corte_train: pd.Timestamp
    fecha_corte_val: pd.Timestamp
    idx_train: np.ndarray
    idx_val: np.ndarray

    def resumen(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_hasta": self.fecha_corte_train.date().isoformat(),
            "val_hasta": self.fecha_corte_val.date().isoformat(),
            "n_train": int(len(self.idx_train)),
            "n_val": int(len(self.idx_val)),
        }


def generar_folds_walk_forward(
    fechas: pd.Series, n_folds: int = N_FOLDS_WALK_FORWARD
) -> list[FoldWalkForward]:
    """Genera `n_folds` folds expanding-window dentro de `fechas`.

    Estrategia: dividir el rango de fechas por cuantiles uniformes; el fold
    `k` entrena con todo lo anterior al `k`-ésimo corte y valida sobre la
    ventana siguiente. Equivale a `TimeSeriesSplit` pero anclado a fechas
    (no a índices) — más interpretable y resistente a desbalance de volumen
    por día.

    Para `n_folds=3` los cortes son los cuantiles {0.50, 0.67, 0.83, 1.00}:
        Fold 1: train ≤ q50,        val (q50,  q67]
        Fold 2: train ≤ q67,        val (q67,  q83]
        Fold 3: train ≤ q83,        val (q83, q100]
    """
    fechas = pd.Series(fechas).reset_index(drop=True)
    # Cuantiles uniformes: para 3 folds → [0.5, 0.67, 0.83, 1.0]
    cuantiles = np.linspace(0.5, 1.0, n_folds + 1)
    cortes = [fechas.quantile(q, interpolation="lower") for q in cuantiles]

    folds: list[FoldWalkForward] = []
    for k in range(n_folds):
        corte_train, corte_val = cortes[k], cortes[k + 1]
        idx_train = np.where(fechas <= corte_train)[0]
        # Val: ventana abierta-por-izquierda, cerrada-por-derecha
        mask_val = (fechas > corte_train) & (fechas <= corte_val)
        idx_val = np.where(mask_val)[0]
        folds.append(FoldWalkForward(
            fold=k + 1,
            fecha_corte_train=pd.Timestamp(corte_train),
            fecha_corte_val=pd.Timestamp(corte_val),
            idx_train=idx_train,
            idx_val=idx_val,
        ))
    for f in folds:
        logger.info("Fold %d: %s", f.fold, f.resumen())
    return folds


# --------------------------------------------------------------------------
# Búsqueda de hiperparámetros (grid 3×3 + early stopping)
# --------------------------------------------------------------------------

def _calcular_scale_pos_weight(y: pd.Series) -> float:
    """Ratio neg/pos. Para y=no-show con prevalencia ~20% → ≈ 4.0."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = n_neg / n_pos
    logger.info("scale_pos_weight = neg/pos = %d/%d = %.3f", n_neg, n_pos, spw)
    return spw


def _instanciar_xgb(
    max_depth: int,
    learning_rate: float,
    scale_pos_weight: float,
    n_estimators: int = N_ESTIMATORS_CAP,
    early_stopping_rounds: int | None = EARLY_STOPPING_ROUNDS,
    seed: int = 42,
) -> xgb.XGBClassifier:
    """Construye un XGBClassifier con la configuración fija + variable."""
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


def buscar_hiperparametros(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[FoldWalkForward],
    grid_max_depth: Iterable[int] = GRID_MAX_DEPTH,
    grid_learning_rate: Iterable[float] = GRID_LEARNING_RATE,
    seed: int = 42,
) -> pd.DataFrame:
    """Tuning walk-forward de `max_depth × learning_rate`.

    Para cada combo entrena un XGBClassifier en cada fold con early stopping
    sobre la ventana de validación, registra AUC y best_iteration. La métrica
    objetivo del tuning es AUC-ROC promediada sobre folds.

    Devuelve un DataFrame con una fila por combo: parámetros, AUC media,
    AUC por fold, best_iteration media, AUC desviación entre folds.
    """
    spw = _calcular_scale_pos_weight(y)
    filas: list[dict[str, Any]] = []

    for max_depth in grid_max_depth:
        for lr in grid_learning_rate:
            aucs: list[float] = []
            best_iters: list[int] = []
            for f in folds:
                Xtr, ytr = X.iloc[f.idx_train], y.iloc[f.idx_train]
                Xva, yva = X.iloc[f.idx_val], y.iloc[f.idx_val]
                modelo = _instanciar_xgb(
                    max_depth=max_depth,
                    learning_rate=lr,
                    scale_pos_weight=spw,
                    seed=seed,
                )
                modelo.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                pred = modelo.predict_proba(Xva)[:, 1]
                aucs.append(float(roc_auc_score(yva, pred)))
                best_iters.append(int(modelo.best_iteration))
            fila = {
                "max_depth": max_depth,
                "learning_rate": lr,
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "best_iteration_mean": float(np.mean(best_iters)),
                **{f"auc_fold{i+1}": a for i, a in enumerate(aucs)},
                **{f"best_iter_fold{i+1}": b for i, b in enumerate(best_iters)},
            }
            logger.info(
                "Combo md=%d lr=%.2f → AUC=%.4f±%.4f, best_iter≈%.0f",
                max_depth, lr, fila["auc_mean"], fila["auc_std"], fila["best_iteration_mean"],
            )
            filas.append(fila)

    df = pd.DataFrame(filas).sort_values("auc_mean", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Modelo final y calibradores
# --------------------------------------------------------------------------

def ajustar_modelo_final(
    X_xgb: pd.DataFrame,
    y_xgb: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    mejores: Mapping[str, Any],
    seed: int = 42,
) -> xgb.XGBClassifier:
    """Ajusta el XGBoost final en `train_xgb` con early stopping sobre `train_calib`.

    `train_calib` cumple aquí doble función:
        1. Detener temprano para evitar overfitting (early stopping).
        2. Aguas abajo, ajustar el calibrador (Platt/Isotonic).
    Esto es coherente: el modelo se entrena para discriminación; el sesgo
    de probabilidad introducido por `scale_pos_weight` se corrige con la
    calibración sobre la MISMA franja temporal — no introduce fuga porque
    `train_calib` se conoce en el momento del ajuste (es pasado respecto a
    test).
    """
    spw = _calcular_scale_pos_weight(y_xgb)
    modelo = _instanciar_xgb(
        max_depth=int(mejores["max_depth"]),
        learning_rate=float(mejores["learning_rate"]),
        scale_pos_weight=spw,
        seed=seed,
    )
    modelo.fit(X_xgb, y_xgb, eval_set=[(X_calib, y_calib)], verbose=False)
    logger.info(
        "Modelo final ajustado (md=%d, lr=%.2f, best_iteration=%d).",
        modelo.max_depth, modelo.learning_rate, modelo.best_iteration,
    )
    return modelo


def ajustar_calibradores(
    modelo: xgb.XGBClassifier,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    delta_tiebreak: float = DELTA_BRIER_TIEBREAK,
) -> tuple[CalibratedClassifierCV, pd.DataFrame]:
    """Ajusta Platt e Isotonic sobre train_calib y selecciona por Brier.

    Regla de selección:
        - Si |Brier_iso - Brier_platt| < delta_tiebreak → Platt (más simple).
        - En otro caso → el de menor Brier.

    El modelo base se congela (FrozenEstimator) para que la calibración no
    re-ajuste el XGBoost. La calibración se aplica a las salidas del modelo
    YA entrenado con `scale_pos_weight` — calibrar sobre una versión sin
    ponderar invalidaría la lógica (la distorsión que corregimos es la que
    `scale_pos_weight` introduce).

    Devuelve `(calibrador_seleccionado, tabla_comparativa)`.
    """
    frozen = FrozenEstimator(modelo)
    resultados: dict[str, dict[str, Any]] = {}
    for metodo in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(frozen, method=metodo).fit(X_calib, y_calib)
        prob_cal = cal.predict_proba(X_calib)[:, 1]
        brier = float(brier_score_loss(y_calib, prob_cal))
        resultados[metodo] = {"calibrador": cal, "brier_calib": brier}

    brier_raw = float(brier_score_loss(
        y_calib, modelo.predict_proba(X_calib)[:, 1]
    ))

    brier_platt = resultados["sigmoid"]["brier_calib"]
    brier_iso = resultados["isotonic"]["brier_calib"]
    diff = brier_iso - brier_platt
    if abs(diff) < delta_tiebreak:
        elegido, motivo = "sigmoid", f"diferencia |Δ|={abs(diff):.5f} < ε={delta_tiebreak} → default Platt"
    elif brier_platt <= brier_iso:
        elegido, motivo = "sigmoid", f"Platt mejor por {-diff:.5f}"
    else:
        elegido, motivo = "isotonic", f"Isotonic mejor por {diff:.5f}"

    logger.info(
        "Brier en train_calib: raw=%.5f, Platt=%.5f, Isotonic=%.5f. Elegido: %s (%s).",
        brier_raw, brier_platt, brier_iso, elegido, motivo,
    )

    tabla = pd.DataFrame([
        {"metodo": "sin_calibrar", "brier_train_calib": brier_raw, "seleccionado": False},
        {"metodo": "platt_sigmoid", "brier_train_calib": brier_platt, "seleccionado": elegido == "sigmoid"},
        {"metodo": "isotonic", "brier_train_calib": brier_iso, "seleccionado": elegido == "isotonic"},
    ])
    tabla.attrs["motivo_seleccion"] = motivo
    tabla.attrs["metodo_elegido"] = elegido
    return resultados[elegido]["calibrador"], tabla


# --------------------------------------------------------------------------
# Evaluación en test
# --------------------------------------------------------------------------

@dataclass
class MetricasTest:
    auc_roc: float
    pr_auc: float
    brier_raw: float
    brier_calibrado: float
    prevalencia: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def calcular_metricas_test(
    modelo: xgb.XGBClassifier,
    calibrador: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[MetricasTest, np.ndarray, np.ndarray]:
    """Calcula AUC-ROC, PR-AUC, Brier (raw + calibrado). Devuelve probabilidades."""
    prob_raw = modelo.predict_proba(X_test)[:, 1]
    prob_cal = calibrador.predict_proba(X_test)[:, 1]
    m = MetricasTest(
        auc_roc=float(roc_auc_score(y_test, prob_cal)),
        pr_auc=float(average_precision_score(y_test, prob_cal)),
        brier_raw=float(brier_score_loss(y_test, prob_raw)),
        brier_calibrado=float(brier_score_loss(y_test, prob_cal)),
        prevalencia=float(y_test.mean()),
    )
    logger.info(
        "Test: AUC=%.4f, PR-AUC=%.4f, Brier raw=%.4f, Brier calibrado=%.4f, prev=%.3f",
        m.auc_roc, m.pr_auc, m.brier_raw, m.brier_calibrado, m.prevalencia,
    )
    return m, prob_raw, prob_cal


def matriz_confusion_tres_umbrales(
    y_test: pd.Series, prob_cal: np.ndarray
) -> pd.DataFrame:
    """Matriz de confusión a tres umbrales: top-20%, 0.5, prevalencia-informado.

    Tres filas con (umbral, TN, FP, FN, TP, precision, recall, n_positivos_predichos).

    - Top-20% (operacional): umbral = percentil 80 de prob_cal. Corresponde
      a un presupuesto realista de SMS sobre el 20% del volumen mensual.
    - 0.5 (referencia de literatura): clasificación nominal; informativo
      pero poco útil con prevalencia ~20%.
    - Prevalencia (≈ tasa observada de no-show): umbral = prevalencia del
      target en test. NO se llama "Youden-like" porque no se optimiza J.
    """
    prev = float(y_test.mean())
    umbral_top20 = float(np.quantile(prob_cal, 0.80))

    filas = []
    for nombre, umbral in [
        ("top20", umbral_top20),
        ("literatura_0.5", 0.5),
        ("prevalencia", prev),
    ]:
        y_pred = (prob_cal >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        filas.append({
            "umbral_nombre": nombre,
            "umbral_valor": float(umbral),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "n_positivos_predichos": int(y_pred.sum()),
            "pct_positivos_predichos": float(y_pred.mean()),
        })
    df = pd.DataFrame(filas)
    logger.info(
        "Matriz de confusión calculada a 3 umbrales: top20=%.4f, 0.5, prev=%.4f",
        umbral_top20, prev,
    )
    return df


# --------------------------------------------------------------------------
# Curva de calibración (test) + sanity check interno
# --------------------------------------------------------------------------

def plot_curva_calibracion(
    y_test: pd.Series,
    prob_raw: np.ndarray,
    prob_cal: np.ndarray,
    ruta_salida: str | Path,
    n_bins: int = 10,
) -> Path:
    """Diagrama de fiabilidad: prob predicha vs frecuencia observada por bin."""
    frac_raw, mean_raw = calibration_curve(y_test, prob_raw, n_bins=n_bins, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_test, prob_cal, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="calibración perfecta")
    ax.plot(mean_raw, frac_raw, marker="o", label="raw (sin calibrar)")
    ax.plot(mean_cal, frac_cal, marker="s", label="calibrado")
    ax.set_xlabel("probabilidad media predicha por bin")
    ax.set_ylabel("frecuencia observada de no-show")
    ax.set_title("Curva de calibración (test, cuantiles)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida)
    plt.close(fig)
    logger.info("Curva de calibración guardada en %s", ruta_salida)
    return ruta_salida


def _curva_calibracion_uniforme_filtrada(
    y: np.ndarray,
    prob: np.ndarray,
    bordes: np.ndarray,
    min_obs_bin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curva de calibración manual con bin edges fijos y filtro por tamaño de bin.

    Devuelve `(mean_prob_por_bin, frac_observada_por_bin, n_por_bin)` SÓLO
    para los bins con `n_por_bin >= min_obs_bin`. Los bins con muestra
    insuficiente se descartan: con ≤ 5 observaciones, la fracción observada
    salta entre {0, 0.2, 0.4, ...} y produce los picos espurios que mata
    visualmente la lectura de calibración.

    Bordes se pasan explícitos para que dos llamadas (slice interno + test)
    usen los MISMOS bins → comparación like-with-like en el eje X.
    """
    asignacion = np.digitize(prob, bordes, right=False) - 1
    asignacion = np.clip(asignacion, 0, len(bordes) - 2)

    n_bins = len(bordes) - 1
    medias = np.full(n_bins, np.nan)
    fracs = np.full(n_bins, np.nan)
    cuentas = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        mascara = asignacion == k
        cuentas[k] = int(mascara.sum())
        if cuentas[k] >= min_obs_bin:
            medias[k] = float(prob[mascara].mean())
            fracs[k] = float(y[mascara].mean())

    valido = ~np.isnan(medias)
    return medias[valido], fracs[valido], cuentas[valido]


def plot_curva_calibracion_sanity(
    y_internal: pd.Series,
    prob_cal_internal: np.ndarray,
    y_test: pd.Series,
    prob_cal_test: np.ndarray,
    ruta_salida: str | Path,
    n_bins: int = 10,
    min_obs_bin: int = 30,
) -> Path:
    """Sanity check: compara la curva de calibración en un slice interno del
    periodo de entrenamiento (justo antes de `train_calib`) vs. en el test.

    Los dos lados se evalúan con LOS MISMOS bin edges uniformes — así el
    eje X coincide y la comparación es like-with-like. Se filtran bins con
    < `min_obs_bin` observaciones porque las probabilidades calibradas
    rara vez superan ~0.5 en este modelo y los bins altos quedarían con
    1-2 muestras (frecuencias observadas saltando a {0, 1} → ruido puro
    que oculta la señal de calibración real).

    Sólo se muestran las regiones del rango de probabilidad donde AMBAS
    curvas tienen suficiente muestra para una lectura estable.
    """
    y_i = pd.Series(y_internal).reset_index(drop=True).to_numpy()
    y_t = pd.Series(y_test).reset_index(drop=True).to_numpy()

    rango_max = float(max(prob_cal_internal.max(), prob_cal_test.max()))
    # Bordes uniformes hasta el máximo observado (con margen) → bins más densos
    # donde realmente vive la data
    lim_sup = min(1.0, rango_max + 0.02)
    bordes = np.linspace(0.0, lim_sup, n_bins + 1)

    mean_i, frac_i, n_i = _curva_calibracion_uniforme_filtrada(
        y_i, prob_cal_internal, bordes, min_obs_bin=min_obs_bin
    )
    mean_t, frac_t, n_t = _curva_calibracion_uniforme_filtrada(
        y_t, prob_cal_test, bordes, min_obs_bin=min_obs_bin
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.plot([0, lim_sup], [0, lim_sup], linestyle="--", color="grey",
            label="calibración perfecta")
    ax.plot(mean_i, frac_i, marker="o", label=f"slice interno (n={len(y_i)})")
    ax.plot(mean_t, frac_t, marker="s", label=f"test (n={len(y_t)})")
    ax.set_xlabel("probabilidad media predicha por bin (bins uniformes)")
    ax.set_ylabel("frecuencia observada de no-show")
    ax.set_title(
        f"Robustez de la calibración: slice interno vs test\n"
        f"(bins con < {min_obs_bin} obs descartados; eje zoom al rango efectivo)"
    )
    ax.legend(loc="upper left")
    ax.set_xlim(0, lim_sup)
    ax.set_ylim(0, lim_sup)

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida)
    plt.close(fig)
    logger.info(
        "Sanity check guardado en %s (slice: %d bins válidos, test: %d)",
        ruta_salida, len(mean_i), len(mean_t),
    )
    return ruta_salida


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------

def calcular_shap_values_test(
    modelo: xgb.XGBClassifier, X_test: pd.DataFrame
) -> shap.Explanation:
    """SHAP values sobre TODO el test set vía TreeExplainer.

    Compute estimado en M5 Pro: 20–60 s para ~21k filas. La submuestra para
    visualización se construye después con `submuestra_estratificada_shap`.
    """
    explicador = shap.TreeExplainer(modelo)
    valores = explicador(X_test)
    logger.info("SHAP values calculados sobre test: %s", valores.values.shape)
    return valores


def submuestra_estratificada_shap(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    prob_cal: np.ndarray,
    n: int = N_SHAP_VISUALIZACION,
    seed: int = 42,
) -> np.ndarray:
    """Índices estratificados por (cuartil de riesgo predicho × outcome).

    Para que el `summary_plot` muestre los cuatro segmentos relevantes
    (alto-riesgo correctamente flagueado, alto-riesgo-no-show-anyway,
    bajo-riesgo correcto, bajo-riesgo-sorpresa). Devuelve índices
    posicionales (`iloc`-compatibles) sobre `X_test`.
    """
    cuartil_riesgo = pd.qcut(prob_cal, q=4, labels=False, duplicates="drop")
    estrato = pd.Series(cuartil_riesgo).astype(str) + "_y" + y_test.astype(str).reset_index(drop=True)
    df_strata = pd.DataFrame({"estrato": estrato.values})
    n_real = min(n, len(df_strata))
    rng = np.random.RandomState(seed)
    # Muestreo proporcional por estrato (sin reemplazo dentro de cada uno)
    grupos = df_strata.groupby("estrato").indices
    idxs: list[int] = []
    for estrato_nombre, idx_grupo in grupos.items():
        n_estrato = max(1, int(round(n_real * len(idx_grupo) / len(df_strata))))
        n_estrato = min(n_estrato, len(idx_grupo))
        elegidos = rng.choice(idx_grupo, size=n_estrato, replace=False)
        idxs.extend(elegidos.tolist())
    idxs = np.array(sorted(idxs))
    logger.info(
        "Submuestra SHAP estratificada: %d filas en %d estratos.",
        len(idxs), len(grupos),
    )
    return idxs


def plot_shap_summary(
    shap_values: shap.Explanation,
    X_test: pd.DataFrame,
    idx_muestra: np.ndarray,
    ruta_salida: str | Path,
) -> Path:
    """Beeswarm plot de SHAP sobre la submuestra estratificada."""
    sub = shap_values[idx_muestra]
    # Etiquetas en castellano sin mutar el objeto original
    feature_names_es = [ETIQUETAS_FEATURES.get(c, c) for c in X_test.columns]
    sub.feature_names = feature_names_es

    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(sub, show=False, max_display=18)
    fig = plt.gcf()
    fig.suptitle("Importancia y dirección de las variables (SHAP, beeswarm)", y=0.995)
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida)
    plt.close(fig)
    logger.info("SHAP summary guardado en %s", ruta_salida)
    return ruta_salida


def plot_shap_dependence(
    shap_values: shap.Explanation,
    X_test: pd.DataFrame,
    idx_muestra: np.ndarray,
    features: Iterable[str],
    ruta_salida: str | Path,
) -> Path:
    """Dependence plots para 3–4 variables top en una sola figura."""
    features = list(features)
    sub_shap = shap_values.values[idx_muestra]
    sub_X = X_test.iloc[idx_muestra].reset_index(drop=True)
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, feat in zip(axes, features):
        j = list(X_test.columns).index(feat)
        ax.scatter(sub_X[feat], sub_shap[:, j], s=5, alpha=0.3)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel(ETIQUETAS_FEATURES.get(feat, feat))
        ax.set_ylabel("SHAP value (impacto sobre log-odds de no-show)")
        ax.set_title(ETIQUETAS_FEATURES.get(feat, feat))
    fig.suptitle("Dependencia: feature vs su SHAP value")
    fig.tight_layout()
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida)
    plt.close(fig)
    logger.info("SHAP dependence guardado en %s", ruta_salida)
    return ruta_salida


# Tres arquetipos para explicación local SHAP.
# La clave identifica el caso en código y en el sidecar; la etiqueta es para
# títulos de plots (académica, no coloquial).
CASOS_LOCALES_ETIQUETAS: dict[str, str] = {
    "caso1_alto_riesgo_y_noshow":
        "Caso 1 — alto riesgo predicho y no-show observado",
    "caso2_alto_riesgo_sin_noshow":
        "Caso 2 — alto riesgo predicho pero asistió",
    "caso3_bajo_riesgo_con_noshow":
        "Caso 3 — bajo riesgo predicho con no-show observado",
}


def elegir_casos_locales(
    y_test: pd.Series, prob_cal: np.ndarray, seed: int = 42
) -> dict[str, int]:
    """Tres casos para explicación local (waterfall): uno por cada arquetipo.

    - caso1_alto_riesgo_y_noshow: prob_cal alta (top 10%), no-show=1
      → alto riesgo correctamente priorizado.
    - caso2_alto_riesgo_sin_noshow: prob_cal alta (top 10%), no-show=0
      → alto riesgo predicho pero asistió (falso positivo del punto operativo).
    - caso3_bajo_riesgo_con_noshow: prob_cal baja (bottom 25%), no-show=1
      → bajo riesgo con no-show observado (falso negativo).

    Las claves siguen la convención de NB05/NB06 (no coloquiales).
    """
    rng = np.random.RandomState(seed)
    p90 = np.quantile(prob_cal, 0.90)
    p25 = np.quantile(prob_cal, 0.25)
    y = y_test.reset_index(drop=True).values

    def primero(mascara: np.ndarray) -> int:
        candidatos = np.where(mascara)[0]
        return int(rng.choice(candidatos))

    casos = {
        "caso1_alto_riesgo_y_noshow": primero((prob_cal >= p90) & (y == 1)),
        "caso2_alto_riesgo_sin_noshow": primero((prob_cal >= p90) & (y == 0)),
        "caso3_bajo_riesgo_con_noshow": primero((prob_cal <= p25) & (y == 1)),
    }
    logger.info("Casos locales SHAP: %s", casos)
    return casos


def plot_shap_local_waterfall_individuales(
    shap_values: shap.Explanation,
    X_test: pd.DataFrame,
    casos: Mapping[str, int],
    prob_cal: np.ndarray,
    directorio_salida: str | Path,
    prefijo: str = "nb03_shap_local_waterfall",
    max_display: int = 8,
) -> dict[str, Path]:
    """Tres waterfall plots independientes — un PNG por caso local.

    Cada uno se renderiza como figura standalone: el SHAP de `shap.plots.waterfall`
    no compone bien dentro de subplots compartidos (los labels se solapan entre
    paneles); separarlos resuelve el problema de raíz.

    El título de cada figura añade la **probabilidad calibrada de no-show** del
    caso, junto al `f(x)` en log-odds que SHAP muestra por defecto. La probabilidad
    es la lectura interpretable para un lector de negocio; el log-odds es la
    escala nativa del modelo.

    Devuelve dict {clave_caso → ruta del PNG generado}.
    """
    feature_names_es = [ETIQUETAS_FEATURES.get(c, c) for c in X_test.columns]
    directorio_salida = Path(directorio_salida)
    directorio_salida.parent.mkdir(parents=True, exist_ok=True)
    rutas: dict[str, Path] = {}

    for clave, idx in casos.items():
        # SHAP gestiona su propia figura — no instanciamos un Figure previo.
        explicacion = shap_values[idx]
        explicacion.feature_names = feature_names_es
        shap.plots.waterfall(explicacion, show=False, max_display=max_display)
        fig = plt.gcf()
        # Tamaño grande para que las etiquetas respiren incluso con 8 features
        fig.set_size_inches(9, 6)
        etiqueta = CASOS_LOCALES_ETIQUETAS.get(clave, clave)
        prob = float(prob_cal[idx])
        fig.suptitle(
            f"{etiqueta}\nProbabilidad calibrada de no-show: {prob:.1%}",
            fontsize=12, y=1.02,
        )
        ruta = directorio_salida / f"{prefijo}_{clave}_v1.png"
        fig.savefig(ruta, bbox_inches="tight")
        plt.close(fig)
        logger.info("SHAP local %s guardado en %s (prob_cal=%.3f)", clave, ruta, prob)
        rutas[clave] = ruta

    return rutas


# --------------------------------------------------------------------------
# Persistencia de artefactos
# --------------------------------------------------------------------------

def guardar_modelo(modelo: xgb.XGBClassifier, ruta: str | Path) -> Path:
    """Persiste el modelo XGBoost en formato JSON nativo (reproducible)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    modelo.save_model(str(ruta))
    logger.info("Modelo XGBoost guardado en %s", ruta)
    return ruta


def guardar_calibrador(
    calibrador: CalibratedClassifierCV, ruta: str | Path
) -> Path:
    """Persiste el calibrador (pickle — sklearn no tiene formato nativo)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("wb") as f:
        pickle.dump(calibrador, f)
    logger.info("Calibrador guardado en %s", ruta)
    return ruta


def exportar_probabilidades_test(
    df_test: pd.DataFrame,
    prob_cal: np.ndarray,
    prob_raw: np.ndarray,
    ruta_csv: str | Path,
) -> Path:
    """Guarda PatientId, AppointmentID, prob no-show calibrada y raw.

    Es la entrada principal de NB05 (segmentación) y NB06 (Monte Carlo).
    Se persisten también las probas raw para auditoría del efecto de la
    calibración aguas abajo.
    """
    salida = pd.DataFrame({
        "PatientId": df_test["PatientId"].values,
        "AppointmentID": df_test["AppointmentID"].values,
        "AppointmentDay": df_test["AppointmentDay"].values,
        "Showed_up": df_test["Showed_up"].values,
        "noshow": (1 - df_test["Showed_up"]).values.astype(int),
        "prob_noshow_raw": prob_raw,
        "prob_noshow_calibrada": prob_cal,
    })
    ruta_csv = Path(ruta_csv)
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(ruta_csv, index=False)
    logger.info("Probabilidades exportadas a %s (%d filas)", ruta_csv, len(salida))
    return ruta_csv


# --------------------------------------------------------------------------
# Sidecar de metadatos NB03
# --------------------------------------------------------------------------

def escribir_metadatos_nb03(
    ruta_json: str | Path,
    *,
    seed: int,
    n_train: int,
    n_train_xgb: int,
    n_train_calib: int,
    n_test: int,
    fecha_corte_xgb_calib: pd.Timestamp,
    folds: list[FoldWalkForward],
    tabla_grid: pd.DataFrame,
    mejores_params: Mapping[str, Any],
    tabla_calibradores: pd.DataFrame,
    metricas_test: MetricasTest,
    tabla_confusion: pd.DataFrame,
    umbral_top20: float,
    features_usadas: list[str],
    scale_pos_weight: float,
    casos_locales: Mapping[str, int],
) -> dict[str, Any]:
    """Persiste los números metodológicamente críticos de NB03."""
    metadatos: dict[str, Any] = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "features_usadas": features_usadas,
        "n_features": len(features_usadas),
        "n_train": n_train,
        "n_train_xgb": n_train_xgb,
        "n_train_calib": n_train_calib,
        "n_test": n_test,
        "fecha_corte_xgb_calib": fecha_corte_xgb_calib.date().isoformat(),
        "scale_pos_weight": float(scale_pos_weight),
        "folds_walk_forward": [f.resumen() for f in folds],
        "mejores_hiperparametros": {
            "max_depth": int(mejores_params["max_depth"]),
            "learning_rate": float(mejores_params["learning_rate"]),
            "best_iteration_mean_cv": float(mejores_params["best_iteration_mean"]),
            "auc_mean_cv": float(mejores_params["auc_mean"]),
            "auc_std_cv": float(mejores_params["auc_std"]),
        },
        "grid_resultados": tabla_grid.to_dict(orient="records"),
        "calibracion": {
            "metodo_elegido": tabla_calibradores.attrs["metodo_elegido"],
            "motivo": tabla_calibradores.attrs["motivo_seleccion"],
            "tabla": tabla_calibradores.drop(columns=["seleccionado"]).to_dict(orient="records"),
        },
        "test_metricas": metricas_test.as_dict(),
        "umbral_top20": float(umbral_top20),
        "matriz_confusion": tabla_confusion.to_dict(orient="records"),
        "casos_locales_idx_iloc": dict(casos_locales),
    }
    ruta_json = Path(ruta_json)
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    with ruta_json.open("w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Metadatos NB03 guardados en %s", ruta_json)
    return metadatos


# --------------------------------------------------------------------------
# Estilo (compartido con NB02)
# --------------------------------------------------------------------------

def configurar_estilo() -> None:
    """Mismo estilo que NB02 para coherencia visual del TFG."""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="colorblind",
        font_scale=1.0,
    )
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })
