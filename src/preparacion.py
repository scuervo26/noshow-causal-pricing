"""Notebook 01 — limpieza de datos y construcción de variables.

Implementa las reglas metodológicas que evitan fugas de información:
    - Codificación por frecuencia (NUNCA por outcome) en Neighbourhood.
    - Regla estricta de tiempo de decisión para historial del paciente:
      sólo se usan citas previas con AppointmentDay < ScheduledDay_actual.
    - División temporal por percentil de AppointmentDay (no aleatoria).

Este módulo se diseña como librería pura: el notebook 01 orquesta y narra.

NOTA SOBRE NOMBRES DE FUNCIONES: este módulo usa nombres en español por
coherencia con la narrativa del notebook y el dominio del TFG. Los
identificadores de bibliotecas estándar (pandas, numpy, etc.) se mantienen
en inglés. Si se decide migrar a inglés, basta con renombrar; la lógica no
depende del idioma.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Bandas de edad fijas (bins clínicos)
AGE_BAND_BINS = [-0.5, 18, 35, 55, 70, 200]
AGE_BAND_LABELS = ["0-18", "19-35", "36-55", "56-70", "70+"]

# Bins operativos para lead_time. NB05 los necesita como dimensión de clustering;
# están elegidos para reflejar políticas de comunicación distintas (mismo día,
# corto plazo, medio plazo, largo plazo). El borde inferior es -inf para
# absorber las pocas filas con anomalía AppointmentDay < ScheduledDay
# (tratadas operativamente como "mismo_dia").
LEAD_TIME_BIN_EDGES = [-np.inf, 0.5, 7.5, 14.5, np.inf]
LEAD_TIME_BIN_LABELS = ["mismo_dia", "1-7d", "8-14d", "15+d"]

# Variables binarias del dataset original que componen la comorbilidad
COLS_COMORBILIDAD = ["Hipertension", "Diabetes", "Alcoholism", "Handcap"]


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------

def cargar_datos_brutos(ruta_csv: str | Path) -> pd.DataFrame:
    """Carga el CSV original de Kaggle sin transformaciones."""
    df = pd.read_csv(ruta_csv)
    logger.info("Cargado dataset bruto: %d filas, %d columnas", *df.shape)
    return df


# --------------------------------------------------------------------------
# Verificación de codificación del outcome
# --------------------------------------------------------------------------

def verificar_codificacion_showed_up(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica e impone la convención: Showed_up=1 → asistió, 0 → no-show.

    Es la primera comprobación obligatoria de cualquier notebook de
    modelado. Una codificación invertida que sobreviva hasta la defensa
    invalida todos los signos de ATE, SHAP y Monte Carlo aguas abajo.

    Devuelve una copia con Showed_up convertida a entero 0/1.
    """
    serie = df["Showed_up"]
    if serie.dtype == bool:
        attended = int(serie.sum())
        no_shows = int((~serie).sum())
    elif pd.api.types.is_integer_dtype(serie):
        attended = int((serie == 1).sum())
        no_shows = int((serie == 0).sum())
    else:
        raise ValueError(
            f"Tipo no soportado para Showed_up: {serie.dtype}. "
            "Esperado bool o entero."
        )

    total = attended + no_shows
    if total != len(df):
        raise ValueError(
            f"Showed_up tiene valores inesperados: total={total}, n={len(df)}"
        )

    pct_attended = attended / total
    logger.info(
        "Showed_up: %d asistieron (%.1f%%), %d no-shows (%.1f%%)",
        attended, pct_attended * 100, no_shows, (1 - pct_attended) * 100,
    )

    # ~80% de asistencia es la firma del dataset Kaggle. Una desviación
    # extrema sugiere codificación invertida (asistió=0, no-show=1).
    if not 0.65 < pct_attended < 0.90:
        raise ValueError(
            f"Tasa de asistencia inesperada ({pct_attended:.1%}). "
            "Verificar codificación de Showed_up antes de continuar."
        )

    df = df.copy()
    df["Showed_up"] = df["Showed_up"].astype(int)
    return df


# --------------------------------------------------------------------------
# Limpieza
# --------------------------------------------------------------------------

def limpiar_valores_imposibles(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas físicamente imposibles y registra anomalías.

    Sólo se eliminan filas con Age < 0 (edad negativa imposible). Las
    anomalías de orden temporal (AppointmentDay < ScheduledDay) se
    registran pero no se eliminan: pueden reflejar mecánicas reales del
    sistema de citación (re-agendamiento, errores de tipeo) y borrarlas
    introduciría sesgo no controlado.
    """
    df = df.copy()
    n_inicial = len(df)

    # PatientId llega como float64 en el CSV (los IDs grandes se serializaron
    # con notación científica). Convertimos a int64 para evitar sorpresas en
    # comparaciones de igualdad — relevante en NB04, donde el bootstrap por
    # clúster reagrupará por PatientId.
    #
    # Auditoría: una minoría de filas del CSV tienen PatientId con parte
    # fraccionaria genuina (no ruido de float). Las detectamos, las contamos,
    # verificamos que la truncación no colisione con IDs enteros existentes,
    # y persistimos el conteo en df.attrs para que llegue al sidecar de NB01.
    parte_fraccionaria = df["PatientId"] - df["PatientId"].astype("int64")
    mask_frac = parte_fraccionaria.abs() > 1e-9
    n_patient_id_frac = int(mask_frac.sum())
    if n_patient_id_frac:
        ids_enteros = set(df.loc[~mask_frac, "PatientId"].astype("int64").tolist())
        ids_truncados = df.loc[mask_frac, "PatientId"].astype("int64").tolist()
        colisiones = [pid for pid in ids_truncados if pid in ids_enteros]
        if colisiones:
            raise ValueError(
                f"Truncar PatientId fraccionarios produciría {len(colisiones)} "
                f"colisión(es) con IDs enteros existentes — revisar el CSV. "
                f"Primeros ejemplos: {colisiones[:5]}"
            )
        logger.info(
            "PatientId con parte fraccionaria (truncados a int64, sin colisiones): %d filas",
            n_patient_id_frac,
        )
    df["PatientId"] = df["PatientId"].astype("int64")
    df.attrs["n_patient_id_fraccionarios"] = n_patient_id_frac

    n_edad_negativa = int((df["Age"] < 0).sum())
    if n_edad_negativa:
        df = df.loc[df["Age"] >= 0].reset_index(drop=True)
    logger.info("Eliminadas %d filas con Age < 0", n_edad_negativa)

    # Anomalía: cita programada para fecha pasada respecto a la programación
    anomalia_orden = pd.to_datetime(df["AppointmentDay"]) < pd.to_datetime(df["ScheduledDay"])
    n_anomalias = int(anomalia_orden.sum())
    logger.info(
        "AppointmentDay < ScheduledDay (anomalía documentada, no eliminada): %d filas (%.2f%%)",
        n_anomalias, 100 * n_anomalias / max(len(df), 1),
    )

    # Handcap: el plan menciona posible multi-nivel. En este dataset es booleano
    # (sólo 0/1 ó True/False), por tanto no se recodifica. Se documenta.
    valores_handcap = sorted(pd.unique(df["Handcap"]).tolist())
    logger.info(
        "Valores únicos de Handcap: %s (binario, no requiere recodificación)",
        valores_handcap,
    )

    logger.info(
        "Limpieza: %d → %d filas (%d eliminadas)",
        n_inicial, len(df), n_inicial - len(df),
    )
    return df


def parsear_fechas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte ScheduledDay y AppointmentDay a datetime y verifica Date.diff."""
    df = df.copy()
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

    diff_calculado = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days
    if "Date.diff" in df.columns:
        n_inconsistencias = int((diff_calculado != df["Date.diff"]).sum())
        logger.info(
            "Date.diff: %d inconsistencias entre la columna existente y el recálculo "
            "(se sobrescribe con el recálculo, autoritativo)",
            n_inconsistencias,
        )
    df["Date.diff"] = diff_calculado
    return df


# --------------------------------------------------------------------------
# Variables derivadas (no requieren historial)
# --------------------------------------------------------------------------

def crear_features_basicas(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas que no requieren historial del paciente.

    No se usan Showed_up ni SMS_received como entradas: el outcome y el
    tratamiento se mantienen separados de la ingeniería de variables.
    """
    df = df.copy()

    # Lead time
    df["lead_time"] = df["Date.diff"]
    df["lead_time_bin"] = pd.cut(
        df["lead_time"],
        bins=LEAD_TIME_BIN_EDGES,
        labels=LEAD_TIME_BIN_LABELS,
        include_lowest=True,
    ).astype(str)

    # Bandas de edad
    df["age_band"] = pd.cut(
        df["Age"],
        bins=AGE_BAND_BINS,
        labels=AGE_BAND_LABELS,
        include_lowest=True,
    ).astype(str)

    # Comorbilidad
    df["comorbidity_count"] = df[COLS_COMORBILIDAD].astype(int).sum(axis=1)
    df["chronic_flag"] = (df["comorbidity_count"] > 0).astype(int)

    # Variables temporales del momento de programación (decision-time)
    df["scheduled_weekday"] = df["ScheduledDay"].dt.dayofweek
    df["scheduled_hour"] = df["ScheduledDay"].dt.hour
    df["scheduled_month"] = df["ScheduledDay"].dt.month

    # Variables temporales del momento de la cita
    df["appointment_weekday"] = df["AppointmentDay"].dt.dayofweek
    df["appointment_month"] = df["AppointmentDay"].dt.month

    # En este CSV las marcas de tiempo se exportaron como fecha pura, por lo
    # que scheduled_hour es constantemente 0 y no aporta señal. La eliminamos
    # explícitamente para no contaminar las listas de features de NB03/NB04.
    # Si en una versión futura del CSV apareciera la hora, este bloque deja
    # un rastro auditable del comportamiento.
    if df["scheduled_hour"].nunique() <= 1:
        valor_constante = df["scheduled_hour"].iloc[0]
        logger.info(
            "scheduled_hour es constante (todo %s) — eliminada de las salidas",
            valor_constante,
        )
        df = df.drop(columns=["scheduled_hour"])

    return df


# --------------------------------------------------------------------------
# Historial del paciente con regla estricta de tiempo de decisión
# --------------------------------------------------------------------------

def crear_features_historial_paciente(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula prior_appointment_count, prior_noshow_rate, is_first_visit.

    Regla estricta de tiempo de decisión:
    para cada cita del paciente p con ScheduledDay = S, sólo se contabilizan
    citas previas del mismo paciente cuyo AppointmentDay < S — es decir, el
    resultado de esas citas ya se conocía cuando se tomó la decisión sobre
    la cita actual.

    Esta regla es estrictamente más fuerte que un expanding window ordenado
    por AppointmentDay: ancla al *tiempo de decisión*, no al orden de
    observación. Permite, por ejemplo, que una cita programada hoy con
    fecha dentro de un mes ya conozca el resultado de otra cita del mismo
    paciente programada después pero ejecutada antes.

    Implementación O(n log n): por paciente, ordenamos su historial por
    AppointmentDay y usamos searchsorted para localizar el corte.

    Limitación importante (CENSURA POR LA IZQUIERDA): el dataset cubre una
    ventana temporal corta y no incluye el historial del paciente anterior
    a esa ventana. Las tres variables que produce esta función describen
    sólo el historial *observado*, no el real:
        - `is_first_visit = 1` ⇔ "no consta cita previa ya ocurrida dentro
           del periodo del dataset y antes de la fecha de programación de
           esta cita". NO equivale a "paciente nuevo en la clínica".
        - `prior_appointment_count` y `prior_noshow_rate` son resúmenes del
          historial observable, no del historial absoluto.
    Esta limitación debe citarse explícitamente como "censura por la
    izquierda" en la sección de limitaciones del TFG escrito.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)

    counts = np.zeros(n, dtype=int)
    rates = np.full(n, np.nan, dtype=float)

    indices_por_paciente = df.groupby("PatientId").indices

    appt_all = df["AppointmentDay"].values
    sched_all = df["ScheduledDay"].values
    showed_all = df["Showed_up"].values.astype(int)

    for idx_grupo in indices_por_paciente.values():
        appt_p = appt_all[idx_grupo]
        sched_p = sched_all[idx_grupo]
        showed_p = showed_all[idx_grupo]

        # Ordenar el historial del paciente por AppointmentDay (orden de
        # disponibilidad del outcome).
        orden = np.argsort(appt_p)
        appt_ordenados = appt_p[orden]
        showed_ordenados = showed_p[orden]
        # Acumulados de no-shows en ese orden cronológico
        cum_noshow = np.cumsum(1 - showed_ordenados)

        for i, idx in enumerate(idx_grupo):
            cutoff = sched_p[i]
            own_appt = appt_p[i]
            own_outcome = showed_p[i]

            # Cuántas entradas tienen AppointmentDay estrictamente < cutoff
            # (incluye potencialmente la propia fila si AppointmentDay <
            # ScheduledDay — anomalía documentada en el dataset).
            k = int(np.searchsorted(appt_ordenados, cutoff, side="left"))

            if own_appt < cutoff:
                # Anomalía: la propia fila figura entre las "previas". La
                # excluimos para respetar la regla estricta de tiempo de
                # decisión (un outcome no puede ser su propio antecedente).
                n_prior = k - 1
                if n_prior > 0:
                    no_show_sum = cum_noshow[k - 1] - (1 - own_outcome)
                    rates[idx] = no_show_sum / n_prior
                # n_prior == 0 → primera visita efectiva; rate queda NaN
            else:
                # Caso normal: own_appt >= cutoff, no entra en el conteo.
                n_prior = k
                if n_prior > 0:
                    rates[idx] = cum_noshow[k - 1] / n_prior

            counts[idx] = n_prior

    df["prior_appointment_count"] = counts
    df["prior_noshow_rate"] = rates
    df["is_first_visit"] = (counts == 0).astype(int)

    n_first = int(df["is_first_visit"].sum())
    logger.info(
        "Historial paciente: %d primeras visitas (%.1f%%); prior_noshow_rate es NaN para esas filas (correcto, no son cero)",
        n_first, 100 * n_first / len(df),
    )
    return df


# --------------------------------------------------------------------------
# División temporal
# --------------------------------------------------------------------------

def dividir_temporal(
    df: pd.DataFrame, percentil: float = 0.80
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """División cronológica por AppointmentDay según percentil indicado.

    Train: AppointmentDay <= corte. Test: AppointmentDay > corte.
    El dataset completo (full_clean) NO se divide aquí — es el llamador
    quien decide qué guardar (train_v1.csv, test_v1.csv, full_clean_v1.csv).
    """
    if not 0 < percentil < 1:
        raise ValueError(f"percentil debe estar en (0, 1); recibido: {percentil}")

    fecha_corte = df["AppointmentDay"].quantile(percentil, interpolation="lower")
    train = df.loc[df["AppointmentDay"] <= fecha_corte].reset_index(drop=True)
    test = df.loc[df["AppointmentDay"] > fecha_corte].reset_index(drop=True)
    logger.info(
        "División temporal en %s (P%d): train=%d (%.1f%%), test=%d (%.1f%%)",
        pd.Timestamp(fecha_corte).date(), int(percentil * 100),
        len(train), 100 * len(train) / len(df),
        len(test), 100 * len(test) / len(df),
    )
    return train, test, pd.Timestamp(fecha_corte)


# --------------------------------------------------------------------------
# Codificación por frecuencia de Neighbourhood
# --------------------------------------------------------------------------

def ajustar_codificacion_frecuencia_barrio(train_df: pd.DataFrame) -> dict[str, float]:
    """Frecuencia (proporción) de cada barrio en el set de entrenamiento.

    NUNCA usa Showed_up como entrada (eso filtraría el outcome al modelo).
    Los barrios que aparecen sólo en test reciben 0 al aplicarse, lo que
    equivale a tratarlos como muy raros (interpretación natural de la
    codificación por frecuencia).
    """
    counts = train_df["Neighbourhood"].value_counts(normalize=True)
    logger.info(
        "Codificación por frecuencia ajustada en train: %d barrios únicos",
        len(counts),
    )
    return counts.to_dict()


def aplicar_codificacion_frecuencia_barrio(
    df: pd.DataFrame, mapping: Mapping[str, float]
) -> pd.DataFrame:
    """Aplica la codificación por frecuencia ajustada en train.

    Barrios fuera del vocabulario reciben 0.0 (no aparecen en entrenamiento
    → frecuencia desconocida → tratados como muy raros).
    """
    df = df.copy()
    encoded = df["Neighbourhood"].map(mapping)
    n_oov = int(encoded.isna().sum())
    df["neighbourhood_encoded"] = encoded.fillna(0.0)
    if n_oov:
        logger.info(
            "%d filas con Neighbourhood fuera de train (codificadas como 0.0)",
            n_oov,
        )
    return df


# --------------------------------------------------------------------------
# Sidecar de metadatos (defensibilidad: números clave persistidos)
# --------------------------------------------------------------------------

def escribir_metadatos_nb01(
    ruta_json: str | Path,
    *,
    seed: int,
    df_bruto: pd.DataFrame,
    df_full: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    fecha_corte: pd.Timestamp,
    mapping_barrio: Mapping[str, float],
) -> dict[str, Any]:
    """Persiste los números metodológicamente críticos de NB01 a JSON.

    Útil para defensa: permite reproducir la narrativa de EDA y la sección
    de metodología del TFG escrito aun si el notebook se vuelve a ejecutar
    meses después con un entorno ligeramente distinto.

    `df_bruto` es el dataframe inmediatamente tras `cargar_datos_brutos`,
    antes de cualquier limpieza — sirve para registrar decisiones defensivas
    cuyo conteo final es cero (ej. eliminación de Age<0 en este CSV).
    """
    barrios_train = set(mapping_barrio.keys())
    barrios_test = set(df_test["Neighbourhood"].unique())
    n_oov_test = len(barrios_test - barrios_train)

    # Cierres de fricción plan↔dato — números que confirman que los pasos
    # defensivos del plan se ejecutaron incluso cuando no había nada que
    # ejecutar en este CSV concreto. Útiles para el TFG escrito.
    n_edad_negativa_eliminadas = int((df_bruto["Age"] < 0).sum())
    valores_unicos_handcap = sorted(
        [bool(v) if isinstance(v, (bool, np.bool_)) else v
         for v in pd.unique(df_bruto["Handcap"])]
    )
    scheduled_hour_eliminada = "scheduled_hour" not in df_full.columns

    metadatos = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_filas_brutas": int(len(df_bruto)),
        "n_filas_total": int(len(df_full)),
        "n_filas_train": int(len(df_train)),
        "n_filas_test": int(len(df_test)),
        "fecha_corte_appointmentday": fecha_corte.date().isoformat(),
        "pct_atendieron": float((df_full["Showed_up"] == 1).mean()),
        "n_primeras_visitas": int(df_full["is_first_visit"].sum()),
        "pct_primeras_visitas": float(df_full["is_first_visit"].mean()),
        "n_barrios_train": len(barrios_train),
        "n_barrios_test_oov": n_oov_test,
        "n_anomalia_appt_lt_sched": int(
            (df_full["AppointmentDay"] < df_full["ScheduledDay"]).sum()
        ),
        "media_prior_noshow_rate": float(
            df_full["prior_noshow_rate"].dropna().mean()
        ),
        # Cierres de fricción plan↔dato (defensa)
        "n_edad_negativa_eliminadas": n_edad_negativa_eliminadas,
        "valores_unicos_handcap": valores_unicos_handcap,
        "scheduled_hour_eliminada_por_constante": scheduled_hour_eliminada,
        # Truncación de PatientId fraccionarios (verificada sin colisiones)
        "n_patient_id_fraccionarios": int(
            df_full.attrs.get("n_patient_id_fraccionarios", 0)
        ),
    }
    ruta_json = Path(ruta_json)
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    with ruta_json.open("w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=2, ensure_ascii=False)
    logger.info("Metadatos NB01 guardados en %s", ruta_json)
    return metadatos
