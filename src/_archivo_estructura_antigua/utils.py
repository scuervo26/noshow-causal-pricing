"""
Utilidades compartidas para el proyecto TFG - aiXtensa
Análisis de No-Shows en citas médicas
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# Rutas del proyecto
# ============================================================
RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS_RAW = RUTA_PROYECTO / "data" / "raw"
RUTA_DATOS_PROCESADOS = RUTA_PROYECTO / "data" / "processed"
RUTA_FIGURAS = RUTA_PROYECTO / "outputs" / "figuras"
RUTA_MODELOS = RUTA_PROYECTO / "outputs" / "modelos"
RUTA_REPORTES = RUTA_PROYECTO / "outputs" / "reportes"

ARCHIVO_CSV_RAW = RUTA_DATOS_RAW / "healthcare_noshows.csv"
ARCHIVO_CSV_LIMPIO = RUTA_DATOS_PROCESADOS / "healthcare_noshows_limpio.csv"


# ============================================================
# Constantes del negocio (supuestos respaldados por literatura)
# ============================================================

# Ingresos medios por consulta por especialidad (€)
INGRESO_POR_CONSULTA = {
    "dental": 80.0,
    "fisioterapia": 45.0,
    "medicina_general": 60.0,
    "default": 60.0,
}

# Coste de enviar un SMS (€)
COSTE_SMS = 0.045

# Tasa de conversión estimada: % de no-shows prevenidos tras recordatorio
# Basado en literatura: SMS reminders reduce no-shows 25-40%
TASA_CONVERSION_SMS = 0.30

# Perfiles de clínica por tamaño
PERFILES_CLINICA = {
    "starter": {
        "nombre": "Starter",
        "medicos": 2,
        "citas_mes": 500,
        "descripcion": "Clínica pequeña (1-2 médicos)",
    },
    "professional": {
        "nombre": "Professional",
        "medicos": 4,
        "citas_mes": 1500,
        "descripcion": "Clínica mediana (3-5 médicos)",
    },
    "enterprise": {
        "nombre": "Enterprise",
        "medicos": 8,
        "citas_mes": 3000,
        "descripcion": "Clínica grande (5+ médicos)",
    },
}


# ============================================================
# Funciones de carga y limpieza
# ============================================================

def cargar_datos_raw() -> pd.DataFrame:
    """Carga el dataset original desde CSV."""
    df = pd.read_csv(
        ARCHIVO_CSV_RAW,
        parse_dates=["ScheduledDay", "AppointmentDay"],
    )
    return df


def cargar_datos_limpios() -> pd.DataFrame:
    """Carga el dataset procesado. Lanza error si no existe."""
    if not ARCHIVO_CSV_LIMPIO.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo limpio en {ARCHIVO_CSV_LIMPIO}. "
            "Ejecuta primero el notebook 01_preparacion_datos.ipynb"
        )
    df = pd.read_csv(
        ARCHIVO_CSV_LIMPIO,
        parse_dates=["ScheduledDay", "AppointmentDay"],
    )
    return df


# ============================================================
# Funciones de feature engineering
# ============================================================

def crear_features_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features basadas en la fecha de la cita."""
    return df.assign(
        dia_semana=df["AppointmentDay"].dt.dayofweek,
        nombre_dia=df["AppointmentDay"].dt.day_name(),
        es_fin_de_semana=df["AppointmentDay"].dt.dayofweek.isin([5, 6]).astype(int),
        mes=df["AppointmentDay"].dt.month,
        semana=df["AppointmentDay"].dt.isocalendar().week.astype(int),
    )


def crear_categorias_antelacion(df: pd.DataFrame) -> pd.DataFrame:
    """Categoriza Date.diff en buckets de antelación."""
    condiciones = [
        df["Date.diff"] == 0,
        df["Date.diff"].between(1, 3),
        df["Date.diff"].between(4, 14),
        df["Date.diff"] > 14,
    ]
    categorias = ["mismo_dia", "corto_1_3d", "medio_4_14d", "largo_15d+"]
    return df.assign(
        categoria_antelacion=np.select(condiciones, categorias, default="desconocido")
    )


def crear_grupos_edad(df: pd.DataFrame) -> pd.DataFrame:
    """Categoriza la edad en grupos clínicamente relevantes."""
    bins = [0, 17, 35, 60, 200]
    labels = ["pediatrico", "adulto_joven", "adulto", "senior"]
    return df.assign(
        grupo_edad=pd.cut(df["Age"], bins=bins, labels=labels, right=True)
    )


def crear_score_comorbilidad(df: pd.DataFrame) -> pd.DataFrame:
    """Suma las condiciones médicas en un score de comorbilidad."""
    cols_medicas = ["Hipertension", "Diabetes", "Alcoholism", "Handcap"]
    return df.assign(
        score_comorbilidad=df[cols_medicas].sum(axis=1)
    )


def crear_historial_paciente(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features basadas en el historial del paciente.
    IMPORTANTE: Solo usa información PREVIA a cada cita para evitar data leakage.
    """
    df = df.sort_values(["PatientId", "AppointmentDay"]).copy()

    # Número de cita del paciente (1ª, 2ª, 3ª...)
    df["num_cita_paciente"] = df.groupby("PatientId").cumcount() + 1

    # No-shows previos acumulados (excluye la cita actual)
    df["no_show_flag"] = (~df["Showed_up"]).astype(int)
    df["noshows_previos"] = (
        df.groupby("PatientId")["no_show_flag"]
        .cumsum()
        - df["no_show_flag"]  # Restar la cita actual
    )

    # Tasa histórica de no-show (previas solamente)
    citas_previas = df["num_cita_paciente"] - 1
    df["tasa_noshow_historica"] = np.where(
        citas_previas > 0,
        df["noshows_previos"] / citas_previas,
        0.0,  # Primera cita: sin historial
    )

    # Es primera cita (flag)
    df["es_primera_cita"] = (df["num_cita_paciente"] == 1).astype(int)

    df = df.drop(columns=["no_show_flag"])
    return df


def crear_riesgo_barrio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la tasa media de no-show por barrio (target encoding).
    Usa suavizado para barrios con pocas observaciones.
    """
    tasa_global = (~df["Showed_up"]).mean()
    stats_barrio = df.groupby("Neighbourhood").agg(
        total_citas=("Showed_up", "count"),
        noshows=("Showed_up", lambda x: (~x).sum()),
    )
    # Suavizado: mezcla con la media global según el tamaño de muestra
    factor_suavizado = 100  # Mínimo de observaciones para confiar en la tasa local
    stats_barrio["tasa_noshow_barrio"] = (
        (stats_barrio["noshows"] + factor_suavizado * tasa_global)
        / (stats_barrio["total_citas"] + factor_suavizado)
    )
    mapa_riesgo = stats_barrio["tasa_noshow_barrio"].to_dict()
    return df.assign(
        riesgo_barrio=df["Neighbourhood"].map(mapa_riesgo)
    )


# ============================================================
# Funciones de valor económico
# ============================================================

def calcular_valor_por_cita(
    prob_noshow: float,
    ingreso_consulta: float = 60.0,
    coste_sms: float = COSTE_SMS,
    tasa_conversion: float = TASA_CONVERSION_SMS,
) -> dict:
    """
    Calcula el valor económico de intervenir en una cita.

    Retorna dict con valor_bruto, coste, valor_neto.
    """
    valor_bruto = prob_noshow * tasa_conversion * ingreso_consulta
    valor_neto = valor_bruto - coste_sms
    return {
        "valor_bruto": valor_bruto,
        "coste_sms": coste_sms,
        "valor_neto": valor_neto,
    }


def calcular_valor_clinica_mensual(
    citas_mes: int,
    tasa_noshow: float,
    ingreso_consulta: float = 60.0,
    estrategia: str = "inteligente",
    umbral_prob: float = 0.3,
    prob_por_cita: np.ndarray = None,
) -> dict:
    """
    Calcula el valor mensual de aiXtensa para una clínica.

    Estrategias:
    - 'sin_sms': baseline, no se envía nada
    - 'sms_todos': se envía SMS a todas las citas
    - 'inteligente': solo se envía a citas con prob > umbral
    """
    noshows_esperados = int(citas_mes * tasa_noshow)

    if estrategia == "sin_sms":
        return {
            "sms_enviados": 0,
            "noshows_prevenidos": 0,
            "valor_recuperado": 0.0,
            "coste_sms_total": 0.0,
            "valor_neto": 0.0,
        }

    if estrategia == "sms_todos":
        sms_enviados = citas_mes
        noshows_prevenidos = int(noshows_esperados * TASA_CONVERSION_SMS)
        valor_recuperado = noshows_prevenidos * ingreso_consulta
        coste_total = sms_enviados * COSTE_SMS
        return {
            "sms_enviados": sms_enviados,
            "noshows_prevenidos": noshows_prevenidos,
            "valor_recuperado": valor_recuperado,
            "coste_sms_total": coste_total,
            "valor_neto": valor_recuperado - coste_total,
        }

    if estrategia == "inteligente":
        if prob_por_cita is not None:
            citas_alto_riesgo = int((prob_por_cita > umbral_prob).sum())
        else:
            # Estimación: ~40% de citas superan el umbral típico
            citas_alto_riesgo = int(citas_mes * 0.4)

        sms_enviados = citas_alto_riesgo
        # Mayor tasa de conversión al focalizar en alto riesgo
        tasa_conversion_ajustada = min(TASA_CONVERSION_SMS * 1.2, 0.5)
        noshows_prevenidos = int(
            min(citas_alto_riesgo * tasa_noshow, noshows_esperados)
            * tasa_conversion_ajustada
        )
        valor_recuperado = noshows_prevenidos * ingreso_consulta
        coste_total = sms_enviados * COSTE_SMS
        return {
            "sms_enviados": sms_enviados,
            "noshows_prevenidos": noshows_prevenidos,
            "valor_recuperado": valor_recuperado,
            "coste_sms_total": coste_total,
            "valor_neto": valor_recuperado - coste_total,
        }

    raise ValueError(f"Estrategia desconocida: {estrategia}")


# ============================================================
# Funciones de visualización
# ============================================================

def formato_euros(valor: float) -> str:
    """Formatea un valor numérico como euros."""
    return f"€{valor:,.2f}"


def formato_porcentaje(valor: float) -> str:
    """Formatea un valor decimal como porcentaje."""
    return f"{valor:.1%}"
