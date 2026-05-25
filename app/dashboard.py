"""
Dashboard Ejecutivo — aiXtensa
Visualización integral de los resultados del análisis de no-shows.
"""

import sys
from pathlib import Path

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUTA_PROYECTO / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

from utils import (
    cargar_datos_limpios,
    RUTA_MODELOS, RUTA_REPORTES,
    INGRESO_POR_CONSULTA, COSTE_SMS, TASA_CONVERSION_SMS,
    formato_euros, formato_porcentaje,
)

# ============================================================
# Configuración
# ============================================================
st.set_page_config(
    page_title="Dashboard aiXtensa",
    page_icon="📈",
    layout="wide",
)

COLOR_ASISTIO = "#4CAF50"
COLOR_NO_ASISTIO = "#F44336"
COLOR_PRIMARIO = "#2196F3"
COLOR_SECUNDARIO = "#FF9800"

# ============================================================
# Carga de datos
# ============================================================

@st.cache_data
def cargar_todo():
    """Carga todos los datos necesarios."""
    df = cargar_datos_limpios()

    ruta_pred = RUTA_MODELOS / "predicciones_noshow.csv"
    if ruta_pred.exists():
        df_pred = pd.read_csv(ruta_pred)
        df["prob_noshow"] = df_pred["prob_noshow"].values
        df["pred_noshow"] = df_pred["pred_noshow"].values
    else:
        df["prob_noshow"] = np.nan
        df["pred_noshow"] = np.nan

    ruta_pricing = RUTA_REPORTES / "propuesta_pricing.json"
    pricing = None
    if ruta_pricing.exists():
        with open(ruta_pricing, "r", encoding="utf-8") as f:
            pricing = json.load(f)

    return df, pricing


try:
    df, pricing = cargar_todo()
    datos_disponibles = True
except FileNotFoundError as e:
    st.error(f"Error cargando datos: {e}")
    st.info("Ejecuta primero los notebooks 01-05 para generar los datos procesados.")
    datos_disponibles = False
    st.stop()

# ============================================================
# Métricas globales
# ============================================================
n_citas = len(df)
n_pacientes = df["id_paciente"].nunique()
tasa_noshow = 1 - df["asistio"].mean()
n_noshows = int(n_citas * tasa_noshow)
ingreso_base = INGRESO_POR_CONSULTA["default"]
valor_en_riesgo = n_noshows * ingreso_base

# ============================================================
# Header: KPIs
# ============================================================
st.title("Dashboard Ejecutivo — aiXtensa")
st.markdown("Análisis integral de no-shows en citas médicas")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Citas", f"{n_citas:,}")
with col2:
    st.metric("Pacientes Únicos", f"{n_pacientes:,}")
with col3:
    st.metric("Tasa de No-Show", f"{tasa_noshow:.1%}")
with col4:
    st.metric("No-Shows Totales", f"{n_noshows:,}")
with col5:
    st.metric("Ingresos en Riesgo", formato_euros(valor_en_riesgo))

st.markdown("---")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Patrones Temporales",
    "Segmentación",
    "Análisis Geográfico",
    "Rendimiento del Modelo",
    "Impacto Financiero",
])

# ============================================================
# Tab 1: Patrones Temporales
# ============================================================
with tab1:
    st.subheader("Patrones Temporales de No-Shows")

    col_izq, col_der = st.columns(2)

    with col_izq:
        # Día de la semana
        dias_orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dias_es = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        mapa_dias = dict(zip(dias_orden, dias_es))

        dia_stats = (
            df.groupby("nombre_dia")
            .agg(tasa_noshow=("asistio", lambda x: 1 - x.mean()))
            .reset_index()
        )
        dia_stats["dia_es"] = dia_stats["nombre_dia"].map(mapa_dias)
        dia_stats["orden"] = dia_stats["nombre_dia"].map({d: i for i, d in enumerate(dias_orden)})
        dia_stats = dia_stats.sort_values("orden")

        fig = px.bar(
            dia_stats, x="dia_es", y="tasa_noshow",
            title="Tasa de No-Show por Día de la Semana",
            labels={"dia_es": "Día", "tasa_noshow": "Tasa No-Show"},
            color="tasa_noshow", color_continuous_scale="RdYlGn_r",
            text=dia_stats["tasa_noshow"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_der:
        # Antelación
        orden = ["mismo_dia", "corto_1_3d", "medio_4_14d", "largo_15d+"]
        etiquetas = ["Mismo día", "1-3 días", "4-14 días", "15+ días"]

        ant_stats = (
            df.groupby("categoria_antelacion")
            .agg(tasa_noshow=("asistio", lambda x: 1 - x.mean()))
            .reindex(orden)
            .reset_index()
        )
        ant_stats["etiqueta"] = etiquetas

        fig = px.bar(
            ant_stats, x="etiqueta", y="tasa_noshow",
            title="Tasa de No-Show por Antelación",
            labels={"etiqueta": "Antelación", "tasa_noshow": "Tasa No-Show"},
            color="tasa_noshow", color_continuous_scale="RdYlGn_r",
            text=ant_stats["tasa_noshow"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Tendencia semanal
    df["fecha_cita"] = pd.to_datetime(df["fecha_cita"])
    semanal = (
        df.groupby(df["fecha_cita"].dt.to_period("W").astype(str))
        .agg(
            total_citas=("asistio", "count"),
            tasa_noshow=("asistio", lambda x: 1 - x.mean()),
        )
        .reset_index()
    )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Volumen Semanal", "Tasa de No-Show"],
    )

    fig.add_trace(
        go.Bar(x=semanal["fecha_cita"], y=semanal["total_citas"],
               marker_color=COLOR_PRIMARIO, name="Citas"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=semanal["fecha_cita"], y=semanal["tasa_noshow"],
                   mode="lines+markers", line=dict(color=COLOR_NO_ASISTIO, width=2),
                   name="Tasa No-Show"),
        row=2, col=1,
    )

    fig.update_layout(template="plotly_white", height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 2: Segmentación
# ============================================================
with tab2:
    st.subheader("Segmentación de Pacientes")

    col_izq, col_der = st.columns(2)

    with col_izq:
        # Grupo de edad
        edad_stats = (
            df.groupby("grupo_edad", observed=True)
            .agg(
                total=("asistio", "count"),
                tasa_noshow=("asistio", lambda x: 1 - x.mean()),
            )
            .reset_index()
        )

        fig = px.bar(
            edad_stats, x="grupo_edad", y="tasa_noshow",
            title="Tasa de No-Show por Grupo de Edad",
            labels={"grupo_edad": "Grupo", "tasa_noshow": "Tasa No-Show"},
            color="tasa_noshow", color_continuous_scale="RdYlGn_r",
            text=edad_stats["tasa_noshow"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_der:
        # Condiciones médicas
        condiciones = ["hipertension", "diabetes", "alcoholismo", "discapacidad", "beca_social"]
        datos_cond = []
        for cond in condiciones:
            for val in [0, 1]:
                subset = df[df[cond] == val]
                datos_cond.append({
                    "condicion": cond.replace("_", " ").title(),
                    "tiene": "Sí" if val == 1 else "No",
                    "tasa": 1 - subset["asistio"].mean(),
                })

        df_cond = pd.DataFrame(datos_cond)

        fig = px.bar(
            df_cond, x="condicion", y="tasa", color="tiene", barmode="group",
            title="Tasa de No-Show por Condición",
            labels={"condicion": "", "tasa": "Tasa No-Show", "tiene": "Tiene condición"},
            color_discrete_map={"Sí": COLOR_NO_ASISTIO, "No": COLOR_PRIMARIO},
        )
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Segmentación de riesgo (si hay predicciones)
    if not df["prob_noshow"].isna().all():
        st.subheader("Segmentación por Riesgo del Modelo")

        df["segmento_riesgo"] = pd.cut(
            df["prob_noshow"],
            bins=[0, 0.15, 0.30, 0.50, 1.0],
            labels=["Bajo (<15%)", "Medio (15-30%)", "Alto (30-50%)", "Muy Alto (>50%)"],
        )

        riesgo_stats = (
            df.groupby("segmento_riesgo", observed=True)
            .agg(
                n_citas=("asistio", "count"),
                tasa_noshow_real=("asistio", lambda x: 1 - x.mean()),
            )
            .reset_index()
        )

        fig = px.bar(
            riesgo_stats, x="segmento_riesgo", y="n_citas",
            title="Distribución de Citas por Segmento de Riesgo",
            labels={"segmento_riesgo": "Segmento", "n_citas": "Nº de Citas"},
            color="tasa_noshow_real", color_continuous_scale="RdYlGn_r",
            text=riesgo_stats.apply(
                lambda r: f"{r['n_citas']:,}\n({r['tasa_noshow_real']:.0%} no-show)", axis=1
            ),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 3: Análisis Geográfico
# ============================================================
with tab3:
    st.subheader("Análisis Geográfico por Barrio")

    barrio_stats = (
        df.groupby("barrio")
        .agg(
            total_citas=("asistio", "count"),
            tasa_noshow=("asistio", lambda x: 1 - x.mean()),
        )
        .reset_index()
    )

    min_citas = st.slider("Mínimo de citas para mostrar barrio", 50, 500, 100)
    barrio_filtrado = barrio_stats[barrio_stats["total_citas"] >= min_citas]

    col_izq, col_der = st.columns(2)

    with col_izq:
        top_15 = barrio_filtrado.nlargest(15, "tasa_noshow")

        fig = px.bar(
            top_15, y="barrio", x="tasa_noshow", orientation="h",
            title="Top 15 Barrios con Mayor No-Show",
            labels={"barrio": "", "tasa_noshow": "Tasa No-Show"},
            color="tasa_noshow", color_continuous_scale="Reds",
            text=top_15["tasa_noshow"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col_der:
        bot_15 = barrio_filtrado.nsmallest(15, "tasa_noshow")

        fig = px.bar(
            bot_15, y="barrio", x="tasa_noshow", orientation="h",
            title="Top 15 Barrios con Menor No-Show",
            labels={"barrio": "", "tasa_noshow": "Tasa No-Show"},
            color="tasa_noshow", color_continuous_scale="Greens_r",
            text=bot_15["tasa_noshow"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: volumen vs tasa
    fig = px.scatter(
        barrio_filtrado, x="total_citas", y="tasa_noshow",
        size="total_citas", hover_name="barrio",
        title="Barrios: Volumen de Citas vs Tasa de No-Show",
        labels={"total_citas": "Nº de Citas", "tasa_noshow": "Tasa No-Show"},
        color="tasa_noshow", color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 4: Rendimiento del Modelo
# ============================================================
with tab4:
    st.subheader("Rendimiento del Modelo Predictivo")

    if df["prob_noshow"].isna().all():
        st.warning("No hay predicciones disponibles. Ejecuta el notebook 03 primero.")
    else:
        col_izq, col_der = st.columns(2)

        with col_izq:
            # Distribución de probabilidades
            fig = px.histogram(
                df, x="prob_noshow", nbins=50,
                title="Distribución de Probabilidades Predichas",
                labels={"prob_noshow": "P(No-Show)", "count": "Frecuencia"},
                color_discrete_sequence=[COLOR_PRIMARIO],
            )
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_der:
            # Calibración: probabilidad predicha vs tasa real
            df["bucket_prob"] = pd.cut(df["prob_noshow"], bins=10)
            calibracion = (
                df.groupby("bucket_prob", observed=True)
                .agg(
                    prob_media=("prob_noshow", "mean"),
                    tasa_real=("asistio", lambda x: 1 - x.mean()),
                    n=("asistio", "count"),
                )
                .reset_index()
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=calibracion["prob_media"], y=calibracion["tasa_real"],
                mode="markers+lines", name="Modelo",
                marker=dict(size=calibracion["n"] / calibracion["n"].max() * 30 + 5),
                line=dict(color=COLOR_PRIMARIO),
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                line=dict(color="gray", dash="dash"),
            ))

            fig.update_layout(
                title="Curva de Calibración",
                xaxis_title="Probabilidad Predicha",
                yaxis_title="Tasa Real de No-Show",
                template="plotly_white", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Importancia de features (si existe la imagen)
        ruta_shap = RUTA_PROYECTO / "outputs" / "figuras" / "shap_importancia_global.png"
        if ruta_shap.exists():
            st.image(str(ruta_shap), caption="Importancia Global de Features (SHAP)")

# ============================================================
# Tab 5: Impacto Financiero
# ============================================================
with tab5:
    st.subheader("Impacto Financiero")

    if pricing is not None:
        # Propuesta de precios
        df_tiers = pd.DataFrame(pricing["tiers"])

        col1, col2, col3 = st.columns(3)

        for i, (_, tier) in enumerate(df_tiers.iterrows()):
            with [col1, col2, col3][i]:
                st.markdown(f"### {tier['Tier']}")
                st.metric("Precio/mes", formato_euros(tier["Precio/mes (€)"]))
                st.metric("ROI esperado", f"{tier['ROI esperado']:.1f}x")
                st.metric("No-shows prevenidos/mes", f"{tier['Noshows prevenidos/mes']:,}")

        st.markdown("---")

        # Percentiles de precio
        st.subheader("Bandas de Precio (Monte Carlo)")

        perc_data = []
        for tier_name, percs in pricing["percentiles_precio"].items():
            for p_name, p_val in percs.items():
                perc_data.append({"Tier": tier_name, "Percentil": p_name, "Precio": p_val})

        df_perc = pd.DataFrame(perc_data)

        fig = go.Figure()
        colores_tier = {"Starter": "#2196F3", "Professional": "#4CAF50", "Enterprise": "#FF9800"}

        for tier_name in df_perc["Tier"].unique():
            subset = df_perc[df_perc["Tier"] == tier_name]
            fig.add_trace(go.Bar(
                x=subset["Percentil"], y=subset["Precio"],
                name=tier_name, marker_color=colores_tier.get(tier_name, "gray"),
            ))

        fig.update_layout(
            title="Percentiles de Precio por Tier (Simulación Monte Carlo)",
            xaxis_title="Percentil", yaxis_title="Precio Mensual (€)",
            template="plotly_white", height=450,
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No hay datos de pricing. Ejecuta el notebook 05 primero.")

    # Valor total en riesgo vs recuperable
    noshows_prevenibles = int(n_noshows * TASA_CONVERSION_SMS)
    valor_recuperable = noshows_prevenibles * ingreso_base

    fig = go.Figure(go.Waterfall(
        x=["Ingresos\nen riesgo", "Prevenibles\ncon SMS", "No prevenibles"],
        y=[-valor_en_riesgo, valor_recuperable, -(valor_en_riesgo - valor_recuperable)],
        text=[
            formato_euros(valor_en_riesgo),
            formato_euros(valor_recuperable),
            formato_euros(valor_en_riesgo - valor_recuperable),
        ],
        textposition="outside",
        connector=dict(line=dict(color="gray")),
        increasing=dict(marker_color=COLOR_ASISTIO),
        decreasing=dict(marker_color=COLOR_NO_ASISTIO),
    ))

    fig.update_layout(
        title="Valor Económico: Ingresos en Riesgo vs Recuperables",
        yaxis_title="€",
        template="plotly_white", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)
