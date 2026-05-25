"""
Calculadora ROI — aiXtensa
Herramienta interactiva para estimar el retorno de inversión
de los servicios de automatización para clínicas sanitarias.
"""

import sys
from pathlib import Path

# Configurar path del proyecto
RUTA_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUTA_PROYECTO / "src"))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    INGRESO_POR_CONSULTA, COSTE_SMS, TASA_CONVERSION_SMS,
    PERFILES_CLINICA, formato_euros,
)

# ============================================================
# Configuración de la página
# ============================================================
st.set_page_config(
    page_title="Calculadora ROI — aiXtensa",
    page_icon="📊",
    layout="wide",
)

st.title("Calculadora ROI — aiXtensa")
st.markdown(
    "Estima el **retorno de inversión** de los servicios de recordatorio "
    "inteligente de aiXtensa para tu clínica."
)

# ============================================================
# Panel lateral: Inputs del usuario
# ============================================================
st.sidebar.header("Datos de tu clínica")

tipo_clinica = st.sidebar.selectbox(
    "Tipo de clínica",
    options=["Dental", "Fisioterapia", "Medicina General"],
    index=0,
)

mapa_tipo = {
    "Dental": "dental",
    "Fisioterapia": "fisioterapia",
    "Medicina General": "medicina_general",
}

n_medicos = st.sidebar.slider(
    "Número de médicos/profesionales", min_value=1, max_value=20, value=3
)

citas_mes = st.sidebar.number_input(
    "Citas por mes (aproximadas)", min_value=50, max_value=10000, value=1000, step=50
)

tasa_noshow_usuario = st.sidebar.slider(
    "Tasa de no-show actual (%)", min_value=5, max_value=50, value=20
) / 100

ingreso_consulta = st.sidebar.number_input(
    "Ingreso medio por consulta (€)",
    min_value=10.0, max_value=300.0,
    value=float(INGRESO_POR_CONSULTA.get(mapa_tipo[tipo_clinica], 60.0)),
    step=5.0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Parámetros avanzados**")

tasa_conversion = st.sidebar.slider(
    "Eficacia del recordatorio (%)",
    min_value=10, max_value=50, value=int(TASA_CONVERSION_SMS * 100),
    help="Porcentaje de no-shows que se previenen con recordatorio inteligente",
) / 100

alpha_pricing = st.sidebar.slider(
    "% del valor como precio aiXtensa",
    min_value=5, max_value=30, value=15,
) / 100

# ============================================================
# Cálculos
# ============================================================

# Determinar tier automáticamente
if citas_mes <= 750:
    tier = "starter"
elif citas_mes <= 2000:
    tier = "professional"
else:
    tier = "enterprise"

cuotas_fijas = {"starter": 49, "professional": 99, "enterprise": 199}
cuota_fija = cuotas_fijas[tier]

# Escenario: Sin intervención
noshows_mes = int(citas_mes * tasa_noshow_usuario)
ingreso_perdido = noshows_mes * ingreso_consulta

# Escenario: SMS a todos
sms_todos_enviados = citas_mes
sms_todos_prevenidos = int(noshows_mes * tasa_conversion)
sms_todos_recuperado = sms_todos_prevenidos * ingreso_consulta
sms_todos_coste = sms_todos_enviados * COSTE_SMS
sms_todos_neto = sms_todos_recuperado - sms_todos_coste

# Escenario: SMS inteligente (aiXtensa)
pct_alto_riesgo = 0.40  # ~40% de citas son de alto riesgo
sms_smart_enviados = int(citas_mes * pct_alto_riesgo)
tasa_conversion_smart = min(tasa_conversion * 1.2, 0.50)
sms_smart_prevenidos = int(noshows_mes * 0.8 * tasa_conversion_smart)
sms_smart_recuperado = sms_smart_prevenidos * ingreso_consulta
sms_smart_coste = sms_smart_enviados * COSTE_SMS
sms_smart_neto = sms_smart_recuperado - sms_smart_coste

# Precio aiXtensa
valor_generado = sms_smart_recuperado
precio_variable = valor_generado * alpha_pricing
precio_aixtensa = cuota_fija + precio_variable

# ROI
beneficio_clinica = sms_smart_neto - precio_aixtensa
roi = beneficio_clinica / precio_aixtensa if precio_aixtensa > 0 else 0
payback_meses = 1  # Valor neto positivo desde el primer mes

# ============================================================
# Layout principal
# ============================================================

# KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="No-shows prevenidos/mes",
        value=f"{sms_smart_prevenidos:,}",
        delta=f"de {noshows_mes:,} totales",
    )

with col2:
    st.metric(
        label="Ingresos recuperados/mes",
        value=formato_euros(sms_smart_recuperado),
        delta=f"+{sms_smart_recuperado/max(ingreso_perdido,1):.0%} vs sin SMS",
    )

with col3:
    st.metric(
        label="Precio aiXtensa/mes",
        value=formato_euros(precio_aixtensa),
        delta=f"Tier: {tier.title()}",
    )

with col4:
    st.metric(
        label="ROI de tu inversión",
        value=f"{roi:.1f}x",
        delta=f"{formato_euros(beneficio_clinica)}/mes neto",
    )

st.markdown("---")

# ============================================================
# Comparación de estrategias
# ============================================================
st.subheader("Comparación de Estrategias")

col_izq, col_der = st.columns(2)

with col_izq:
    fig = go.Figure()

    estrategias = ["Sin SMS", "SMS a todos", "SMS inteligente\n(aiXtensa)"]
    valores_netos = [0, sms_todos_neto, sms_smart_neto]
    colores = ["#9E9E9E", "#FF9800", "#4CAF50"]

    fig.add_trace(go.Bar(
        x=estrategias, y=valores_netos,
        marker_color=colores,
        text=[formato_euros(v) for v in valores_netos],
        textposition="outside",
    ))

    fig.update_layout(
        title="Valor Neto Mensual por Estrategia",
        yaxis_title="Valor Neto (€/mes)",
        template="plotly_white", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_der:
    fig = go.Figure()

    sms_counts = [0, sms_todos_enviados, sms_smart_enviados]

    fig.add_trace(go.Bar(
        x=estrategias, y=sms_counts,
        marker_color=colores,
        text=[f"{s:,}" for s in sms_counts],
        textposition="outside",
    ))

    fig.update_layout(
        title="SMS Enviados por Estrategia",
        yaxis_title="SMS/mes",
        template="plotly_white", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Proyección anual
# ============================================================
st.subheader("Proyección Anual")

meses = list(range(1, 13))
beneficio_acumulado = [beneficio_clinica * m for m in meses]
coste_acumulado = [precio_aixtensa * m for m in meses]
valor_acumulado = [sms_smart_recuperado * m for m in meses]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=meses, y=valor_acumulado,
    mode="lines+markers", name="Ingresos recuperados",
    line=dict(color="#4CAF50", width=2),
    fill="tozeroy", fillcolor="rgba(76,175,80,0.1)",
))

fig.add_trace(go.Scatter(
    x=meses, y=coste_acumulado,
    mode="lines+markers", name="Coste aiXtensa",
    line=dict(color="#F44336", width=2, dash="dash"),
))

fig.add_trace(go.Scatter(
    x=meses, y=beneficio_acumulado,
    mode="lines+markers", name="Beneficio neto",
    line=dict(color="#2196F3", width=3),
))

fig.update_layout(
    title="Proyección Acumulada a 12 Meses",
    xaxis_title="Mes", yaxis_title="€ acumulados",
    template="plotly_white", height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Resumen ejecutivo
# ============================================================
st.subheader("Resumen Ejecutivo")

st.markdown(f"""
| Concepto | Mensual | Anual |
|----------|---------|-------|
| No-shows prevenidos | {sms_smart_prevenidos:,} | {sms_smart_prevenidos * 12:,} |
| Ingresos recuperados | {formato_euros(sms_smart_recuperado)} | {formato_euros(sms_smart_recuperado * 12)} |
| Precio aiXtensa | {formato_euros(precio_aixtensa)} | {formato_euros(precio_aixtensa * 12)} |
| **Beneficio neto** | **{formato_euros(beneficio_clinica)}** | **{formato_euros(beneficio_clinica * 12)}** |
| **ROI** | **{roi:.1f}x** | **{roi:.1f}x** |
""")

st.info(
    f"Por cada euro invertido en aiXtensa, tu clínica recupera "
    f"**{formato_euros(roi + 1)}** en ingresos que de otra forma se perderían."
)
