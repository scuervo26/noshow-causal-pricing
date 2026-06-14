# Predicción del valor económico de la automatización en salud
### Inferencia causal y simulación Monte Carlo para la gestión de inasistencias en clínicas privadas

Trabajo de Fin de Grado (TFG) en la Universidad Pontificia Comillas (ICADE). El proyecto estima cuánto valor económico puede recuperar una estrategia automatizada de recordatorios por SMS sobre las citas médicas perdidas, y traduce esa estimación en una recomendación de pricing basado en valor para una clínica privada española.

Es también la base empírica de **aiXtensa**, un proyecto de automatización para el sector salud que desarrollo en paralelo.

---

## El problema

Las inasistencias (*no-shows*) en consulta externa erosionan el margen de la clínica: huecos que no se recuperan, tiempo clínico ocioso y demanda de pacientes sin atender. Normalmente se mezclan dos preguntas que conviene separar:

1. **¿Quién** tiene más probabilidad de faltar a su cita? (aprendizaje supervisado)
2. **¿Cuánto** cambiaría realmente ese resultado un recordatorio? (un problema *causal*: datos observacionales con sesgo de selección)

El proyecto las mantiene separadas a propósito y luego combina ambas respuestas en una simulación financiera Monte Carlo que arrastra la incertidumbre desde la estimación causal hasta la tarifa recomendada.

---

## Metodología

| Capa | Método | Resultado |
|---|---|---|
| Modelo de riesgo | XGBoost calibrado (isotónica / Platt) | Probabilidad de no-show por paciente |
| Interpretabilidad | SHAP (TreeExplainer, test completo) | Atribución global y local de variables |
| Capa causal | IPW estabilizado + overlap weights, bootstrap por paciente | Efecto del SMS sobre la asistencia bajo ajuste, con intervalo de confianza |
| Segmentación | K-means sobre riesgo calibrado + variables de comportamiento | Arquetipos de paciente con riesgo basal diferenciado |
| Pricing | Monte Carlo, 10.000 iteraciones | Distribución del valor recuperable (€) por cita y mes |
| Parámetros de mercado | Webscraping ligero de tarifas de clínicas privadas españolas | Rangos empíricos anclados al mercado español |

El modelo de pricing Monte Carlo es el núcleo del trabajo: convierte una estimación metodológicamente cuidada en algo accionable para el responsable de una clínica. La recomendación se entrega como tarifas de cuota fija + variable (porcentaje del valor recuperado) en los percentiles P10 / P50 / P90 de la distribución simulada, de modo que el cliente elige su propio perfil de riesgo en lugar de recibir un precio único.

Decisiones metodológicas clave: `SMS_received` se excluye del modelo de riesgo (es la variable de tratamiento); `Neighbourhood` se codifica por frecuencia (sin fuga del outcome); la partición train/test es temporal porque los pacientes se repiten; el resultado del análisis causal se reporta como *asociación bajo ajuste*, no como prueba de causalidad.

---

## Datos

[Medical Appointment No-Shows — Kaggle (Joni Arroba)](https://www.kaggle.com/datasets/joniarroba/noshowappointments): ~107.000 citas de consulta externa, Vitória (ES, Brasil), 2016. Se usa como proxy de comportamiento; los parámetros económicos se recalibran a España con datos del INE y las tarifas obtenidas en el Notebook 07.

El CSV bruto no se versiona. Tras clonar, colócalo en `datos/brutos/healthcare_noshows.csv`.

---

## Estructura del repositorio

```
notebooks/   01 → 07: limpieza, EDA, XGBoost+SHAP, IPW, segmentación, Monte Carlo, scraping
src/         preparacion.py, eda.py, modelado.py, causal.py, pricing.py, rutas.py — lógica de los notebooks
app/         prototipos en Streamlit (calculadora de ROI, dashboard)
datos/       brutos/ (sin versionar), procesados/ (regenerables desde NB01)
outputs/     figuras/, modelos/, reportes/, bootstrap/ (regenerables desde los notebooks)
```

La narrativa de los notebooks está en español (el TFG se deposita en una universidad española). En el código, los identificadores siguen el idioma del dominio (español) o de las librerías (inglés) según corresponde.

---

## Reproducción

```bash
git clone https://github.com/scuervo26/noshow-causal-pricing.git
cd noshow-causal-pricing

# Coloca el CSV de Kaggle en: datos/brutos/healthcare_noshows.csv

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # o requirements-lock.txt para las versiones exactas

jupyter notebook notebooks/
```

Todos los procesos estocásticos (XGBoost, K-means, bootstrap, Monte Carlo) usan semilla fija, y los artefactos intermedios están versionados (`train_v1.csv`, `xgboost_v1.json`, `ate_bootstrap_v1.npy`) para que las re-ejecuciones sean estables.

---

## Estado del proyecto

| Notebook | Etapa | Estado |
|---|---|---|
| `01_limpieza_y_variables` | Limpieza + variables + partición temporal | Completado |
| `02_exploracion_datos` | EDA + diagnóstico del sesgo de selección | Completado |
| `03_modelo_xgboost_calibracion_shap` | XGBoost calibrado + SHAP | Completado |
| `04_inferencia_causal_ipw` | IPW estabilizado + overlap weights + bootstrap | Completado |
| `05_segmentacion_pacientes` | Segmentación de pacientes (K-means) | Pendiente |
| `06_pricing_montecarlo` | Simulación de pricing Monte Carlo | Completado |
| `07_webscraping_mercado` | Parámetros de mercado (España) | Pendiente |
| `app/` | Calculadora de ROI + dashboard en Streamlit | Prototipo |

---

## Autor

**Sergio Cuervo Arango** — *Administración y Dirección de Empresas + Business Analytics* (E-2 + BA), Universidad Pontificia Comillas (ICADE). Fundador de aiXtensa.

Licencia MIT (ver [`LICENSE`](LICENSE)). El dataset no se redistribuye; obtenlo de Kaggle bajo su propia licencia.
