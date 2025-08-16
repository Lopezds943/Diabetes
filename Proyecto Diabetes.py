# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from matplotlib import pyplot
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
import streamlit as st

st.title("📊 Análisis de Rehospitalización en Pacientes con Diabetes (1999–2008)")

st.markdown("""
Este dataset proviene del repositorio **UCI Machine Learning** y contiene más de **100 mil registros** 
de pacientes hospitalizados en Estados Unidos entre 1999 y 2008 con diagnóstico de diabetes.  

Incluye información demográfica (edad, raza, género), clínica (diagnósticos ICD-9, 
resultados de laboratorio, medicamentos), y administrativa (tipo de admisión, alta hospitalaria).  

El objetivo principal del análisis es **explorar patrones asociados al reingreso hospitalario** 
de pacientes diabéticos y aplicar técnicas de **bioestadística multivariada** 
como *PCA* (Análisis de Componentes Principales) y *MCA* (Análisis de Correspondencias Múltiples).
""")

##############################################################################
# Consolidar Dataset
##############################################################################
# fetch dataset
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

# data (as pandas dataframes)
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

# Combinar en un solo DataFrame
df = pd.concat([X, y], axis=1)

##############################################################################
# Limipar Dastaset
##############################################################################

# Remplazar los ? por np.nan
df = df.replace('?', np.nan)

# Quitar espacios accidentales y uniformar mayúsculas en columnas tipo string
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].astype(str).str.strip()

#Duplicados
df = df.drop_duplicates()

# Solo queremos Conservemos el primer encuentro por paciente
if {'patient_nbr', 'encounter_id'}.issubset(df.columns):
    df = df.sort_values(['patient_nbr','encounter_id'], ascending=[True, True])
    df = df.drop_duplicates(subset='patient_nbr', keep='first')

# Género inválido (el dataset trae 'Unknown/Invalid')
if 'gender' in df.columns:
    df['gender'] = df['gender'].replace({'Unknown/Invalid': np.nan})

st.subheader("🧹 Justificación de la limpieza de datos")

st.markdown("""
Durante el proceso de preparación de los datos se aplicaron dos criterios principales de limpieza:

### 1. Eliminación de variables
- **`examide` y `citoglipton`**: presentan **varianza nula**, ya que todos los registros tienen el mismo valor ("No").  
- **`weight`**: más del **95% de los registros están vacíos**, lo que impide su uso en análisis robustos.  
- **`payer_code`**: aunque contiene información sobre el tipo de asegurador, alrededor del **40% de los valores están ausentes**.  
  Por esta razón se descartó para garantizar consistencia en los modelos estadísticos.  

👉 La eliminación de estas variables asegura que el análisis no se vea afectado por columnas sin variabilidad o con demasiados valores faltantes.

### 2. Eliminación de registros específicos
Para el análisis de rehospitalización en pacientes con diabetes, cada fila del dataset corresponde a un **encuentro hospitalario** (`encounter_id`) y un mismo paciente (`patient_nbr`) puede aparecer varias veces.

**Decisión metodológica:** conservar **solo el primer encuentro** de cada paciente.

**Justificación estadística y clínica**
- **Independencia de observaciones:** múltiples filas del mismo paciente generan dependencia intrapaciente y sesgan los contrastes.""")

# Variables numericas 
# Convierte a numéricas las que deben serlo (si quedaron como object por NA o parsing)
num_maybe = [
    'time_in_hospital','num_lab_procedures','num_procedures','num_medications',
    'number_outpatient','number_emergency','number_inpatient','number_diagnoses',
    'admission_type_id','discharge_disposition_id','admission_source_id'
]
for c in num_maybe:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Asegurar no-negatividad en contadores (si algo raro aparece, lo pasamos a NA)
for c in [col for col in num_maybe if col in df.columns]:
    df.loc[df[c] < 0, c] = np.nan

# Columna objetivo
if 'readmitted' in df.columns:
    # Normalizamos las palabras de la columna readmitted
    df['readmitted'] = df['readmitted'].str.upper()
    # Creacion de columna de readmision binaria
    df['readmitted_any'] = np.where(df['readmitted'].isin(['<30','>30']), 1, 0)


# Convertir categorías en ordinales (cuando tienen un orden natural).
if 'A1Cresult' in df.columns:
    map_a1c = {'None': np.nan, 'Norm': 0, '>7': 1, '>8': 2}
    df['A1Cresult_ord'] = df['A1Cresult'].map(map_a1c)

if 'max_glu_serum' in df.columns:
    map_glu = {'None': np.nan, 'Norm': 0, '>200': 1, '>300': 2}
    df['max_glu_serum_ord'] = df['max_glu_serum'].map(map_glu)


# Mostrar primeras filas
df

