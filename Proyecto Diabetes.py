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

st.subheader("📑 Diccionario de variables principales")

st.markdown("""
### Identificación y demografía
- **encounter_id**: Identificador único del encuentro hospitalario.  
- **patient_nbr**: Identificador único de cada paciente.  
- **race**: Raza del paciente (Caucasian, AfricanAmerican, Asian, etc.).  
- **gender**: Sexo del paciente (Male, Female).  
- **age**: Grupo de edad en intervalos de 10 años (ej. [50-60)).  

### Información administrativa
- **admission_type_id**: Tipo de admisión (urgencias, electiva, etc.).  
- **discharge_disposition_id**: Tipo de alta (hogar, transferencia, fallecimiento, etc.).  
- **admission_source_id**: Fuente de admisión (referencia médica, urgencias, etc.).  
- **time_in_hospital**: Días de estancia en el hospital.  
- **payer_code**: Tipo de asegurador (Medicare, BlueCross, etc.) [eliminada por alta cantidad de faltantes].  
- **medical_specialty**: Especialidad médica del médico tratante.  

### Información clínica
- **num_lab_procedures**: Número de procedimientos de laboratorio realizados.  
- **num_procedures**: Número de procedimientos médicos no relacionados con laboratorio.  
- **num_medications**: Número de medicamentos administrados durante la estancia.  
- **number_outpatient**: Número de visitas ambulatorias previas.  
- **number_emergency**: Número de visitas a urgencias previas.  
- **number_inpatient**: Número de hospitalizaciones previas.  
- **number_diagnoses**: Número de diagnósticos registrados en el encuentro.  

### Diagnósticos
- **diag_1, diag_2, diag_3**: Diagnósticos principales y secundarios (códigos ICD-9).  

### Resultados clínicos
- **A1Cresult**: Resultado de hemoglobina glicosilada A1C (`None`, `Norm`, `>7`, `>8`).  
- **max_glu_serum**: Nivel máximo de glucosa en suero (`None`, `Norm`, `>200`, `>300`).  

### Tratamiento farmacológico
- **metformin, insulin, glipizide, etc.**: Variables de medicamentos (valores: `No`, `Steady`, `Up`, `Down`).  
- **change**: Indica si hubo cambio en la medicación (`Yes`, `No`).  
- **diabetesMed**: Indica si el paciente recibió medicación para diabetes (`Yes`, `No`).  

### Variable objetivo
- **readmitted**: Indica si el paciente fue readmitido en los 30 días siguientes al alta.  
   - Valores: `NO`, `>30` (después de 30 días), `<30` (dentro de 30 días).  
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


#Eliminar columnas con muchos NA / varianza nula
# =======================
na_rate = df.isna().mean()
MISSING_COL_THRESH = 0.50

cols_many_na = na_rate[na_rate > MISSING_COL_THRESH].index.tolist()
low_var_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

force_drop = [c for c in ['weight','examide','citoglipton','payer_code'] if c in df.columns]
# Si quieres conservar 'payer_code' para EDA, quítala de force_drop.

cols_to_drop = sorted(set(cols_many_na + low_var_cols + force_drop))
df = df.drop(columns=cols_to_drop, errors='ignore').copy()

st.markdown("**Columnas eliminadas por criterio de limpieza:**")
st.write(cols_to_drop if cols_to_drop else "Ninguna")

# Eliminar filas con género NaN (después de reemplazar Unknown/Invalid)
# =======================
if 'gender' in df.columns:
    before_gender = len(df)
    df = df[~df['gender'].isna()].copy()
    st.caption(f"Filas eliminadas por género inválido: {before_gender - len(df):,}")


# Medicaciones como categóricas ordenadas
# =======================
from pandas.api.types import CategoricalDtype
med_possible = {'No','Steady','Up','Down'}
med_order = CategoricalDtype(categories=['No','Steady','Down','Up'], ordered=True)

med_cols = [c for c in df.columns
            if df[c].dtype == 'object' and set(df[c].dropna().unique()).issubset(med_possible)]
for c in med_cols:
    df[c] = df[c].astype(med_order)


#Edad como ordinal (para análisis/EDA)
# =======================
if 'age' in df.columns:
    age_order = CategoricalDtype(
        categories=['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)',
                    '[60-70)','[70-80)','[80-90)','[90-100)'],
        ordered=True
    )
    df['age'] = df['age'].astype(age_order)


df
