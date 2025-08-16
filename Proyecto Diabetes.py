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

# Mostrar primeras filas
df
