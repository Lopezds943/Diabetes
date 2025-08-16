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

# fetch dataset
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

# data (as pandas dataframes)
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

# Combinar en un solo DataFrame
df = pd.concat([X, y], axis=1)

# Remplazar los ? por np.nan
df = df.replace('?', np.nan)

# Quitar espacios accidentales y uniformar mayúsculas en columnas tipo string
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].astype(str).str.strip()

# Mostrar primeras filas
df
