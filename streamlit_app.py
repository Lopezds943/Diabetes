# streamlit_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Opcional: intentar cargar dataset desde UCI si no hay archivo local ni subida
def load_uci_if_needed():
    try:
        from ucimlrepo import fetch_ucirepo
        diabetes = fetch_ucirepo(id=296)  # Diabetes 130-US hospitals
        X = diabetes.data.features
        y = diabetes.data.targets
        df = pd.concat([X, y], axis=1)
        # UCI trae nombres similares; si no existe 'readmitted', intenta detectar
        if 'readmitted' not in df.columns:
            # Normalizar posibles variantes
            possible = [c for c in df.columns if c.lower() == 'readmitted']
            if possible:
                df.rename(columns={possible[0]: 'readmitted'}, inplace=True)
        return df
    except Exception:
        return None

st.set_page_config(page_title="Diabetes 130 - Streamlit", layout="wide")
st.title("📊 Diabetes 130-Hospitales de EE. UU — App Interactiva")

st.markdown(
    """
Esta app carga el dataset **Diabetes 130-US hospitals (1999–2008)**, 
aplica **PCA** a variables numéricas y **MCA** a categóricas (con `prince`), y muestra resultados.
Puedes **subir tu CSV** (`diabetic_data.csv` del notebook) o dejar que la app intente cargarlo desde el repo o UCI.
"""
)

# === Panel lateral ===
st.sidebar.header("Carga de datos")
uploaded = st.sidebar.file_uploader("Sube tu `diabetic_data.csv`", type=["csv"])

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif os.path.exists("diabetic_data.csv"):
    df = pd.read_csv("diabetic_data.csv")
else:
    st.info("No encontré `diabetic_data.csv`. Intentaré cargar desde UCI automáticamente…")
    df = load_uci_if_needed()

if df is None:
    st.error("No fue posible cargar datos. Sube un CSV o añade `diabetic_data.csv` al repo.")
    st.stop()

st.write("**Shape:**", df.shape)

# === Limpieza y mapeos (basado en tu notebook) ===
# Reemplazo de 'None' y '?' por NA
df = df.replace(["None", "?"], pd.NA)

# Convertir ids a string si existen
for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

map_admission_type = {
    "1": "Emergencia", "2": "Urgente", "3": "Electivo", "4": "Recién nacido",
    "5": "Sin_info", "6": "Sin_info", "7": "Centro de trauma", "8": "Sin_info"
}
map_discharge_disposition = {
    "1": "Alta a casa","2": "Alta a otro hospital a corto plazo","3": "Alta a centro de enfermería especializada",
    "4": "Alta a centro de cuidados intermedios","5": "Alta a otro tipo de atención hospitalaria","6": "Cuidados de salud en casa",
    "7": "Salida contra recomendación médica","8": "Alta a casa con cuidados","9": "Admitido como paciente hospitalizado",
    "10": "Cuidados paliativos en casa","11": "Cuidados paliativos en centro médico","12": "Alta a hospital psiquiátrico",
    "13": "Alta a otra instalación de rehabilitación","14": "Sin_info","15": "Sin_info","16": "Alta a hospital federal",
    "17": "Alta a otra institución","18": "Alta a custodia policial","19": "Sin_info","20": "Alta por orden judicial",
    "21": "Sin_info","22": "Falleció en casa","23": "Falleció en instalación médica","24": "Falleció en lugar desconocido",
    "25": "Falleció en cuidados paliativos","28": "Falleció en centro de enfermería especializada"
}
map_admission_source = {
    "1": "Referencia médica","2": "Referencia desde clínica","3": "Referencia desde aseguradora HMO",
    "4": "Transferencia desde hospital","5": "Transferencia desde centro de enfermería especializada",
    "6": "Transferencia desde otro centro de salud","7": "Sala de emergencias","8": "Corte o custodia policial","9": "Sin_info",
    "10": "Transferencia desde hospital de acceso crítico","11": "Parto normal","12": "Sin_info","13": "Nacido en hospital",
    "14": "Nacido fuera de hospital","15": "Sin_info","17": "Sin_info","20": "Sin_info","22": "Sin_info","25": "Sin_info"
}

for col, mp in [
    ("admission_type_id", map_admission_type),
    ("discharge_disposition_id", map_discharge_disposition),
    ("admission_source_id", map_admission_source)
]:
    if col in df.columns:
        df[col] = df[col].map(lambda x: mp.get(str(x), "Desconocido"))

# Rellenar categóricas faltantes con 'Sin_info'
cat_cols_all = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols_all:
    if df[col].isna().any():
        df[col] = df[col].fillna("Sin_info")

# Quitar identificadores si existen
for col in ["encounter_id", "patient_nbr"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Definir columnas según tu notebook (si existen)
num_cols_pca = [c for c in [
    "time_in_hospital","num_lab_procedures","num_procedures","num_medications",
    "number_outpatient","number_emergency","number_inpatient","number_diagnoses"
] if c in df.columns]

cat_cols_mca = [c for c in [
    "race","gender","age","weight","payer_code","medical_specialty","diag_1","diag_2","diag_3",
    "max_glu_serum","A1Cresult","metformin","repaglinide","nateglinide","chlorpropamide",
    "glimepiride","acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
    "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide","examide","citoglipton",
    "insulin","glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone",
    "metformin-rosiglitazone","metformin-pioglitazone","change","diabetesMed",
    "admission_type_id","discharge_disposition_id","admission_source_id"
] if c in df.columns]

st.subheader("Vista previa de datos")
st.dataframe(df.head())

# Target (si existe)
target = "readmitted" if "readmitted" in df.columns else None
if target:
    st.write("Distribución de **readmitted**:")
    st.write(df[target].value_counts())

# ====== PCA ======
st.header("🔹 PCA en variables numéricas")
if len(num_cols_pca) >= 2:
    with st.spinner("Calculando PCA…"):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[num_cols_pca])

        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        explained_var = np.cumsum(pca.explained_variance_ratio_)

    k85 = int(np.argmax(explained_var >= 0.85) + 1)
    st.write(f"Número de componentes para ≥85% varianza explicada: **{k85}**")

    # Curva varianza acumulada
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(range(1, len(explained_var) + 1), explained_var, marker='o', linestyle='--')
    ax1.axhline(y=0.85)
    ax1.set_xlabel('Componentes')
    ax1.set_ylabel('Varianza acumulada')
    ax1.set_title('PCA - Varianza acumulada')
    ax1.grid(True)
    st.pyplot(fig1)

    # Scatter PC1 vs PC2
    fig2, ax2 = plt.subplots(figsize=(6,4))
    if target and target in df.columns:
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df[target].astype(str), alpha=0.6, ax=ax2)
        ax2.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], alpha=0.6, ax=ax2)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('PCA - PC1 vs PC2')
    st.pyplot(fig2)

    # Heatmap loadings (primeras 10 PCs o menos)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=num_cols_pca
    )
    n_show = min(10, loadings.shape[1])
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(loadings.iloc[:, :n_show], annot=True, center=0, ax=ax3)
    ax3.set_title('Loadings (primeras PCs)')
    st.pyplot(fig3)
else:
    st.warning("No hay suficientes variables numéricas para PCA (se requieren ≥ 2).")

# ====== MCA ======
st.header("🔸 MCA en variables categóricas")
if len(cat_cols_mca) >= 2:
    try:
        import prince
        X_cat = df[cat_cols_mca].astype(str)

        with st.spinner("Calculando MCA…"):
            x_mca = prince.MCA(n_components=min(15, len(cat_cols_mca)-1), random_state=42).fit(X_cat)
            X_mca = x_mca.transform(X_cat)
            eigvals = x_mca.eigenvalues_
            var_exp = eigvals / eigvals.sum()
            cum_var_exp = np.cumsum(var_exp)

        d85 = int(np.argmax(cum_var_exp >= 0.85) + 1)
        st.write(f"Dimensiones MCA para ≥85% varianza explicada: **{d85}**")

        # Curva varianza acumulada
        fig4, ax4 = plt.subplots(figsize=(6,4))
        ax4.plot(range(1, len(cum_var_exp)+1), cum_var_exp, marker='o', linestyle='--')
        ax4.axhline(y=0.85)
        ax4.set_xlabel('Dimensiones')
        ax4.set_ylabel('Varianza acumulada')
        ax4.set_title('MCA - Varianza acumulada')
        ax4.grid(True)
        st.pyplot(fig4)

        # Contribución por variable (Dim 1 y 2)
        loadings_cat = x_mca.column_coordinates(X_cat).iloc[:, :2]
        loadings_sq = loadings_cat ** 2
        contrib_cat = loadings_sq.div(loadings_sq.sum(axis=0), axis=1)
        contrib_cat.index = contrib_cat.index.str.split('__').str[0]
        contrib_var = contrib_cat.groupby(contrib_cat.index).sum().sum(axis=1)
        contrib_pct = contrib_var / contrib_var.sum() * 100
        contrib_pct_sorted = contrib_pct.sort_values()

        fig5, ax5 = plt.subplots(figsize=(8,8))
        sns.barplot(x=contrib_pct_sorted.values, y=contrib_pct_sorted.index, ax=ax5)
        ax5.set_xlabel('Contribución (%) a Dim 1 y 2')
        ax5.set_title('MCA - Contribución de variables')
        for i, v in enumerate(contrib_pct_sorted.values):
            ax5.text(v + 0.2, i, f"{v:.2f}%", va='center')
        st.pyplot(fig5)

        # Concatenación de representaciones reducidas (PCA+MCA) opcional
        if len(num_cols_pca) >= 2:
            n_pca = int(np.argmax(np.cumsum(PCA().fit(StandardScaler().fit_transform(df[num_cols_pca])).explained_variance_ratio_) >= 0.85) + 1)
            X_pca_reduced = X_pca[:, :n_pca]
        else:
            X_pca_reduced = np.empty((len(df), 0))

        n_mca = d85
        X_mca_reduced = X_mca.iloc[:, :n_mca].values if hasattr(X_mca, "iloc") else X_mca[:, :n_mca]
        X_reduced = np.hstack((X_pca_reduced, X_mca_reduced))
        st.success(f"Dimensionalidad final concatenada: {X_reduced.shape}")
        st.write("Vista previa (primeras filas):")
        st.dataframe(pd.DataFrame(X_reduced).head())

    except Exception as e:
        st.error(f"No se pudo ejecutar MCA. Revisa que `prince` esté instalado y que las columnas categóricas existan. Detalle: {e}")
else:
    st.warning("No hay suficientes variables categóricas para MCA (se requieren ≥ 2).")

st.caption("© Bioestadística — App de demostración con Streamlit")
