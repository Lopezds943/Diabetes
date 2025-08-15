# streamlit_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ========= Configuración de página =========
st.set_page_config(page_title="Diabetes 130 — PCA & MCA", layout="wide")
st.title("📊 Diabetes 130-US Hospitals — PCA & MCA (Streamlit)")
st.caption("App interactiva con validaciones, carga de datos flexible y manejo de errores.")

# ========= Utilidades =========
@st.cache_data(show_spinner=False)
def load_local_csv(path: str):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=True)
def load_from_uci():
    """Carga dataset desde UCI (id=296) si no hay CSV local ni subida."""
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=296)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        # Normaliza nombre de la columna objetivo si llega con variantes
        for c in df.columns:
            if c.lower() == "readmitted":
                if c != "readmitted":
                    df = df.rename(columns={c: "readmitted"})
                break
        return df
    except Exception:
        return None

def map_code_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica mapeos de códigos a texto en 3 columnas, si existen."""
    # Asegura tipo string
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

    if "admission_type_id" in df.columns:
        df["admission_type_id"] = df["admission_type_id"].map(lambda x: map_admission_type.get(str(x), "Desconocido"))
    if "discharge_disposition_id" in df.columns:
        df["discharge_disposition_id"] = df["discharge_disposition_id"].map(lambda x: map_discharge_disposition.get(str(x), "Desconocido"))
    if "admission_source_id" in df.columns:
        df["admission_source_id"] = df["admission_source_id"].map(lambda x: map_admission_source.get(str(x), "Desconocido"))

    return df

def numeric_cols_present(df):
    return [c for c in [
        "time_in_hospital","num_lab_procedures","num_procedures","num_medications",
        "number_outpatient","number_emergency","number_inpatient","number_diagnoses"
    ] if c in df.columns]

def categorical_cols_present(df):
    return [c for c in [
        "race","gender","age","weight","payer_code","medical_specialty","diag_1","diag_2","diag_3",
        "max_glu_serum","A1Cresult","metformin","repaglinide","nateglinide","chlorpropamide",
        "glimepiride","acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
        "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide","examide","citoglipton",
        "insulin","glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone",
        "metformin-rosiglitazone","metformin-pioglitazone","change","diabetesMed",
        "admission_type_id","discharge_disposition_id","admission_source_id"
    ] if c in df.columns]

# ========= Sidebar: carga y opciones =========
st.sidebar.header("⚙️ Opciones")
uploaded = st.sidebar.file_uploader("Sube tu `diabetic_data.csv`", type=["csv"])

sample_on = st.sidebar.checkbox("Usar muestra para visualizar más rápido", value=True)
sample_size = st.sidebar.slider("Tamaño de muestra", min_value=200, max_value=20000, value=3000, step=200)

variance_threshold = st.sidebar.slider("Umbral de varianza acumulada (%)", 70, 95, 85, 1) / 100.0

# ========= Carga de datos =========
df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = load_local_csv("diabetic_data.csv")
    if df is None:
        df = load_from_uci()

if df is None or df.empty:
    st.error("❌ No se pudieron cargar datos. Sube un CSV o permite la carga desde UCI.")
    st.stop()

# Limpieza básica y mapeos
df = df.replace(["None", "?"], pd.NA)
df = map_code_columns(df)

# Rellena categóricas faltantes
cat_all = df.select_dtypes(include="object").columns.tolist()
for c in cat_all:
    if df[c].isna().any():
        df[c] = df[c].fillna("Sin_info")

# Quita identificadores si existen
for c in ["encounter_id", "patient_nbr"]:
    if c in df.columns:
        df = df.drop(columns=[c])

# Muestra previa + muestreo opcional
st.subheader("👀 Vista rápida de datos")
st.write("Shape:", df.shape)
st.dataframe(df.head())

if sample_on and len(df) > sample_size:
    df_view = df.sample(sample_size, random_state=42)
    st.caption(f"Mostrando muestra aleatoria de {len(df_view)} filas (de {len(df)}).")
else:
    df_view = df.copy()

# Target (si existe)
target = "readmitted" if "readmitted" in df_view.columns else None
if target:
    st.write("Distribución de **readmitted**:")
    st.write(df_view[target].value_counts(dropna=False))

# ========= Determina columnas válidas =========
num_cols_pca = numeric_cols_present(df_view)
cat_cols_mca = categorical_cols_present(df_view)

with st.expander("🔎 Columnas detectadas"):
    st.write("Numéricas (PCA):", num_cols_pca)
    st.write("Categóricas (MCA):", cat_cols_mca)

if len(num_cols_pca) < 2 and len(cat_cols_mca) < 2:
    st.error("❌ No hay suficientes columnas válidas para PCA/MCA. Revisa los nombres de columnas del dataset.")
    st.stop()

# ========= Sección PCA =========
st.header("🔹 PCA (variables numéricas)")
if len(num_cols_pca) >= 2:
    try:
        # Imputación + escalado
        X_num = df_view[num_cols_pca].copy()
        imputer = SimpleImputer(strategy="median")
        X_num_imp = imputer.fit_transform(X_num)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num_imp)

        # PCA completo para curva de varianza
        pca_full = PCA()
        X_pca_full = pca_full.fit_transform(X_scaled)
        explained_cum = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.argmax(explained_cum >= variance_threshold) + 1)

        st.write(f"Componentes para ≥{int(variance_threshold*100)}% varianza explicada: **{k}**")

        # Gráfico de varianza acumulada
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(range(1, len(explained_cum)+1), explained_cum, marker="o", linestyle="--")
        ax1.axhline(y=variance_threshold)
        ax1.set_xlabel("Componentes")
        ax1.set_ylabel("Varianza acumulada")
        ax1.set_title("PCA - Varianza acumulada")
        ax1.grid(True)
        st.pyplot(fig1, clear_figure=True)

        # Scatter PC1 vs PC2
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        if target:
            # Colorea por target (como texto para evitar problemas)
            ax2.scatter(X_pca_full[:, 0], X_pca_full[:, 1], c=pd.factorize(df_view[target].astype(str))[0], alpha=0.6)
            ax2.set_title("PCA - PC1 vs PC2 (coloreado por 'readmitted')")
        else:
            ax2.scatter(X_pca_full[:, 0], X_pca_full[:, 1], alpha=0.6)
            ax2.set_title("PCA - PC1 vs PC2")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        st.pyplot(fig2, clear_figure=True)

        # Heatmap de loadings (primeras PCs)
        loadings = pd.DataFrame(
            pca_full.components_.T,
            columns=[f"PC{i+1}" for i in range(pca_full.n_components_)],
            index=num_cols_pca
        )
        n_show = min(10, loadings.shape[1])
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        im = ax3.imshow(loadings.iloc[:, :n_show], aspect="auto", cmap="coolwarm", vmin=-np.max(np.abs(loadings.values)), vmax=np.max(np.abs(loadings.values)))
        ax3.set_xticks(range(n_show))
        ax3.set_xticklabels([f"PC{i+1}" for i in range(n_show)], rotation=45, ha="right")
        ax3.set_yticks(range(len(num_cols_pca)))
        ax3.set_yticklabels(num_cols_pca)
        ax3.set_title("PCA - Loadings (primeras PCs)")
        fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        st.pyplot(fig3, clear_figure=True)

        # PCA reducido a k componentes
        pca_k = PCA(n_components=k)
        X_pca_k = pca_k.fit_transform(X_scaled)

    except Exception as e:
        st.error("⚠️ Falló la sección de PCA.")
        st.exception(e)
        st.stop()
else:
    st.warning("No hay suficientes columnas numéricas para PCA (se requieren ≥ 2).")
    X_pca_k = np.empty((len(df_view), 0))
    k = 0

# ========= Sección MCA =========
st.header("🔸 MCA (variables categóricas)")
if len(cat_cols_mca) >= 2:
    try:
        import prince  # import aquí para mostrar error claro si falta

        X_cat = df_view[cat_cols_mca].astype(str)

        # Ajuste de MCA
        max_components = max(2, min(15, len(cat_cols_mca) - 1))
        mca_model = prince.MCA(n_components=max_components, random_state=42)
        mca_model = mca_model.fit(X_cat)
        X_mca_all = mca_model.transform(X_cat)

        # Varianza acumulada
        eig = mca_model.eigenvalues_
        eig_vals = np.array(eig.values if hasattr(eig, "values") else eig)
        var_exp = eig_vals / eig_vals.sum()
        cum_var = np.cumsum(var_exp)
        d = int(np.argmax(cum_var >= variance_threshold) + 1)

        st.write(f"Dimensiones para ≥{int(variance_threshold*100)}% varianza explicada: **{d}**")

        # Curva varianza acumulada
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.plot(range(1, len(cum_var)+1), cum_var, marker="o", linestyle="--")
        ax4.axhline(y=variance_threshold)
        ax4.set_xlabel("Dimensiones")
        ax4.set_ylabel("Varianza acumulada")
        ax4.set_title("MCA - Varianza acumulada")
        ax4.grid(True)
        st.pyplot(fig4, clear_figure=True)

        # Contribución por variable (Dim 1 y 2)
        coords = mca_model.column_coordinates(X_cat).iloc[:, :2]
        coords_sq = coords ** 2
        contrib = coords_sq.div(coords_sq.sum(axis=0), axis=1)
        contrib.index = contrib.index.str.split("__").str[0]
        contrib_var = contrib.groupby(contrib.index).sum().sum(axis=1)
        contrib_pct = (contrib_var / contrib_var.sum() * 100).sort_values()

        fig5, ax5 = plt.subplots(figsize=(10, 8))
        ax5.barh(contrib_pct.index, contrib_pct.values)
        ax5.set_xlabel("Contribución (%) a Dim 1 y 2")
        ax5.set_title("MCA — Contribución por variable")
        for i, v in enumerate(contrib_pct.values):
            ax5.text(v + 0.2, i, f"{v:.2f}%")
        st.pyplot(fig5, clear_figure=True)

        # Reduce a d dimensiones
        # X_mca_all puede ser DataFrame o ndarray
        X_mca_used = X_mca_all.iloc[:, :d].values if hasattr(X_mca_all, "iloc") else X_mca_all[:, :d]

    except Exception as e:
        st.error("⚠️ Falló la sección de MCA.")
        st.exception(e)
        st.stop()
else:
    st.warning("No hay suficientes columnas categóricas para MCA (se requieren ≥ 2).")
    X_mca_used = np.empty((len(df_view), 0))
    d = 0

# ========= Concatenación final opcional =========
st.header("🧱 Representación reducida (PCA + MCA)")
try:
    X_reduced = np.hstack((X_pca_k, X_mca_used))
    st.success(f"Dimensionalidad final: {X_reduced.shape}  (PCA={k}, MCA={d})")
    preview_cols = [f"PCA_{i+1}" for i in range(X_pca_k.shape[1])] + [f"MCA_{i+1}" for i in range(X_mca_used.shape[1])]
    preview_df = pd.DataFrame(X_reduced, columns=preview_cols)
    st.dataframe(preview_df.head())
except Exception as e:
    st.error("No fue posible concatenar las representaciones reducidas.")
    st.exception(e)

st.caption("© Bioestadística — Demo PCA/MCA con Streamlit")
