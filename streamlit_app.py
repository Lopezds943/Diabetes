# streamlit_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt

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
    """Lee un CSV local si existe; si falla devuelve None."""
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
    """Mapea IDs de admisión/alta/fuente a etiquetas en español si las columnas existen."""
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
sample_size = st.sidebar.slider("Tamaño de muestra", 200, 20000, 3000, 200)
variance_threshold = st.sidebar.slider("Umbral de varianza acumulada (%)", 70, 95, 85, 1) / 100.0

# ========= Carga de datos (robusta) =========
df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ No pude leer el CSV subido: {e}")
        st.stop()
else:
    local_df = load_local_csv("diabetic_data.csv")
    if local_df is not None:
        df = local_df
    else:
        df = load_from_uci()

if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("❌ No se pudieron cargar datos. Sube un CSV o permite la carga desde UCI.")
    st.stop()

# ========= Limpieza básica =========
df = df.replace(["None", "?"], pd.NA)
df = map_code_columns(df)

for c in df.select_dtypes(include="object").columns:
    if df[c].isna().any():
        df[c] = df[c].fillna("Sin_info")

for c in ["encounter_id", "patient_nbr"]:
    if c in df.columns:
        df = df.drop(columns=[c])

# ========= Preview =========
st.subheader("👀 Vista rápida de datos")
st.write("Shape:", df.shape)
st.dataframe(df.head())

df_view = df.sample(sample_size, random_state=42) if (sample_on and len(df) > sample_size) else df.copy()
if sample_on and len(df) > sample_size:
    st.caption(f"Mostrando muestra aleatoria de {len(df_view)} filas (de {len(df)}).")

target = "readmitted" if "readmitted" in df_view.columns else None
if target:
    st.write("Distribución de **readmitted**:")
    st.write(df_view[target].value_counts(dropna=False))

# ========= Columnas válidas =========
num_cols_pca = numeric_cols_present(df_view)
cat_cols_mca = categorical_cols_present(df_view)

with st.expander("🔎 Columnas detectadas"):
    st.write("Numéricas (PCA):", num_cols_pca)
    st.write("Categóricas (MCA):", cat_cols_mca)

if len(num_cols_pca) < 2 and len(cat_cols_mca) < 2:
    st.error("❌ No hay suficientes columnas válidas para PCA/MCA. Revisa los nombres de columnas del dataset.")
    st.stop()

# ========= PCA =========
st.header("🔹 PCA (variables numéricas)")
if len(num_cols_pca) >= 2:
    try:
        X_num = df_view[num_cols_pca].copy()
        imputer = SimpleImputer(strategy="median")
        X_num_imp = imputer.fit_transform(X_num)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num_imp)

        pca_full = PCA()
        X_pca_full = pca_full.fit_transform(X_scaled)
        explained_cum = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.argmax(explained_cum >= variance_threshold) + 1)

        st.write(f"Componentes para ≥{int(variance_threshold*100)}% varianza explicada: **{k}**")

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(range(1, len(explained_cum)+1), explained_cum, marker="o", linestyle="--")
        ax1.axhline(y=variance_threshold); ax1.grid(True)
        ax1.set_xlabel("Componentes"); ax1.set_ylabel("Varianza acumulada"); ax1.set_title("PCA - Varianza acumulada")
        st.pyplot(fig1, clear_figure=True)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        if target:
            ax2.scatter(X_pca_full[:, 0], X_pca_full[:, 1], c=pd.factorize(df_view[target].astype(str))[0], alpha=0.6)
            ax2.set_title("PCA - PC1 vs PC2 (coloreado por 'readmitted')")
        else:
            ax2.scatter(X_pca_full[:, 0], X_pca_full[:, 1], alpha=0.6)
            ax2.set_title("PCA - PC1 vs PC2")
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
        st.pyplot(fig2, clear_figure=True)

        # Loadings (primeras PCs)
        loadings = pd.DataFrame(
            pca_full.components_.T,
            columns=[f"PC{i+1}" for i in range(pca_full.n_components_)],
            index=num_cols_pca
        )
        n_show = min(10, loadings.shape[1])
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        im = ax3.imshow(
            loadings.iloc[:, :n_show], aspect="auto",
            cmap="coolwarm",
            vmin=-np.max(np.abs(loadings.values)), vmax=np.max(np.abs(loadings.values))
        )
        ax3.set_xticks(range(n_show)); ax3.set_xticklabels([f"PC{i+1}" for i in range(n_show)], rotation=45, ha="right")
        ax3.set_yticks(range(len(num_cols_pca))); ax3.set_yticklabels(num_cols_pca)
        ax3.set_title("PCA - Loadings (primeras PCs)")
        fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        st.pyplot(fig3, clear_figure=True)

        pca_k = PCA(n_components=k)
        X_pca_k = pca_k.fit_transform(X_scaled)

    except Exception as e:
        st.error("⚠️ Falló la sección de PCA.")
        st.exception(e)
        X_pca_k = np.empty((len(df_view), 0))
        k = 0
else:
    st.warning("No hay suficientes columnas numéricas para PCA (se requieren ≥ 2).")
    X_pca_k = np.empty((len(df_view), 0))
    k = 0

# ========= MCA (mejorada y robusta) =========
st.header("🔸 MCA (variables categóricas)")
if len(cat_cols_mca) >= 2:
    try:
        X_cat = df_view[cat_cols_mca].astype(str)

        max_components = max(2, min(15, len(cat_cols_mca) - 1))
        import prince
        mca_model = prince.MCA(n_components=max_components, random_state=42).fit(X_cat)
        X_mca_all = mca_model.transform(X_cat)

        eig = mca_model.eigenvalues_
        eig_vals = np.array(eig.values if hasattr(eig, "values") else eig)
        var_exp = eig_vals / eig_vals.sum()
        cum_var = np.cumsum(var_exp)
        d = int(np.argmax(cum_var >= variance_threshold) + 1)
        st.write(f"Dimensiones para ≥{int(variance_threshold*100)}% varianza explicada: **{d}**")

        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.plot(range(1, len(cum_var)+1), cum_var, marker="o", linestyle="--")
        ax4.axhline(y=variance_threshold); ax4.grid(True)
        ax4.set_xlabel("Dimensiones"); ax4.set_ylabel("Varianza acumulada"); ax4.set_title("MCA - Varianza acumulada")
        st.pyplot(fig4, clear_figure=True)

        # ---------- Contribuciones (parser robusto de etiquetas) ----------
        coords = mca_model.column_coordinates(X_cat).iloc[:, :2]
        coords_sq = coords ** 2
        contrib = coords_sq.div(coords_sq.sum(axis=0), axis=1)  # contrib por categoría a Dim1 y Dim2

        def split_label(label: str, known_vars: list[str]):
            label = str(label)
            for sep in ["__", "=", ":", "|"]:
                if sep in label:
                    a, b = label.split(sep, 1)
                    return a, b
            if "_" in label:
                pref, suf = label.split("_", 1)
                if pref in known_vars:
                    return pref, suf
            return label, ""

        pct_by_label = (contrib.sum(axis=1) / contrib.sum(axis=1).sum() * 100)
        labels = pct_by_label.index.tolist()

        vars_parsed, cats_parsed = [], []
        for lab in labels:
            v, c = split_label(lab, cat_cols_mca)
            vars_parsed.append(v); cats_parsed.append(c)

        # DataFrames para graficar
        df_cat = (
            pd.DataFrame({"variable": vars_parsed, "categoria": cats_parsed, "pct": pct_by_label.values})
            .sort_values("pct", ascending=False)
            .reset_index(drop=True)
        )
        df_var = (
            df_cat.groupby("variable", as_index=False)["pct"].sum()
            .sort_values("pct", ascending=False)
            .reset_index(drop=True)
        )

        st.subheader("📈 Contribuciones a Dim 1 y 2")
        mode = st.radio("Vista", ["Variables (agrupado)", "Categorías (niveles)"], horizontal=True)
        max_items = 60
        max_count = max(1, min(max_items, max(len(df_var), len(df_cat))))
        top_n = st.slider("Top N a mostrar", 1, max_count, min(15, max_count))

        if mode == "Variables (agrupado)":
            df_plot = df_var.copy()
            top = df_plot.head(top_n)
            if len(df_plot) > top_n:
                otros = pd.DataFrame([{"variable": "Otros", "pct": df_plot["pct"].iloc[top_n:].sum()}])
                df_show = pd.concat([top, otros], ignore_index=True)
            else:
                df_show = top
            height = int(22 * len(df_show) + 120)

            chart = (
                alt.Chart(df_show)
                .mark_bar()
                .encode(
                    x=alt.X("pct:Q", title="Contribución (%) a Dim 1 y 2"),
                    y=alt.Y("variable:N", sort="-x", title=None),
                    tooltip=[alt.Tooltip("variable:N", title="Variable"),
                             alt.Tooltip("pct:Q", format=".2f", title="Contribución %")]
                )
                .properties(height=height, title="MCA — Contribución por variable (Top N + Otros)")
            )
            st.altair_chart(chart, use_container_width=True)

        else:
            df_plot = df_cat.copy().head(top_n)
            height = int(22 * len(df_plot) + 140)

            chart = (
                alt.Chart(df_plot)
                .mark_bar()
                .encode(
                    x=alt.X("pct:Q", title="Contribución (%) a Dim 1 y 2"),
                    y=alt.Y("categoria:N", sort="-x", title=None),
                    color=alt.Color("variable:N", legend=alt.Legend(title="Variable", orient="right")),
                    tooltip=[alt.Tooltip("variable:N", title="Variable"),
                             alt.Tooltip("categoria:N", title="Categoría"),
                             alt.Tooltip("pct:Q", format=".2f", title="Contribución %")]
                )
                .properties(height=height, title="MCA — Contribución por categoría (Top N)")
            )
            st.altair_chart(chart, use_container_width=True)

        # Reduce a d dimensiones para la concatenación final
        X_mca_used = X_mca_all.iloc[:, :d].values if hasattr(X_mca_all, "iloc") else X_mca_all[:, :d]

    except Exception as e:
        st.error("⚠️ Falló la sección de MCA.")
        st.exception(e)
        X_mca_used = np.empty((len(df_view), 0))
        d = 0
else:
    st.warning("No hay suficientes columnas categóricas para MCA (se requieren ≥ 2).")
    X_mca_used = np.empty((len(df_view), 0))
    d = 0

# ========= Concatenación final =========
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
