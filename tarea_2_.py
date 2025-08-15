# streamlit_app.py
# -------------------------------------------------------------
# App Streamlit para selección de variables (Chi² y ANOVA F)
# con el dataset "Diabetes 130 US hospitals (UCI id=296)".
# Permite:
#  - Elegir método: categóricas (Chi²) o numéricas (ANOVA F)
#  - Ajustar umbral de porcentaje acumulado
#  - Ver ranking de variables y descargarlas
#  - Exportar dataset final con variables seleccionadas + target
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.impute import SimpleImputer

# ============ Config básica de la página ============
st.set_page_config(page_title="Diabetes 130 — Selección de Variables", layout="wide")
st.title("📊 Diabetes 130 — Selección de Variables (Chi² / ANOVA F)")
st.caption("App para filtrar variables por importancia acumulada con Chi² (categóricas) o ANOVA F (numéricas).")

# ================= Utilidades =================
@st.cache_data(show_spinner=True)
def load_dataset():
    """Descarga dataset UCI id=296 con ucimlrepo y arma df completo."""
    try:
        from ucimlrepo import fetch_ucirepo
    except Exception as e:
        raise RuntimeError(
            "No se pudo importar 'ucimlrepo'. Revisa requirements.txt.\n"
            f"Detalle: {e}"
        )
    data = fetch_ucirepo(id=296)
    X = data.data.features.copy()
    y = data.data.targets.copy()
    df = pd.concat([X, y], axis=1)
    return df, X, y

def select_categorical_cols(X: pd.DataFrame):
    return X.select_dtypes(include=["object", "category"]).columns.tolist()

def select_numeric_cols(X: pd.DataFrame):
    return X.select_dtypes(include=[np.number]).columns.tolist()

def chi2_rank(X_train: pd.DataFrame, y_train_enc: np.ndarray, cols: list) -> pd.DataFrame:
    """Calcula ranking Chi² para columnas categóricas codificadas ordinalmente."""
    # Rellenar vacíos (mantener filas)
    X_train_f = X_train.fillna("MISSING")

    # OrdinalEncoder con manejo de categorías desconocidas
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_enc = ord_enc.fit_transform(X_train_f)

    # Desplazar +1 (chi² requiere no-negativos)
    X_train_enc = X_train_enc + 1

    selector_all = SelectKBest(score_func=chi2, k="all")
    selector_all.fit(X_train_enc, y_train_enc)

    scores = np.nan_to_num(selector_all.scores_, nan=0.0)
    rank = (
        pd.DataFrame({"feature": cols, "score": scores})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return rank

def anova_rank(X_train: pd.DataFrame, y_train_enc: np.ndarray, cols: list) -> pd.DataFrame:
    """Calcula ranking ANOVA F para columnas numéricas."""
    selector_all = SelectKBest(score_func=f_classif, k="all")
    selector_all.fit(X_train.values, y_train_enc)
    scores = np.nan_to_num(selector_all.scores_, nan=0.0)
    rank = (
        pd.DataFrame({"feature": cols, "score": scores})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return rank

def cut_by_cum_threshold(rank: pd.DataFrame, cum_threshold: float) -> tuple[list, pd.DataFrame, int]:
    """Devuelve lista de variables hasta alcanzar el porcentaje acumulado solicitado."""
    total = rank["score"].sum()
    if total == 0:
        rank["norm"] = 0.0
        rank["cum_pct"] = 0.0
        cut_idx = 0
    else:
        rank["norm"] = rank["score"] / total
        rank["cum_pct"] = rank["norm"].cumsum()
        cut_idx = (rank["cum_pct"] >= cum_threshold).idxmax()
    selected = rank.loc[:cut_idx, "feature"].tolist()
    return selected, rank, cut_idx

@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ================== Sidebar ==================
with st.sidebar:
    st.header("⚙️ Configuración")

    # Cargar dataset
    df, X, y = load_dataset()

    # Target: por defecto 'readmitted' si existe
    y_cols = y.columns.tolist()
    default_target = "readmitted" if "readmitted" in y_cols else y_cols[0]
    target_col = st.selectbox("Variable objetivo (target)", options=y_cols, index=y_cols.index(default_target))

    # Método
    method = st.radio(
        "Método de selección",
        options=["cat_chi2", "num_f", "ambos"],
        format_func=lambda x: {"cat_chi2": "Categóricas · Chi²", "num_f": "Numéricas · ANOVA F", "ambos": "Unión de ambos"}[x],
    )

    # Umbral
    cum_threshold = st.slider("Umbral de porcentaje acumulado", min_value=0.50, max_value=0.95, value=0.80, step=0.05)

    # Tamaño test
    test_size = st.slider("Proporción de test", 0.2, 0.4, 0.30, 0.05)

# ================== Lógica principal ==================
st.subheader("1) Vista rápida del dataset")
st.write("Filas x Columnas:", df.shape)
st.dataframe(df.head(10), use_container_width=True)

# Separar X/y con el target seleccionado
X_all = df.drop(columns=[target_col])
y_all = df[target_col]

# Split estratificado si y tiene >1 clase
stratify_arg = y_all if y_all.nunique() > 1 else None
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_all, y_all, test_size=test_size, random_state=42, stratify=stratify_arg
)

# Codificar target
lab_y = LabelEncoder()
y_train_enc = lab_y.fit_transform(y_train)
y_test_enc = lab_y.transform(y_test)

# ——— Calcular rankings según método ———
rank_cat = None
rank_num = None
selected_vars_cat = []
selected_vars_num = []

if method in ("cat_chi2", "ambos"):
    cat_cols = select_categorical_cols(X_all)
    if len(cat_cols) == 0:
        st.warning("No hay columnas categóricas para Chi².")
    else:
        X_train_cat = X_train_full[cat_cols].copy()
        rank_cat = chi2_rank(X_train_cat, y_train_enc, cat_cols)
        selected_vars_cat, rank_cat, cut_idx_cat = cut_by_cum_threshold(rank_cat, cum_threshold)
        st.subheader("2) Ranking Categóricas · Chi²")
        st.caption(f"Seleccionadas (umbral {int(cum_threshold*100)}%): **{len(selected_vars_cat)}**")
        st.dataframe(rank_cat, use_container_width=True, height=300)

        # Gráfico chi²
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(rank_cat["score"])), rank_cat["score"])
        ax.set_title("Chi² scores (categóricas)")
        ax.set_ylabel("Chi²")
        ax.set_xlabel("Variables (ordenadas)")
        st.pyplot(fig, clear_figure=True)

        st.download_button(
            "⬇️ Descargar ranking Chi² (CSV)",
            data=to_csv_bytes(rank_cat),
            file_name="ranking_cat_chi2.csv",
            mime="text/csv",
        )

if method in ("num_f", "ambos"):
    num_cols = select_numeric_cols(X_all)
    if len(num_cols) == 0:
        st.warning("No hay columnas numéricas para ANOVA F.")
    else:
        X_train_num = X_train_full[num_cols].copy()
        rank_num = anova_rank(X_train_num, y_train_enc, num_cols)
        selected_vars_num, rank_num, cut_idx_num = cut_by_cum_threshold(rank_num, cum_threshold)
        st.subheader("3) Ranking Numéricas · ANOVA F")
        st.caption(f"Seleccionadas (umbral {int(cum_threshold*100)}%): **{len(selected_vars_num)}**")
        st.dataframe(rank_num, use_container_width=True, height=300)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(range(len(rank_num["score"])), rank_num["score"])
        ax2.set_title("ANOVA F scores (numéricas)")
        ax2.set_ylabel("F-statistic")
        ax2.set_xlabel("Variables (ordenadas)")
        st.pyplot(fig2, clear_figure=True)

        st.download_button(
            "⬇️ Descargar ranking ANOVA F (CSV)",
            data=to_csv_bytes(rank_num),
            file_name="ranking_num_anova.csv",
            mime="text/csv",
        )

# ——— Generar datasets de salida ———
st.subheader("4) Datasets seleccionados")

dfs_to_offer = []
if rank_cat is not None and len(selected_vars_cat) > 0:
    df_out_cat = pd.concat([X_all[selected_vars_cat], y_all.rename(target_col)], axis=1)
    st.markdown(f"**Categóricas (Chi²)** — columnas: {len(selected_vars_cat)} · forma: {df_out_cat.shape}")
    st.dataframe(df_out_cat.head(10), use_container_width=True)
    dfs_to_offer.append(("dataset_cat_chi2.csv", df_out_cat))

if rank_num is not None and len(selected_vars_num) > 0:
    df_out_num = pd.concat([X_all[selected_vars_num], y_all.rename(target_col)], axis=1)
    st.markdown(f"**Numéricas (ANOVA F)** — columnas: {len(selected_vars_num)} · forma: {df_out_num.shape}")
    st.dataframe(df_out_num.head(10), use_container_width=True)
    dfs_to_offer.append(("dataset_num_anova.csv", df_out_num))

if method == "ambos":
    final_vars = sorted(set(selected_vars_cat) | set(selected_vars_num))
    if len(final_vars) > 0:
        df_final = pd.concat([X_all[final_vars], y_all.rename(target_col)], axis=1)
        st.markdown(f"**Unión (ambos)** — columnas: {len(final_vars)} · forma: {df_final.shape}")
        st.dataframe(df_final.head(10), use_container_width=True)
        dfs_to_offer.append(("dataset_union_ambos.csv", df_final))
    else:
        st.info("No hubo variables seleccionadas en alguno de los métodos.")

# Botones de descarga
if len(dfs_to_offer) > 0:
    st.subheader("5) Descargas")
    cols_dl = st.columns(min(3, len(dfs_to_offer)))
    for i, (fname, dframe) in enumerate(dfs_to_offer):
        with cols_dl[i % len(cols_dl)]:
            st.download_button(
                f"⬇️ {fname}",
                data=to_csv_bytes(dframe),
                file_name=fname,
                mime="text/csv",
            )

st.divider()
st.caption("Tip: si vas a hacer modelado con las categóricas, recuerda volver a codificar (One-Hot o similar) de forma consistente entre train/test.")
