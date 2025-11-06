# modelo.py
import os
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.utils.multiclass import type_of_target
import joblib


# =========================
# CONFIG
# =========================
CSV_PATH   = "heart_disease_examen.csv"  # Ajusta si tu CSV está en otra ruta
TARGET_COL = "target"                    # Cambia si tu objetivo se llama distinto
TOP_K      = 10                          # Nº de variables a conservar (tras one-hot)


# =========================
# UTILIDADES
# =========================
def detectar_problema(y: pd.Series) -> str:
    """
    Devuelve 'binary', 'multiclass' o 'regression' según y.
    """
    y_clean = y.dropna()
    t = type_of_target(y_clean)
    # Posibles valores: 'binary', 'multiclass', 'continuous', etc.
    if t == "binary":
        return "binary"
    elif t == "multiclass":
        return "multiclass"
    else:
        return "regression"


def obtener_preprocesador(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )
    return preprocess, numeric_cols, categorical_cols


def nombres_transformados(prep: ColumnTransformer,
                          numeric_cols: list,
                          categorical_cols: list) -> list:
    """
    Devuelve la lista de nombres de features luego del ColumnTransformer.
    num -> mismos nombres, cat -> nombres one-hot.
    """
    out_names = []
    # Numéricas (luego de StandardScaler no cambian nombres)
    out_names += numeric_cols

    # Categóricas (obtener nombres del OneHotEncoder)
    if categorical_cols:
        ohe = prep.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
        out_names += cat_names

    return out_names


def split_con_o_sin_estratificar(X, y, problem_type: str, test_size=0.2, random_state=42):
    """
    Si es clasificación e intentamos estratificar, verifica que cada clase tenga >= 2 individuos.
    Si no se cumple, hace split sin stratify.
    """
    if problem_type in ("binary", "multiclass"):
        vc = y.value_counts()
        if vc.min() < 2:
            print("[AVISO] Alguna clase tiene < 2 muestras. Haré split sin estratificar.")
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
        else:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        # Regresión: no aplica estratificar
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)


# =========================
# CARGA
# =========================
df = pd.read_csv(CSV_PATH)
if TARGET_COL not in df.columns:
    TARGET_COL = df.columns[-1]
    print(f"[AVISO] No encontré '{TARGET_COL}'. Usaré como target la última columna: {TARGET_COL}")

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# Diagnóstico rápido
print("Valores únicos del target (primeros 20 únicos):", sorted(y.dropna().unique())[:20])
print("Tipo de problema detectado (sklearn):", type_of_target(y.dropna()))

problem_type = detectar_problema(y)
print(f"==> Problema detectado: {problem_type.upper()}")


# =========================
# PREPROCESO
# =========================
preprocess, numeric_cols, categorical_cols = obtener_preprocesador(X)

# Selector según tipo de problema
if problem_type in ("binary", "multiclass"):
    from sklearn.feature_selection import mutual_info_classif
    selector = SelectKBest(score_func=mutual_info_classif, k=min(TOP_K, max(1, X.shape[1])))
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model_filename = "model_classification.pkl"
else:
    from sklearn.feature_selection import mutual_info_regression
    selector = SelectKBest(score_func=mutual_info_regression, k=min(TOP_K, max(1, X.shape[1])))
    model = LinearRegression()
    model_filename = "model_regression.pkl"

pipe = Pipeline([
    ("prep", preprocess),
    ("sel", selector),
    ("clf", model),
])

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = split_con_o_sin_estratificar(X, y, problem_type)

# =========================
# ENTRENAR
# =========================
pipe.fit(X_train, y_train)

# =========================
# MÉTRICAS
# =========================
if problem_type in ("binary", "multiclass"):
    y_pred = pipe.predict(X_test)

    # Elegir proba si existe
    y_proba = None
    if hasattr(pipe, "predict_proba"):
        # Para binario -> proba clase positiva (1 si existe)
        if problem_type == "binary":
            classes = pipe.named_steps["clf"].classes_
            pos_idx = list(classes).index(1) if 1 in classes else 1 if len(classes) > 1 else 0
            y_proba = pipe.predict_proba(X_test)[:, pos_idx]
        else:
            # Multiclase: proba completa
            y_proba = pipe.predict_proba(X_test)

    # Promedios adecuados
    avg = "weighted"  # recomendado para multiclase y/o desbalance
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)

    print("\n=== MÉTRICAS (TEST) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f} (average={avg})")
    print(f"Recall   : {rec:.4f} (average={avg})")
    print(f"F1       : {f1:.4f} (average={avg})")

    if problem_type == "binary" and y_proba is not None:
        try:
            roc = roc_auc_score(y_test, y_proba)
            print(f"ROC AUC  : {roc:.4f}")
        except Exception as e:
            print("[AVISO] No se pudo calcular ROC AUC:", e)

    print("\nReporte de clasificación (weighted):")
    print(classification_report(y_test, y_pred, zero_division=0))

else:
    # REGRESIÓN
    y_pred = pipe.predict(X_test)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print("\n=== MÉTRICAS (TEST) ===")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")


# =========================
# TOP FEATURES SELECCIONADAS
# =========================
# nombres después del preprocess (num + onehot)
prep = pipe.named_steps["prep"]
all_names = []
# num
all_names += numeric_cols
# cat one-hot
if categorical_cols:
    ohe = prep.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
else:
    cat_names = []
all_names += cat_names

mask   = pipe.named_steps["sel"].get_support()
scores = pipe.named_steps["sel"].scores_

selected_names  = [name for name, keep in zip(all_names, mask) if keep]
selected_scores = [s for s, keep in zip(scores, mask) if keep]

# Ordenar desc por score
order = np.argsort(selected_scores)[::-1]
selected_sorted = [(selected_names[i], float(selected_scores[i])) for i in order]

print("\n=== TOP FEATURES (por Mutual Information) ===")
for name, sc in selected_sorted:
    print(f"{name:40s}  score={sc:.6f}")

# Guardar lista de features seleccionadas
os.makedirs("models", exist_ok=True)
with open("models/selected_features.txt", "w", encoding="utf-8") as f:
    for name, sc in selected_sorted:
        f.write(f"{name}\t{sc:.6f}\n")

# =========================
# GUARDAR MODELO
# =========================
model_path = os.path.join("models", "model.pkl")  # nombre genérico
joblib.dump(pipe, model_path)
print(f"\n✅ Modelo guardado en: {model_path}")
print("✅ Features seleccionadas guardadas en: models/selected_features.txt")
