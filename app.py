import os
import glob
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.utils.multiclass import type_of_target

# === Rutas base ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")

# === Cargar modelo ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# === Descubrir CSV para obtener columnas originales ===
csv_candidates = glob.glob(os.path.join(MODELS_DIR, "*.csv"))
if not csv_candidates:
    raise FileNotFoundError("No se encontró ningún CSV dentro de 'models/'. Coloca tu dataset ahí.")
CSV_PATH = csv_candidates[0]  # toma el primero que encuentre
df = pd.read_csv(CSV_PATH)

# === Inferir target (igual lógica que en tu entrenamiento) ===
COMMON_TARGETS = ["target", "Target", "label", "Label", "y", "output", "Outcome", "outcome"]
target_col = None
for c in COMMON_TARGETS:
    if c in df.columns:
        target_col = c
        break
if target_col is None:
    target_col = df.columns[-1]  # última columna como fallback

# === Columnas de entrada (en el mismo orden original) ===
FEATURES = [c for c in df.columns if c != target_col]

# Detectar tipo de problema para la presentación del resultado
y_type = type_of_target(df[target_col].dropna())
if y_type == "binary":
    problem_type = "binary"
elif y_type == "multiclass":
    problem_type = "multiclass"
else:
    problem_type = "regression"

# Para el formulario: sugerir si una columna podría ser numérica o categórica
num_cols = df[FEATURES].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in FEATURES if c not in num_cols]

# Para opciones prellenadas (si una cat tiene pocos valores únicos)
suggest_options = {}
for c in cat_cols:
    vals = df[c].dropna().unique()
    if len(vals) <= 8:
        # guarda como strings para que el form no truene
        suggest_options[c] = [str(v) for v in sorted(vals, key=lambda x: str(x))]

# IMPORTANTE: indicar carpeta de templates explícitamente
app = Flask(__name__, template_folder=TEMPLATES_DIR)

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        features=FEATURES,
        num_cols=set(num_cols),
        cat_cols=set(cat_cols),
        suggest_options=suggest_options,
        problem_type=problem_type,
        csv_name=os.path.basename(CSV_PATH),
        target_name=target_col
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = []
        for col in FEATURES:
            raw = request.form.get(col, "")

            # numéricos: float o NaN
            if col in num_cols:
                if raw == "":
                    row.append(np.nan)
                else:
                    try:
                        row.append(float(raw))
                    except:
                        row.append(np.nan)
            else:
                # categóricos: convierte vacío a NaN para que el Imputer actúe
                val = raw.strip()
                if val == "":
                    row.append(np.nan)
                else:
                    row.append(val)

        # >>> clave: DataFrame con mismas columnas que en el entrenamiento
        X_df = pd.DataFrame([row], columns=FEATURES)

        y_pred = model.predict(X_df)[0]

        proba_txt = None
        if hasattr(model, "predict_proba") and problem_type in ("binary", "multiclass"):
            proba = model.predict_proba(X_df)[0]
            try:
                classes = model.named_steps["clf"].classes_
                if problem_type == "binary" and 1 in classes:
                    idx_pos = list(classes).index(1)
                    proba_txt = f"{proba[idx_pos]:.4f}"
                else:
                    proba_txt = ", ".join([f"{p:.3f}" for p in proba])
            except Exception:
                proba_txt = ", ".join([f"{p:.3f}" for p in proba])

        return render_template(
            "index.html",
            features=FEATURES,
            num_cols=set(num_cols),
            cat_cols=set(cat_cols),
            suggest_options=suggest_options,
            problem_type=problem_type,
            csv_name=os.path.basename(CSV_PATH),
            target_name=target_col,
            prediction=str(y_pred),
            probability=proba_txt,
            form_data=request.form
        )
    except Exception as e:
        return render_template(
            "index.html",
            features=FEATURES,
            num_cols=set(num_cols),
            cat_cols=set(cat_cols),
            suggest_options=suggest_options,
            problem_type=problem_type,
            csv_name=os.path.basename(CSV_PATH),
            target_name=target_col,
            error=str(e),
            form_data=request.form
        )

if __name__ == "__main__":
    # Ejecutar: python app.py  → http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
