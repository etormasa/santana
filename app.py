# app.py
import os, glob, joblib, numpy as np, pandas as pd
from flask import Flask, render_template, request
from sklearn.utils.multiclass import type_of_target

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# ---- CARGA PEREZOSA (evita crashear al importar en Vercel) ----
_model = None
_cached_meta = None  # (FEATURES, num_cols, cat_cols, problem_type, CSV_PATH, target_col, suggest_options)

def _load_everything_once():
    global _model, _cached_meta
    if _model is not None and _cached_meta is not None:
        return _model, _cached_meta

    # 1) Modelo
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pkl no existe en: {model_path}")
    _model = joblib.load(model_path)

    # 2) CSV
    csv_candidates = glob.glob(os.path.join(MODELS_DIR, "*.csv"))
    if not csv_candidates:
        raise FileNotFoundError("No hay CSV en models/. Coloca tu dataset ahí.")
    CSV_PATH = csv_candidates[0]
    df = pd.read_csv(CSV_PATH)

    # 3) Target y FEATURES
    COMMON_TARGETS = ["target", "Target", "label", "Label", "y", "output", "Outcome", "outcome"]
    target_col = next((c for c in COMMON_TARGETS if c in df.columns), df.columns[-1])
    FEATURES = [c for c in df.columns if c != target_col]

    # 4) Tipo de problema
    y_type = type_of_target(df[target_col].dropna())
    if y_type == "binary":
        problem_type = "binary"
    elif y_type == "multiclass":
        problem_type = "multiclass"
    else:
        problem_type = "regression"

    # 5) Columnas num/cat + opciones sugeridas
    num_cols = df[FEATURES].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in FEATURES if c not in num_cols]
    suggest_options = {}
    for c in cat_cols:
        vals = df[c].dropna().unique()
        if len(vals) <= 8:
            suggest_options[c] = [str(v) for v in sorted(vals, key=lambda x: str(x))]

    _cached_meta = (FEATURES, set(num_cols), set(cat_cols), problem_type, CSV_PATH, target_col, suggest_options)
    return _model, _cached_meta

@app.get("/healthz")
def healthz():
    return "ok", 200

@app.route("/", methods=["GET"])
def index():
    try:
        _, (FEATURES, num_cols, cat_cols, problem_type, CSV_PATH, target_col, suggest_options) = _load_everything_once()
        return render_template(
            "index.html",
            features=FEATURES,
            num_cols=num_cols,
            cat_cols=cat_cols,
            suggest_options=suggest_options,
            problem_type=problem_type,
            csv_name=os.path.basename(CSV_PATH),
            target_name=target_col
        )
    except Exception as e:
        app.logger.exception("Fallo en index()")
        return f"Error al renderizar index: {e}", 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model, (FEATURES, num_cols, cat_cols, problem_type, CSV_PATH, target_col, suggest_options) = _load_everything_once()

        row = []
        for col in FEATURES:
            raw = request.form.get(col, "")
            if col in num_cols:
                if raw == "":
                    row.append(np.nan)
                else:
                    try:
                        row.append(float(raw))
                    except:
                        row.append(np.nan)
            else:
                val = (raw or "").strip()
                row.append(np.nan if val == "" else val)

        import pandas as pd
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
            num_cols=num_cols,
            cat_cols=cat_cols,
            suggest_options=suggest_options,
            problem_type=problem_type,
            csv_name=os.path.basename(CSV_PATH),
            target_name=target_col,
            prediction=str(y_pred),
            probability=proba_txt,
            form_data=request.form
        )
    except Exception as e:
        app.logger.exception("Fallo en /predict")
        return f"Error en predict: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
