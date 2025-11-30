# backend/predict_routes.py

from fastapi import APIRouter, UploadFile, File, Depends
import joblib, numpy as np, json, os, uuid
import tensorflow as tf
from sqlalchemy.orm import Session

from .gradcam import generate_gradcam
from .schemas import TabularInput
from .database import SessionLocal, History
from .auth import get_current_user
from .rule_engine import rule_engine
from .comparison_utils import build_comparisons
from .report_generator import generate_pdf
from backend.gpt2_generator import generate_advice_with_gpt2, fallback_template
from .brain_loader import load_brain_model
  # ✅ brain CNN loader

predict_router = APIRouter()

# === Load models ===
heart_model = joblib.load("backend/models/heart_optimal.pkl")
diabetes_model = joblib.load("backend/models/diabetes_optimal.pkl")
breast_model = joblib.load("backend/models/breast_optimal.pkl")

# ✅ Brain CNN model (MobileNetV3-based) + class map
brain_model, class_map = load_brain_model()

# Diabetes feature order
DIABETES_FEATURE_ORDER = [
    "Sex",
    "HighBP",
    "HighChol",
    "Smoker",
    "PhysActivity",
    "GenHlth",
    "MentHlth",
    "BMI",
    "Age",
]


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ================= HEART SPECIAL FIX =================
def prepare_heart_input(model_dict, data: dict):
    """
    Build input vector in the exact same order the model was trained on.
    Missing values become None → imputer fills median.
    """
    feature_names = model_dict["feature_names"]  # 13 features
    row = []
    for col in feature_names:
        row.append(data.get(col, None))
    return np.array([row], dtype=float)


# ================= GENERIC TABULAR =================
def run_tabular(model, data_dict, feature_order=None):
    imputer = model["imputer"]
    scaler = model["scaler"]
    clf = model["model"]

    if feature_order is None:
        X = np.array([list(data_dict.values())], dtype=float)
    else:
        X = np.array([[data_dict[f] for f in feature_order]], dtype=float)

    X = imputer.transform(X)
    X = scaler.transform(X)

    prob = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(prob))
    max_prob = float(max(prob))
    return pred_idx, max_prob, prob.tolist()


def run_breast_tabular(model, data_dict):
    imputer = model["imputer"]
    scaler = model["scaler"]
    clf = model["model"]
    feature_names = model.get("feature_names")

    if not feature_names:
        return run_tabular(model, data_dict)

    row = []
    for f in feature_names:
        if f in data_dict and data_dict[f] not in ("", None):
            row.append(float(data_dict[f]))
        else:
            row.append(np.nan)

    X = np.array([row], dtype=float)
    X = imputer.transform(X)
    X = scaler.transform(X)

    prob = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(prob))
    max_prob = float(max(prob))
    return pred_idx, max_prob, prob.tolist()


def make_report_html(disease, label, probability, comparisons, suggestion_text):
    prob_pct = round(probability * 100, 2)
    rows = ""
    for c in comparisons:
        rows += f"""
        <tr>
          <td>{c['feature']}</td>
          <td>{c['user_value']}</td>
          <td>{c.get('normal_min','')}</td>
          <td>{c.get('normal_max','')}</td>
          <td>{c.get('status','')}</td>
        </tr>
        """

    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        h1 {{ color: #3b82f6; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 12px; }}
        th {{ background: #f3f4f6; text-align: left; }}
      </style>
    </head>
    <body>
      <h1>{disease.capitalize()} Prediction Report</h1>
      <p><strong>Prediction:</strong> {label}</p>
      <p><strong>Probability:</strong> {prob_pct}%</p>

      <h2>Feature Comparisons</h2>
      <table>
        <tr>
          <th>Feature</th>
          <th>User Value</th>
          <th>Normal Min</th>
          <th>Normal Max</th>
          <th>Status</th>
        </tr>
        {rows}
      </table>

      <h2>Advisory</h2>
      <p>{suggestion_text}</p>
    </body>
    </html>
    """


# ================= HEART (FIXED) =================
@predict_router.post("/heart")
def predict_heart(
    payload: TabularInput,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    data = payload.data

    # 13-feature order from pickle
    X = prepare_heart_input(heart_model, data)
    X = heart_model["imputer"].transform(X)
    X = heart_model["scaler"].transform(X)
    probs = heart_model["model"].predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    prob = float(max(probs))

    label = "High Risk" if pred_idx == 1 else "Low Risk"

    comparisons, chart_data = build_comparisons("heart", data)
    rule_out = rule_engine("heart", label, prob, data)
    base_advice = rule_out["base_advice"]

    override = rule_out.get("override_label")
    if override:
        label = override

    context = (
        f"Heart disease prediction: {label} with probability {prob:.2f}. "
        f"Key features: {data}. Clinical-style base advice: {base_advice}. "
        "Write 3–5 short, patient-friendly counselling sentences about heart health. "
        "Do not repeat any sentence."
    )
    suggestion_text = generate_advice_with_gpt2(context, "heart", label, prob)


    report_id = uuid.uuid4().hex
    pdf_path = generate_pdf(
        report_id,
        {
            "disease": "heart",
            "label": label,
            "probability": prob,
            "comparisons": comparisons,
            "advisory": suggestion_text,
            "gradcam_path": "",  # none for heart
        },
    )
    report_url = "/reports/" + os.path.basename(pdf_path)

    rec = History(
        user_id=current_user.id,
        disease="heart",
        input_data=json.dumps(data),
        prediction=label,
        probability=str(prob),
        report_path=report_url,
        gradcam_path="",
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    return {
        "id": rec.id,
        "disease": "heart",
        "label": label,
        "probability": prob,
        "raw_probs": probs.tolist(),
        "comparisons": comparisons,
        "chart_data": chart_data,
        "suggestion_text": suggestion_text,
        "lifestyle_recommendations": rule_out["lifestyle_recommendations"],
        "urgency": rule_out["urgency"],
        "suggestion_source": "rule+gpt2",
        "report_url": report_url,
    }


# ================= DIABETES =================
@predict_router.post("/diabetes")
def predict_diabetes(
    payload: TabularInput,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    data = payload.data
    pred_idx, prob, probs = run_tabular(
        diabetes_model, data, feature_order=DIABETES_FEATURE_ORDER
    )

    label_map = {0: "Normal", 1: "Prediabetes", 2: "Diabetes"}
    label = label_map.get(pred_idx, f"Class {pred_idx}")

    comparisons, chart_data = build_comparisons("diabetes", data)
    rule_out = rule_engine("diabetes", label, prob, data)
    base_advice = rule_out["base_advice"]

    context = (
        f"Diabetes status prediction: {label} with probability {prob:.2f}. "
        f"Key features: {data}. Clinical-style base advice: {base_advice}. "
        "Write 3–5 simple sentences explaining what this risk means and general lifestyle advice. "
        "Do not repeat sentences."
    )
    suggestion_text = generate_advice_with_gpt2(
    context,
    "diabetes",
    label,
    prob
)


    report_id = uuid.uuid4().hex
    pdf_path = generate_pdf(
        report_id,
        {
            "disease": "diabetes",
            "label": label,
            "probability": prob,
            "comparisons": comparisons,
            "advisory": suggestion_text,
            "gradcam_path": "",
        },
    )
    report_url = "/reports/" + os.path.basename(pdf_path)

    rec = History(
        user_id=current_user.id,
        disease="diabetes",
        input_data=json.dumps(data),
        prediction=label,
        probability=str(prob),
        report_path=report_url,
        gradcam_path="",
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    return {
        "id": rec.id,
        "disease": "diabetes",
        "label": label,
        "probability": prob,
        "raw_probs": probs,
        "comparisons": comparisons,
        "chart_data": chart_data,
        "suggestion_text": suggestion_text,
        "lifestyle_recommendations": rule_out["lifestyle_recommendations"],
        "urgency": rule_out["urgency"],
        "suggestion_source": "rule+gpt2",
        "report_url": report_url,
    }



# ================= BREAST =================
@predict_router.post("/breast")
def predict_breast(
    payload: TabularInput,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    data = payload.data

    pred_idx, prob, probs = run_breast_tabular(breast_model, data)
    label = "Malignant" if pred_idx == 1 else "Benign"

    comparisons, chart_data = build_comparisons("breast", data)
    rule_out = rule_engine("breast", label, prob, data)
    base_advice = rule_out["base_advice"]

    context = (
        f"Breast cancer prediction: {label} with probability {prob:.2f}. "
        f"Key features: {data}. Clinical-style base advice: {base_advice}. "
        "Write 3–5 clear, supportive counselling sentences for the patient. "
        "Do not repeat any sentence."
    )
    suggestion_text = generate_advice_with_gpt2(context, "breast", label, prob)


    report_id = uuid.uuid4().hex
    pdf_path = generate_pdf(
        report_id,
        {
            "disease": "breast",
            "label": label,
            "probability": prob,
            "comparisons": comparisons,
            "advisory": suggestion_text,
            "gradcam_path": "",
        },
    )
    report_url = "/reports/" + os.path.basename(pdf_path)

    rec = History(
        user_id=current_user.id,
        disease="breast",
        input_data=json.dumps(data),
        prediction=label,
        probability=str(prob),
        report_path=report_url,
        gradcam_path="",
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    return {
        "id": rec.id,
        "disease": "breast",
        "label": label,
        "probability": prob,
        "raw_probs": probs,
        "comparisons": comparisons,
        "chart_data": chart_data,
        "suggestion_text": suggestion_text,
        "lifestyle_recommendations": rule_out["lifestyle_recommendations"],
        "urgency": rule_out["urgency"],
        "suggestion_source": "rule+gpt2",
        "report_url": report_url,
    }


# ================= BRAIN =================
@predict_router.post("/brain")
async def predict_brain(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # -----------------------------
    # SAVE IMAGE
    # -----------------------------
    os.makedirs("backend/temp", exist_ok=True)
    img_path = f"backend/temp/{uuid.uuid4().hex}_{file.filename}"

    with open(img_path, "wb") as f:
        f.write(await file.read())

    # -----------------------------
    # PREPROCESS
    # -----------------------------
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, 0).astype("float32")

    # -----------------------------
    # PREDICT
    # -----------------------------
    preds = brain_model.predict(img)[0]
    pred_class = int(np.argmax(preds))
    prob = float(preds[pred_class])
    label = class_map[str(pred_class)]

    # -----------------------------
    # GRAD-CAM
    # -----------------------------
    gradcam_url = ""
    try:
        grad_path = generate_gradcam(brain_model, img_path, pred_class)
        gradcam_url = "/gradcam/" + os.path.basename(grad_path)
    except Exception as e:
        print("GradCAM error:", e)

    # -----------------------------
    # RULE ENGINE + GPT-2
    # -----------------------------
    rule_out = rule_engine("brain", label, prob, {})
    base_text = rule_out.get("base_advice", "")

    context = (
        f"Brain MRI prediction: {label} with probability {prob:.2f}. "
        f"Base clinical guidance: {base_text}. "
    )

    try:
        suggestion_text = generate_advice_with_gpt2(context, "brain", label, prob)

    except:
        suggestion_text = (
            "This AI analysis is intended for screening support only. "
            "Please consult a qualified clinician for further evaluation."
        )

    # -----------------------------
    # GENERATE PDF
    # -----------------------------
    report_id = uuid.uuid4().hex
    pdf_path = generate_pdf(
        report_id,
        {
            "disease": "Brain",
            "label": label,
            "probability": prob,
            "comparisons": [],
            "advisory": suggestion_text,
            "gradcam_path": gradcam_url,
        },
    )
    report_url = "/reports/" + os.path.basename(pdf_path)

    # -----------------------------
    # SAVE HISTORY
    # -----------------------------
    rec = History(
        user_id=current_user.id,
        disease="brain",
        input_data=json.dumps({"file": img_path}),
        prediction=label,
        probability=str(prob),
        report_path=report_url,
        gradcam_path=gradcam_url,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    # -----------------------------
    # RETURN CLEAN JSON
    # -----------------------------
    return {
        "id": rec.id,
        "disease": "brain",
        "label": label,
        "probability": prob,
        "gradcam_url": gradcam_url,
        "suggestion_text": suggestion_text,
        "lifestyle_recommendations": rule_out["lifestyle_recommendations"],
        "urgency": rule_out["urgency"],
        "report_url": report_url,
    }
