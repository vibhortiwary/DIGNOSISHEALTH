# backend/gpt2_generator.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# -------------------------------
# Load GPT-2
# -------------------------------
MODEL_NAME = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()


# ============================================================
# FALLBACK: GPT-4-STYLE TEMPLATE (Deterministic)
# ============================================================
def fallback_template(disease: str, label: str, prob: float) -> str:
    p = round(prob * 100)

    disease = disease.lower()
    label = label.lower()

    # ----------------- DIABETES -----------------
    if disease == "diabetes":
        return f"""
Clinical Interpretation:
Based on your screening inputs, the pattern is consistent with a "{label}" metabolic profile, with an estimated probability of {p}%. This suggests alterations in glucose regulation that may require lifestyle adjustments or confirmatory blood testing.

Recommended Follow-ups:
- Consider blood tests such as HbA1c or fasting glucose.
- Review eating patterns, weight trends, and family history.
- Schedule a clinical evaluation if symptoms such as fatigue, frequent urination, or excessive thirst occur.

General Care Tips:
- Reduce added sugars and refined carbohydrates.
- Increase daily activity (20–30 minutes walking).
- Maintain hydration and regular sleep routine.
        """.strip()

    # ----------------- HEART -----------------
    if disease == "heart":
        return f"""
Clinical Interpretation:
Your results indicate a "{label}" cardiovascular risk profile, estimated at {p}%. While this does not diagnose heart disease, it suggests the need for monitoring and possibly further evaluation.

Recommended Follow-ups:
- Check blood pressure, lipid profile, and consider an ECG.
- Discuss your risk factors with a cardiologist if symptoms are present.
- Track chest discomfort, exertional breathlessness, or palpitations.

General Care Tips:
- Reduce salt intake and avoid smoking.
- Practice moderate physical activity most days of the week.
- Maintain a heart-healthy diet rich in fruits and fiber.
        """.strip()

    # ----------------- BREAST -----------------
    if disease == "breast":
        return f"""
Clinical Interpretation:
Model findings suggest a pattern consistent with "{label}" with a probability of {p}%. This represents an imaging-based assessment and requires radiological and clinical confirmation.

Recommended Follow-ups:
- Obtain a mammogram or ultrasound if not done recently.
- Consult a breast specialist for physical examination.
- Monitor for new lumps, skin dimpling, or changes in breast shape.

General Care Tips:
- Maintain a healthy weight and limit alcohol.
- Perform routine breast self-awareness checks.
- Follow your physician's screening schedule.
        """.strip()

    # ----------------- BRAIN -----------------
    if disease == "brain":
        if "glioma" in label:
            subtype_info = "a glioma pattern, which is a primary tumor arising from glial cells."
        elif "meningioma" in label:
            subtype_info = "a meningioma pattern, often slow-growing and arising from the meninges."
        elif "pituitary" in label:
            subtype_info = "a pituitary-region mass pattern, which can influence hormone levels."
        elif "notumor" in label or "no tumor" in label:
            subtype_info = "no tumor-like abnormality."
        else:
            subtype_info = "an indeterminate MRI pattern."

        return f"""
Clinical Interpretation:
The MRI appearance corresponds to {subtype_info} Estimated model confidence: {p}%. This is only an AI-assisted interpretation and requires clinical correlation.

Recommended Follow-ups:
- Review MRI findings with a neurologist or neurosurgeon.
- Consider follow-up imaging for progression monitoring.
- Report headaches, visual symptoms, seizures, or new weakness promptly.

General Care Tips:
- Maintain regular sleep and hydration.
- Avoid smoking and alcohol.
- Engage in stress-reduction practices such as light exercise.
        """.strip()

    # ----------------- UNKNOWN DISEASE FALLBACK -----------------
    return f"""
Clinical Interpretation:
Your results indicate a prediction of "{label}" with an estimated probability of {p}%. Further evaluation is recommended for clarification.

Recommended Follow-ups:
- Discuss findings with a healthcare professional.
- Consider confirmatory testing based on symptoms and risk factors.

General Care Tips:
- Maintain balanced nutrition.
- Stay active and well-hydrated.
- Monitor symptoms and seek care if they worsen.
    """.strip()


# ============================================================
# GPT-2 ADVISORY GENERATOR (With Safety & Fallback)
# ============================================================
def generate_advice_with_gpt2(context: str, disease: str, label: str, probability: float) -> str:

    # -------------------------------
    # Strong Structured Prompt
    # -------------------------------
    prompt = f"""
You are a clinical advisory generator. Based on the findings below, generate a clear and structured medical-style advisory.

FINDINGS:
{context}

Your response MUST follow this structure:

Clinical Interpretation:
- Short, factual explanation of what the finding suggests.

Recommended Follow-ups:
- 2 to 3 practical next steps.
- Mention specialists only if relevant.

General Care Tips:
- 2 to 3 lifestyle recommendations.
- Avoid exaggeration or diagnosis claims.

Important rules:
- NEVER say "I am an AI".
- NEVER say "I cannot".
- NEVER give emergency instructions.
- Keep entire output between 120 and 170 words.
Begin now:
"""

    # Encode
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 180,
                temperature=0.65,
                top_p=0.85,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )

        full = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip prompt
        if "Begin now:" in full:
            advisory = full.split("Begin now:")[-1].strip()
        else:
            advisory = full.strip()

        # -------------------------------
        # INVALID OUTPUT CHECK (Fixes ..)
        # -------------------------------
        if (
            not advisory
            or advisory in [".", "..", "..."]
            or len(advisory) < 40
            or "Clinical Interpretation" not in advisory
        ):
            raise ValueError("GPT-2 output invalid")

        return advisory

    except Exception:
        # GPT-2 failed → use deterministic fallback
        return fallback_template(disease, label, probability)
