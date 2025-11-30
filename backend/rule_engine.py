# backend/rule_engine.py
from typing import Dict, Any, List


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(v)
    except Exception:
        return default


def rule_engine(
    disease: str,
    prediction_label: str,
    probability: float,
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Central rule engine.
    - Uses model label + probability
    - Adds rule-based checks on key inputs (BMI, BP, cholesterol, etc.)
    - Returns:
        urgency: "low" | "medium" | "high"
        base_advice: short clinical-style explanation
        lifestyle_recommendations: list[str]
        override_label: optional manual override for model label (used for heart)
    """

    lifestyle: List[str] = []
    urgency = "low"
    base_advice = ""
    override_label = None

    prob_pct = probability * 100
    label_lower = (prediction_label or "").lower()

    # ------------------------------------------------------------------
    # HEART DISEASE  (UCI heart features)
    # ------------------------------------------------------------------
    if disease == "heart":
        age = _to_float(inputs.get("age"))
        trestbps = _to_float(inputs.get("trestbps"))     # resting BP
        chol = _to_float(inputs.get("chol"))             # cholesterol
        thalach = _to_float(inputs.get("thalach"))       # max heart rate
        fbs = _to_int(inputs.get("fbs"))                 # fasting blood sugar > 120
        exang = _to_int(inputs.get("exang"))             # exercise angina
        oldpeak = _to_float(inputs.get("oldpeak"))
        slope = _to_int(inputs.get("slope"))
        ca = _to_int(inputs.get("ca"))

        risk_score = 0
        critical = False

        # --- Age ---
        if age >= 60:
            risk_score += 2
        elif age >= 50:
            risk_score += 1

        # --- Blood pressure ---
        if trestbps >= 160:
            risk_score += 2
            critical = True
        elif trestbps >= 140:
            risk_score += 1

        # --- Cholesterol ---
        if chol >= 280:
            risk_score += 2
            critical = True
        elif chol >= 240:
            risk_score += 1

        # --- Max heart rate (low is bad) ---
        if thalach <= 100:
            risk_score += 2
        elif thalach <= 120:
            risk_score += 1

        # --- Other risk markers ---
        if fbs == 1:
            risk_score += 1
        if exang == 1:
            risk_score += 1
        if oldpeak >= 2.0:
            risk_score += 1
        if slope == 2:  # downsloping
            risk_score += 1
        if ca >= 1:     # visible vessels
            risk_score += 1

        # Decide overall severity
        model_high = prediction_label.lower() == "high risk"
        rule_high_risk = critical or risk_score >= 6
        rule_medium_risk = (3 <= risk_score < 6)

        if model_high or prob_pct >= 70 or rule_high_risk:
            urgency = "high"
            override_label = "High Risk"

            if rule_high_risk and not model_high:
                reason = (
                    "Based on your blood pressure, cholesterol and exercise-related values, "
                    "your profile appears high risk even though the model confidence is moderate. "
                )
            else:
                reason = (
                    "The model output and your entered parameters together suggest a high risk "
                    "for significant heart disease. "
                )

            base_advice = (
                reason
                + "You should arrange an appointment with a cardiologist as soon as possible "
                "for detailed evaluation and confirmatory tests."
            )

            lifestyle += [
                "Completely avoid smoking and limit or avoid alcohol.",
                "Reduce salt, deep-fried food and saturated fat in your diet.",
                "If your doctor allows, start gentle daily walking (15–30 minutes) and build up gradually.",
                "Check your blood pressure and cholesterol regularly and record the values.",
            ]
        elif rule_medium_risk or prob_pct >= 55:
            urgency = "medium"
            base_advice = (
                "Your current parameters and model prediction suggest a borderline or moderate "
                "heart risk. Addressing lifestyle factors now can prevent progression."
            )
            lifestyle += [
                "Aim for at least 30 minutes of moderate physical activity on most days.",
                "Choose a heart-friendly diet rich in fruits, vegetables, whole grains and lean protein.",
                "Limit salt intake and avoid frequent fast food or sugary drinks.",
                "Discuss your risk with a physician within the next few months, especially if you notice symptoms.",
            ]
        else:
            urgency = "low"
            base_advice = (
                "Your current heart risk profile appears lower according to both the model and "
                "the entered parameters. Maintaining healthy habits is still important."
            )
            lifestyle += [
                "Continue regular physical activity that is comfortable for you.",
                "Keep a balanced diet with limited processed and fried foods.",
                "Monitor blood pressure and cholesterol during routine check-ups.",
                "Manage stress with relaxation techniques, hobbies or counselling if needed.",
            ]

    # ------------------------------------------------------------------
    # DIABETES (BRFSS-style features; includes BMI, activity, BP, etc.)
    # ------------------------------------------------------------------
    elif disease == "diabetes":
        bmi = _to_float(inputs.get("BMI"))
        high_bp = _to_int(inputs.get("HighBP"))
        high_chol = _to_int(inputs.get("HighChol"))
        smoker = _to_int(inputs.get("Smoker"))
        phys = _to_int(inputs.get("PhysActivity"))
        genhlth = _to_int(inputs.get("GenHlth"))   # 1=Excellent ... 5=Poor
        menthlth = _to_float(inputs.get("MentHlth"))
        age_code = _to_int(inputs.get("Age"))

        risk_score = 0

        # BMI contribution
        if bmi >= 35:
            risk_score += 3
        elif bmi >= 30:
            risk_score += 2
        elif bmi >= 25:
            risk_score += 1

        # Blood pressure / cholesterol
        if high_bp == 1:
            risk_score += 2
        if high_chol == 1:
            risk_score += 1

        # Lifestyle & mental health
        if phys == 0:
            risk_score += 2
        if smoker == 1:
            risk_score += 1
        if genhlth >= 4:
            risk_score += 1
        if menthlth >= 10:
            risk_score += 1

        # Age (BRFSS age group; 9 ≈ 60–64 and above)
        if age_code >= 9:
            risk_score += 1

        is_diabetes = "diabetes" in label_lower and "pre" not in label_lower
        is_prediabetes = "pre" in label_lower

        # Severity decision
        if is_diabetes or (risk_score >= 6 and prob_pct >= 60):
            urgency = "high"
            base_advice = (
                "Your profile and model output suggest a high likelihood of diabetes. "
                "You should consult a doctor or endocrinologist soon for blood tests, "
                "treatment planning and regular follow-up."
            )
        elif is_prediabetes or (3 <= risk_score <= 5) or (50 <= prob_pct < 60):
            urgency = "medium"
            base_advice = (
                "Your results fall in a prediabetes or higher-than-normal risk range. "
                "This is a warning stage where lifestyle changes can often prevent or delay "
                "type 2 diabetes."
            )
        else:
            urgency = "low"
            base_advice = (
                "Your current diabetes risk appears relatively low. "
                "Continuing healthy habits will help maintain this profile."
            )

        # Tailored lifestyle bullets
        # Weight / BMI
        if bmi >= 25:
            lifestyle.append(
                "Aim for gradual weight reduction (about 0.5–1 kg per month) through diet and activity."
            )
            lifestyle.append(
                "Reduce portion sizes of high-calorie foods and avoid frequent sugary drinks."
            )
        else:
            lifestyle.append(
                "Maintain your current healthy weight with balanced meals and regular activity."
            )

        # Physical activity
        if phys == 0:
            lifestyle.append(
                "Try to build up to at least 150 minutes per week of moderate exercise "
                "such as brisk walking, cycling or swimming."
            )
        else:
            lifestyle.append(
                "Continue regular physical activity and try to include both aerobic and light strength exercises."
            )

        # Blood pressure / cholesterol
        if high_bp == 1 or high_chol == 1:
            lifestyle.append(
                "Limit salt, saturated fats and fried foods; choose home-cooked meals, vegetables and whole grains."
            )

        # Smoking
        if smoker == 1:
            lifestyle.append(
                "If you smoke, consider a structured quitting plan or cessation program—smoking greatly increases diabetes complications."
            )

        # General/mental health
        if menthlth >= 10 or genhlth >= 4:
            lifestyle.append(
                "Pay attention to sleep, stress and mental health; speak with a clinician if low mood or stress is persistent."
            )

        # Always add a generic safety tip
        lifestyle.append(
            "Schedule regular check-ups to monitor blood sugar (fasting glucose, HbA1c), "
            "blood pressure and cholesterol."
        )

    # ------------------------------------------------------------------
    # BREAST CANCER (Benign vs Malignant)
    # ------------------------------------------------------------------
    elif disease == "breast":
        if "malignant" in label_lower or ("cancer" in label_lower and "no" not in label_lower):
            urgency = "high"
            base_advice = (
                "The prediction pattern is concerning for a malignant breast lesion. "
                "You should urgently see a breast specialist or oncologist for examination, "
                "imaging and possible biopsy to confirm the diagnosis."
            )
            lifestyle += [
                "Do not delay follow-up appointments, imaging or biopsy recommended by your doctor.",
                "Gather any previous imaging and pathology reports and bring them to your consultation.",
                "Discuss your personal and family history of breast or ovarian cancer with your clinician.",
                "Seek emotional support from trusted family, friends or counsellors during this period.",
            ]
        elif prob_pct >= 60:
            urgency = "medium"
            base_advice = (
                "The model output is less concerning but still suggests you should arrange "
                "a follow-up visit. Your doctor may advise repeat imaging or closer observation."
            )
            lifestyle += [
                "Follow your doctor’s schedule for repeat imaging or check-ups.",
                "Be familiar with your usual breast appearance and report any new changes promptly.",
                "Maintain a healthy lifestyle including regular exercise and limited alcohol intake.",
            ]
        else:
            urgency = "low"
            base_advice = (
                "The current breast cancer prediction appears reassuring. "
                "Even with low predicted risk, regular screening and awareness of any breast changes "
                "remain important."
            )
            lifestyle += [
                "Keep up with age-appropriate screening (clinical exam, mammogram or ultrasound as advised).",
                "Report any new lumps, skin changes, nipple discharge or persistent pain.",
                "Maintain a healthy weight, exercise regularly and limit alcohol consumption.",
            ]

    # ------------------------------------------------------------------
    # BRAIN MRI (glioma / meningioma / pituitary / notumor)
    # ------------------------------------------------------------------
    elif disease == "brain":
        label = label_lower

        # Default safe outputs
        urgency = "medium"
        base_advice = (
            "MRI findings should always be interpreted together with your symptoms by a neurologist "
            "or neurosurgeon. This AI result is only a screening aid."
        )
        lifestyle = [
            "Maintain good hydration and a balanced diet.",
            "Ensure 7–8 hours of sleep each night where possible.",
            "Avoid smoking and excess alcohol, which can worsen overall neurological health.",
            "Monitor for headaches, vision changes, seizures or new weakness and report them promptly.",
        ]

        # Glioma
        if "glioma" in label:
            urgency = "high"
            base_advice = (
                "The pattern is suggestive of a glioma, a primary brain tumor arising from glial cells. "
                "Urgent review by a neurosurgeon or neuro-oncologist is recommended for further imaging "
                "and possible biopsy to determine the tumor grade and treatment plan."
            )
            lifestyle = [
                "Avoid activities where a seizure could be dangerous (driving, swimming alone) until cleared by a doctor.",
                "Keep a regular sleep schedule to reduce seizure risk and neurological stress.",
                "Avoid alcohol and smoking to minimise additional brain and vascular stress.",
                "Engage only in light activities such as walking unless your doctor advises otherwise.",
                "Keep a diary of headaches, weakness, speech or vision changes and bring it to consultations.",
            ]

        # Meningioma
        elif "meningioma" in label:
            urgency = "high"
            base_advice = (
                "The pattern is consistent with a meningioma, a tumor arising from the brain coverings (meninges). "
                "Many meningiomas are benign but can compress brain tissue depending on size and location. "
                "Evaluation by a neurosurgeon is recommended to decide on monitoring versus treatment."
            )
            lifestyle = [
                "Avoid heavy physical strain if you experience dizziness, headaches or balance problems.",
                "Try to maintain calm daily routines and manage stress levels.",
                "Avoid smoking and heavy alcohol intake.",
                "Follow the MRI follow-up schedule recommended by your specialist.",
                "Seek immediate medical care if you notice new seizures, weakness, vision loss or speech problems.",
            ]

        # Pituitary tumor
        elif "pituitary" in label:
            urgency = "high"
            base_advice = (
                "The pattern suggests a pituitary region lesion. Pituitary tumors can affect hormone levels "
                "and vision. You should see an endocrinologist and neurosurgeon for hormone tests and "
                "specialised pituitary MRI to decide on treatment."
            )
            lifestyle = [
                "Notice symptoms such as fatigue, unexpected weight change, menstrual changes, decreased libido or mood changes and report them.",
                "Avoid activities that rely heavily on perfect vision until your visual fields are assessed.",
                "Maintain a balanced diet and regular sleep schedule to support hormonal balance.",
                "Seek urgent medical care for sudden severe headache or sudden vision loss.",
            ]

        # No tumor
        elif "notumor" in label or "no tumor" in label or "no_tumor" in label:
            urgency = "low"
            base_advice = (
                "No tumor-like abnormality is detected by the model on this MRI slice. "
                "If symptoms persist, a clinician should still review your scans and examine you "
                "to look for other neurological causes."
            )
            lifestyle = [
                "Continue healthy sleep and hydration habits.",
                "Limit excessive screen time if you have frequent headaches.",
                "Engage in light regular exercise such as walking or yoga as tolerated.",
                "Avoid smoking and excess alcohol to support long-term brain health.",
            ]

    # ------------------------------------------------------------------
    # Fallback (unknown disease key)
    # ------------------------------------------------------------------
    else:
        urgency = "low"
        base_advice = (
            "No specific rule-set was found for this condition. "
            "Use this result only as a screening aid and consult a clinician for full interpretation."
        )
        lifestyle = [
            "Maintain a balanced diet and regular physical activity.",
            "Avoid smoking and limit alcohol consumption.",
            "Keep regular medical check-ups according to your doctor’s advice.",
        ]

    return {
        "urgency": urgency,
        "base_advice": base_advice,
        "lifestyle_recommendations": lifestyle,
        "override_label": override_label,
    }
