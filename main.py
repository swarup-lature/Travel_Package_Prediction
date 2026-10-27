from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="TripWise | Travel Package Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Custom styling -----------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --ink:#172033; --muted:#68738a; --blue:#3157d5; --teal:#18a999; }
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #f5f7fc; }
      [data-testid="stHeader"] { background: rgba(245,247,252,.92); }
      .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1220px; }
      [data-testid="stSidebar"] { background: #111a31; }
      [data-testid="stSidebar"] * { color: #edf2ff !important; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b7c2df !important; }
      .brand { font-family:'Manrope',sans-serif; font-size:1.1rem; font-weight:800; letter-spacing:.2px; }
      .hero { background:linear-gradient(120deg,#17264c 0%,#3157d5 68%,#4779e8 100%); padding:2rem 2.1rem; border-radius:24px; color:white; margin-bottom:1.25rem; box-shadow:0 14px 36px rgba(39,70,160,.18); }
      .hero h1 { font-family:'Manrope',sans-serif; font-size:2.15rem; line-height:1.15; margin:0 0 .65rem 0; color:white; }
      .hero p { margin:0; color:#e0e8ff; font-size:1rem; max-width:740px; }
      .eyebrow { text-transform:uppercase; font-size:.72rem; font-weight:700; letter-spacing:1.7px; color:#a9c1ff; margin-bottom:.65rem; }
      .section-title { font-family:'Manrope',sans-serif; color:#172033; font-weight:800; font-size:1.18rem; margin:.3rem 0 .2rem; }
      .section-sub { color:#68738a; font-size:.9rem; margin-bottom:1rem; }
      .panel { background:white; border:1px solid #e6eaf3; border-radius:18px; padding:1.25rem 1.35rem; }
      .mini-card { background:white; border:1px solid #e6eaf3; border-radius:16px; padding:1rem 1.15rem; min-height:96px; }
      .mini-label { color:#68738a; font-size:.78rem; font-weight:600; margin-bottom:.4rem; }
      .mini-value { color:#172033; font-family:'Manrope',sans-serif; font-weight:800; font-size:1.3rem; }
      div[data-testid="stForm"] { background:white; border:1px solid #e6eaf3; border-radius:18px; padding:1.2rem 1.35rem 1.4rem; }
      .stButton > button, [data-testid="stFormSubmitButton"] > button { border:0; border-radius:11px; min-height:3rem; font-weight:700; background:linear-gradient(100deg,#3157d5,#4779e8); color:white; box-shadow:0 6px 14px rgba(49,87,213,.18); }
      .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { border:0; background:#2447bd; color:white; }
      div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { border-radius:10px; }
      .result-good { background:#e8f8f2; border:1px solid #bcebd9; color:#12664f; padding:1rem 1.1rem; border-radius:14px; font-weight:700; }
      .result-warn { background:#fff5e7; border:1px solid #f4d8a7; color:#8a5200; padding:1rem 1.1rem; border-radius:14px; font-weight:700; }
      .footnote { color:#7c879b; font-size:.78rem; text-align:center; padding-top:1.5rem; }
      hr { border-color:#e6eaf3; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown('<div class="brand">✈️ TripWise</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Travel package insights")
    st.markdown("Use customer and trip details to estimate the likelihood of a travel package purchase.")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Enter customer details.\n2. Select **Predict**.\n3. Review the model result.")
    st.markdown("---")
    st.caption("Machine-learning demo · Predictions are estimates")

# ----------------------------- Hero -----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">CUSTOMER ANALYTICS · ML PREDICTION</div>
      <h1>Travel Package Predictor</h1>
      <p>Understand whether a customer may purchase a travel package. Enter the customer profile below to generate a prediction from your trained model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick feature cards
k1, k2, k3 = st.columns(3)
for col, label, value in [
    (k1, "INPUT", "Customer profile"),
    (k2, "ENGINE", "Trained ML model"),
    (k3, "OUTPUT", "Purchase likelihood"),
]:
    with col:
        st.markdown(
            f'<div class="mini-card"><div class="mini-label">{label}</div><div class="mini-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
st.markdown('<div class="section-title">Customer details</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Complete the fields. Values start with sensible examples and can be adjusted.</div>', unsafe_allow_html=True)

# ----------------------------- Load model assets -----------------------------
@st.cache_resource
def load_assets():
    model_path = Path("model.joblib")
    prep_path = Path("preprocessor.joblib")
    if not model_path.exists() or not prep_path.exists():
        missing = [p.name for p in (model_path, prep_path) if not p.exists()]
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))
    saved_models = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    if isinstance(saved_models, dict):
        model = saved_models.get("Xg boost", next(iter(saved_models.values())))
    else:
        model = saved_models
    return model, preprocessor

# ----------------------------- Input form -----------------------------
with st.form("customer_input_form", clear_on_submit=False):
    st.markdown("#### 01 · Customer profile")
    a, b, c = st.columns(3)
    with a:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital status", ["Married", "Unmarried", "Divorced"])
    with b:
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly income", min_value=0, max_value=1000000, value=25000, step=1000)
    with c:
        city_tier = st.selectbox("City tier", [1, 2, 3])
        contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        passport = st.selectbox("Has passport?", ["No", "Yes"])

    st.markdown("---")
    st.markdown("#### 02 · Travel & sales interaction")
    d, e, f = st.columns(3)
    with d:
        product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        property_star = st.selectbox("Preferred property star", [3, 4, 5])
        trips = st.number_input("Number of trips", min_value=0, max_value=100, value=2, step=1)
    with e:
        duration = st.number_input("Duration of pitch (minutes)", min_value=0, max_value=300, value=15, step=1)
        followups = st.number_input("Number of follow-ups", min_value=0, max_value=50, value=3, step=1)
        pitch_satisfaction = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5])
    with f:
        own_car = st.selectbox("Owns a car?", ["No", "Yes"])
        persons_visiting = st.number_input("Adults / persons visiting", min_value=1, max_value=20, value=2, step=1)
        children_visiting = st.number_input("Children visiting", min_value=0, max_value=20, value=0, step=1)

    st.write("")
    submitted = st.form_submit_button("✦  Predict purchase likelihood", use_container_width=True)

# ----------------------------- Prediction -----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "TotalVisiting": persons_visiting + children_visiting,
    }])

    try:
        model, preprocessor = load_assets()
        transformed_input = preprocessor.transform(input_data)
        prediction = int(model.predict(transformed_input)[0])
        st.markdown("---")
        st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
        left, right = st.columns([1.3, 1])
        with left:
            if prediction == 1:
                st.markdown('<div class="result-good">✓  Higher purchase intent predicted</div>', unsafe_allow_html=True)
                st.success("The model predicts that this customer may purchase the travel package.")
            else:
                st.markdown('<div class="result-warn">↗  Lower purchase intent predicted</div>', unsafe_allow_html=True)
                st.info("The model predicts that this customer may not purchase the travel package.")
        with right:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed_input)[0]
                classes = list(getattr(model, "classes_", range(len(probabilities))))
                positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
                probability = float(probabilities[positive_index]) * 100
                st.metric("Estimated purchase probability", f"{probability:.1f}%")
                st.progress(max(0.0, min(1.0, probability / 100)))
            else:
                st.caption("This model does not provide probability estimates.")
        with st.expander("View submitted feature values"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)
    except FileNotFoundError as exc:
        st.error(str(exc) + ". Place both files in the same folder as main.py.")
    except Exception as exc:
        st.error("Prediction could not be completed. Check that model.joblib and preprocessor.joblib match this notebook and its feature columns.")
        st.exception(exc)

st.markdown('<div class="footnote">TripWise · Travel Package Purchase Prediction · For demonstration purposes only</div>', unsafe_allow_html=True)
