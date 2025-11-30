import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype

# ------------ LOAD DATA ------------

df = pd.read_excel("Hospital_Patient_Data.xlsx")
df.columns = df.columns.str.strip()

def find_col(candidates_contains, required=True):
    candidates_contains = [s.lower() for s in candidates_contains]
    for col in df.columns:
        name = col.lower()
        if all(s in name for s in candidates_contains):
            return col
    if required:
        raise ValueError(f"Column containing {candidates_contains} not found. Columns: {list(df.columns)}")
    return None

age_col = find_col(["age"])
gender_col = find_col(["gender"])
admin_col = find_col(["admin"], required=False)
if admin_col is None:
    admin_col = find_col(["admission"])

ref_col = find_col(["referr"], required=False)
if ref_col is None:
    ref_col = find_col(["department"], required=False)

# optional: patient id column
patient_id_col = find_col(["patient"], required=False)

wait_col = None
for col in df.columns:
    if "wait" in col.lower() and is_numeric_dtype(df[col]):
        wait_col = col
        break
if wait_col is None:
    raise ValueError("Waiting time column not found (needs 'wait' in name and numeric type).")

rename_map = {
    age_col: "Age",
    gender_col: "Gender",
    admin_col: "AdminType",
    wait_col: "WaitingTime",
}
if ref_col is not None:
    rename_map[ref_col] = "ReferralDept"
if patient_id_col is not None:
    rename_map[patient_id_col] = "PatientID"

df = df.rename(columns=rename_map)
df = df.dropna(subset=["WaitingTime"])

# ------------ FEATURES FOR MODEL (NO PATIENT ID, NO SATISFACTION) ------------

base_features = ["Age", "Gender", "AdminType", "ReferralDept"]
feature_cols = [c for c in base_features if c in df.columns]

if not feature_cols:
    raise ValueError("No usable feature columns found. Check that Age, Gender, Admin, Referral Dept exist.")

X = df[feature_cols]
y = df["WaitingTime"]

numeric_features = [c for c in feature_cols if is_numeric_dtype(X[c])]
categorical_features = [c for c in feature_cols if not is_numeric_dtype(X[c])]

numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", RandomForestRegressor(n_estimators=300, random_state=42)),
])

model.fit(X, y)

gender_values = sorted(df["Gender"].dropna().unique().tolist()) if "Gender" in df.columns else []
admin_values = sorted(df["AdminType"].dropna().unique().tolist()) if "AdminType" in df.columns else []
ref_values = sorted(df["ReferralDept"].dropna().unique().tolist()) if "ReferralDept" in df.columns else []
patient_ids = sorted(df["PatientID"].dropna().unique().tolist()) if "PatientID" in df.columns else []

# ------------ STREAMLIT UI ------------

st.set_page_config(page_title="Hospital Waiting Time Predictor", layout="wide")

st.markdown(
    """
    <div style="background:#eaf3ff;padding:20px;border-radius:12px;
                display:flex;align-items:center;gap:12px;justify-content:center">
        <div style="font-size:32px;">🏥</div>
        <div>
            <h2 style="color:#1f4f80;margin:0;">Hospital Management Dashboard</h2>
            <p style="color:#4f7fb3;margin:0;">Predictive Waiting Time System</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div style="background:#ffffff;padding:18px;border-radius:12px;
                    border:1px solid #dbe9ff;box-shadow:0 1px 3px rgba(15,23,42,0.06);">
        <h4 style="color:#1f4f80;margin-bottom:10px">Patient Input</h4>
        """,
        unsafe_allow_html=True,
    )

    if patient_ids:
        patient_id = st.selectbox("Patient ID", patient_ids)
    else:
        patient_id = st.text_input("Patient ID")

    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    visiting_hour = st.number_input("Visiting Hour (0–23)", min_value=0, max_value=23, value=10)

    if gender_values:
        gender = st.selectbox("Gender", gender_values)
    else:
        gender = st.text_input("Gender")

    if admin_values:
        admin_type = st.selectbox("Admission Type", admin_values)
    else:
        admin_type = st.text_input("Admission Type")

    if ref_values:
        referral_dept = st.selectbox("Referral Department", ref_values)
    else:
        referral_dept = st.text_input("Referral Department")

    predict_click = st.button("Predict Waiting Time")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(
        """
        <div style="background:#ffffff;padding:18px;border-radius:12px;
                    border:1px solid #dbe9ff;box-shadow:0 1px 3px rgba(15,23,42,0.06);text-align:center">
        <h4 style="color:#1f4f80;margin-bottom:10px">Predicted KPI</h4>
        """,
        unsafe_allow_html=True,
    )

    if predict_click:
        row = {}
        if "Age" in feature_cols: row["Age"] = age
        if "Gender" in feature_cols: row["Gender"] = gender
        if "AdminType" in feature_cols: row["AdminType"] = admin_type
        if "ReferralDept" in feature_cols: row["ReferralDept"] = referral_dept

        df_input = pd.DataFrame([row])[feature_cols]
        pred = model.predict(df_input)[0]
        st.markdown(f"<h2 style='color:#1f4f80'>{pred:.2f} minutes</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:#9ca3af'>Enter details and click Predict</h3>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
