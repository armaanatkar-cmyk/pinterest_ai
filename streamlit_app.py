import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import StringIO

st.set_page_config(page_title="Pinterest Campaign Scorer", page_icon="📈")

@st.cache_resource
def load_artifacts(model_path):
    art = joblib.load(model_path)
    return art

def prepare_features(df, numeric_features, one_hot_features, feature_order):
    if "CTR_pct" not in df.columns:
        if "Pin clicks" in df.columns and "Impressions" in df.columns:
            df["CTR_pct"] = np.where(df["Impressions"] > 0, (df["Pin clicks"] / df["Impressions"]) * 100, 0.0)
        else:
            df["CTR_pct"] = 0.0

    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = dt.dt.month.fillna(0).astype(int)
        df["DayOfWeek"] = dt.dt.dayofweek.fillna(0).astype(int)
    else:
        for c in ["Month", "DayOfWeek"]:
            if c not in df.columns:
                df[c] = 0

    if "Campaign status" in df.columns:
        status_dummies = pd.get_dummies(df["Campaign status"], prefix="Status")
    else:
        status_dummies = pd.DataFrame()

    for col in one_hot_features:
        if col not in status_dummies.columns:
            status_dummies[col] = 0
    status_dummies = status_dummies[one_hot_features] if one_hot_features else pd.DataFrame()

    X_num = df.reindex(columns=numeric_features, fill_value=0).astype(float)
    X = pd.concat([X_num, status_dummies], axis=1)
    X = X.reindex(columns=feature_order, fill_value=0)
    return X

st.title("📈 Pinterest Campaign Scorer")
st.caption("Upload a CSV of campaigns. Get WIN probabilities + labels.")

model_path = "pinterest_baseline_logreg.pkl"
art = load_artifacts(model_path)
numeric_features = art["numeric_features"]
one_hot_features = art["one_hot_features"]
feature_order = art["feature_order"]
scaler = art["scaler"]
model = art["model"]

uploaded = st.file_uploader("Upload CSV", type=["csv"])

threshold = st.slider("WIN threshold", 0.0, 1.0, 0.5, 0.01)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    X = prepare_features(df.copy(), numeric_features, one_hot_features, feature_order)

    X_num = scaler.transform(X[numeric_features])
    X_oh = X.drop(columns=numeric_features).to_numpy()
    X_final = np.hstack([X_num, X_oh])

    probs = model.predict_proba(X_final)[:, 1]
    preds = (probs >= threshold).astype(int)

    out_df = df.copy()
    out_df["win_probability"] = probs
    out_df["predicted_label"] = np.where(preds == 1, "WINNER", "NOT_WINNER")

    st.success(f"Scored {len(out_df)} rows.")
    st.dataframe(out_df.head(50))

    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.markdown("**Required columns** (any extras are ignored):")
st.code(", ".join(sorted(set(numeric_features + ['Date','Campaign status','Pin clicks','Impressions']))))
