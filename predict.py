#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import joblib
import os

def load_artifacts(model_path):
    art = joblib.load(model_path)
    return (
        art["numeric_features"],
        art["one_hot_features"],
        art["feature_order"],
        art["scaler"],
        art["model"],
    )

def prepare_features(df, numeric_features, one_hot_features, feature_order):
    # Ensure CTR_pct exists (compute if columns present)
    if "CTR_pct" not in df.columns:
        if "Pin clicks" in df.columns and "Impressions" in df.columns:
            df["CTR_pct"] = np.where(df["Impressions"] > 0, (df["Pin clicks"] / df["Impressions"]) * 100, 0.0)
        else:
            df["CTR_pct"] = 0.0

    # Ensure temporal features exist if Date present
    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = dt.dt.month.fillna(0).astype(int)
        df["DayOfWeek"] = dt.dt.dayofweek.fillna(0).astype(int)
    else:
        for c in ["Month", "DayOfWeek"]:
            if c not in df.columns:
                df[c] = 0

    # One-hot for Campaign status to match training columns
    if "Campaign status" in df.columns:
        status_dummies = pd.get_dummies(df["Campaign status"], prefix="Status")
    else:
        status_dummies = pd.DataFrame()

    # Make sure all expected one-hot columns exist
    for col in one_hot_features:
        if col not in status_dummies.columns:
            status_dummies[col] = 0
    status_dummies = status_dummies[one_hot_features] if one_hot_features else pd.DataFrame()

    # Numeric slice (fill missing with 0)
    X_num = df.reindex(columns=numeric_features, fill_value=0).astype(float)

    # Final feature matrix in the same order as training
    X = pd.concat([X_num, status_dummies], axis=1)
    X = X.reindex(columns=feature_order, fill_value=0)
    return X

def main():
    ap = argparse.ArgumentParser(description="Score Pinterest campaigns as WIN probability.")
    ap.add_argument("--model", default="pinterest_baseline_logreg.pkl", help="Path to model .pkl")
    ap.add_argument("--in", dest="inp", required=True, help="Path to input CSV")
    ap.add_argument("--out", dest="out", default="predictions.csv", help="Path to output CSV")
    ap.add_argument("--threshold", type=float, default=0.5, help="WIN classification threshold")
    args = ap.parse_args()

    numeric_features, one_hot_features, feature_order, scaler, model = load_artifacts(args.model)

    df = pd.read_csv(args.inp)
    X = prepare_features(df.copy(), numeric_features, one_hot_features, feature_order)

    # Scale numeric block only
    num_count = len(numeric_features)
    X_num = scaler.transform(X[numeric_features])
    X_oh = X.drop(columns=numeric_features).to_numpy()
    X_final = np.hstack([X_num, X_oh])

    probs = model.predict_proba(X_final)[:, 1]
    preds = (probs >= args.threshold).astype(int)

    out_df = df.copy()
    out_df["win_probability"] = probs
    out_df["predicted_label"] = np.where(preds == 1, "WINNER", "NOT_WINNER")

    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows → {args.out}")

if __name__ == "__main__":
    main()
