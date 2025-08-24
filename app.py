pip install openpyxl
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta

st.set_page_config(page_title="Production Early Warning System", layout="wide")

# ------------------------------
# Helpers
# ------------------------------
def parse_datetime(col):
    try:
        return pd.to_datetime(col, errors="coerce")
    except Exception:
        return pd.to_datetime(col, errors="coerce", dayfirst=True)

def rolling_zscore(series, window=14, eps=1e-9):
    r = series.rolling(window)
    mu = r.mean()
    sd = r.std().replace(0, eps)
    return (series - mu) / sd

def ewma_signal(series, span=20):
    return series.ewm(span=span, adjust=False).mean()

def pct_change(series, periods=1):
    return series.pct_change(periods=periods)

def cusum_detect(series, k=0.5, h=5):
    """ Simple two-sided CUSUM returning indices of change points (approximate) """
    s_pos = np.zeros(len(series))
    s_neg = np.zeros(len(series))
    idx = []
    x = series.fillna(method="ffill").fillna(method="bfill").values
    if len(x) == 0:
        return idx
    mean = np.nanmean(x)
    for i in range(1, len(x)):
        s_pos[i] = max(0, s_pos[i-1] + (x[i] - mean - k))
        s_neg[i] = min(0, s_neg[i-1] + (x[i] - mean + k))
        if s_pos[i] > h or s_neg[i] < -h:
            idx.append(i)
            s_pos[i] = 0
            s_neg[i] = 0
    return idx

def compute_productivity_index(qo, pwf, pr):
    """ PI = Qo / (Pr - Pwf). Inputs may be arrays/Series. """
    drawdown = (pr - pwf)
    with np.errstate(divide='ignore', invalid='ignore'):
        pi = qo / drawdown.replace(0, np.nan)
    return pi

def map_recommendations(row):
    recs = []
    # Water cut rising fast
    if row.get("WCUT_z", 0) > 2 or row.get("WCUT_trend_7d", 0) > 0.1:
        recs.append("Investigate water source (coning/channeling). Run water shutoff diagnostics; consider choke reduction 5–10% and zonal isolation checks.")
    # PI dropping
    if row.get("PI_z", 0) < -2 or row.get("PI_7d_drop", 0) < -0.15:
        recs.append("Possible skin/restriction. Schedule well test, check for scale/asphaltene; consider acid wash or surfactant flush.")
    # DHP rising (for same rate) suggests restriction
    if row.get("DHP_z", 0) > 2 and row.get("Qo_z", 0) <= 0:
        recs.append("Lift performance degradation suspected. Verify artificial lift, check separator/choke erosion; adjust choke by -3% and re-evaluate.")
    # GOR surge
    if row.get("GOR_z", 0) > 2 or row.get("GOR_7d_rise", 0) > 0.15:
        recs.append("Gas coning risk. Reduce drawdown (increase WHP or reduce choke) temporarily and monitor.")
    # Stable/healthy
    if not recs:
        recs.append("System stable. Maintain settings; schedule routine well test/PM.")
    return " • ".join(recs)

def grade_health(row):
    score = 100
    # Penalize signals
    penal = 0
    penal += 10 if row.get("WCUT_z",0) > 2 else 0
    penal += 15 if row.get("PI_z",0) < -2 else 0
    penal += 10 if row.get("DHP_z",0) > 2 else 0
    penal += 10 if row.get("GOR_z",0) > 2 else 0
    penal += 5 if row.get("Qo_z",0) < -2 else 0
    return max(0, score - penal)

def infer_columns(df):
    # Try to guess columns by common names
    cols = df.columns.str.lower()
    map_guess = {}
    candidates = {
        "date": ["date","day","time","timestamp"],
        "qo": ["oil", "oil_rate","oil production","qo","q_o","stb","stb/day"],
        "qw": ["water","water_rate","qw"],
        "wcut": ["water cut","water_cut","wcut","wc"],
        "pr": ["reservoir pressure","pr","avg reservoir pressure"],
        "pwf": ["downhole pressure","bhp","pwf","bottomhole pressure","flowing bhp"],
        "whp": ["whp","wellhead pressure"],
        "qg": ["gas","gas_rate","qg","scf","gor*oil"], # loose
        "gor": ["gor","gas oil ratio"]
    }
    for key, keys in candidates.items():
        for k in keys:
            hit = [c for c in df.columns if k in c.lower()]
            if hit:
                map_guess[key] = hit[0]
                break
    return map_guess

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.title("Controls")
st.sidebar.caption("Configure detection thresholds")
win = st.sidebar.slider("Rolling window (days)", 7, 60, 21, step=1)
z_thr = st.sidebar.slider("Anomaly z-score threshold", 1.5, 4.0, 2.0, 0.1)
cusum_k = st.sidebar.slider("CUSUM k", 0.1, 2.0, 0.5, 0.1)
cusum_h = st.sidebar.slider("CUSUM h", 2, 20, 6, 1)

st.title("⛽ Production Early Warning System")
st.write("Monitors key indicators (Qo, WCUT, PI, pressures, GOR) to flag early risks and suggest interventions.")

# Data input
upload = st.file_uploader("Upload production CSV/Excel (date, rates, pressures...)", type=["csv","xlsx"])
if upload is None:
    st.info("You can upload your file above. If you already placed a file in the app directory, you can proceed after mapping columns.")
    df = None
else:
    try:
        if upload.name.lower().endswith(".csv"):
            df = pd.read_csv(upload)
        else:
            df = pd.read_excel(upload)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        df = None

if df is not None and len(df):
    # Try to parse date
    colmap = infer_columns(df)
    with st.expander("↔ Map columns", expanded=False):
        cols = {k: st.selectbox(f"{k.upper()} column", ["<none>"] + list(df.columns), index=(["<none>"]+list(df.columns)).index(colmap.get(k,"<none>"))) for k in ["date","qo","qw","wcut","pr","pwf","whp","qg","gor"]}
    if cols["date"] != "<none>":
        df[cols["date"]] = parse_datetime(df[cols["date"]])
        df = df.sort_values(cols["date"]).reset_index(drop=True)
        df = df.dropna(subset=[cols["date"]])
        df = df.rename(columns={cols["date"]: "DATE"})
    else:
        st.error("Please select a date/time column.")
        st.stop()

    # Normalize and fill
    k = {k:v for k,v in cols.items() if v != "<none>"}
    # Build derived features
    Qo = df[k.get("qo")].astype(float) if "qo" in k else pd.Series(index=df.index, dtype="float64")
    Qw = df[k.get("qw")].astype(float) if "qw" in k else pd.Series(index=df.index, dtype="float64")
    WC = df[k.get("wcut")].astype(float) if "wcut" in k else (Qw / (Qw + Qo).replace(0, np.nan) * 100.0)
    PR = df[k.get("pr")].astype(float) if "pr" in k else pd.Series(index=df.index, dtype="float64")
    PWF = df[k.get("pwf")].astype(float) if "pwf" in k else pd.Series(index=df.index, dtype="float64")
    WHP = df[k.get("whp")].astype(float) if "whp" in k else pd.Series(index=df.index, dtype="float64")
    Qg = df[k.get("qg")].astype(float) if "qg" in k else pd.Series(index=df.index, dtype="float64")
    GOR = df[k.get("gor")].astype(float) if "gor" in k else (Qg / Qo.replace(0, np.nan))

    PI = compute_productivity_index(Qo, PWF, PR)

    out = pd.DataFrame({
        "DATE": df["DATE"],
        "Qo": Qo,
        "Qw": Qw,
        "WCUT": WC,
        "PR": PR,
        "PWF": PWF,
        "WHP": WHP,
        "Qgas": Qg,
        "GOR": GOR,
        "PI": PI,
    })

    # Signals
    for col in ["Qo","WCUT","PR","PWF","WHP","GOR","PI"]:
        if col in out.columns:
            out[f"{col}_z"] = rolling_zscore(out[col], window=win)
            out[f"{col}_ewma"] = ewma_signal(out[col], span=win)
    out["WCUT_trend_7d"] = pct_change(out["WCUT"], periods=7)
    out["PI_7d_drop"] = pct_change(out["PI"], periods=7)
    out["GOR_7d_rise"] = pct_change(out["GOR"], periods=7)
    # CUSUM change points
    for col in ["Qo","WCUT","PI","GOR","PWF","WHP"]:
        try:
            cp = cusum_detect(out[col], k=cusum_k, h=cusum_h)
        except Exception:
            cp = []
        out[f"{col}_cp"] = 0
        out.loc[out.index.isin(cp), f"{col}_cp"] = 1

    # Health & recs
    out["HEALTH"] = out.apply(grade_health, axis=1)
    out["RECOMMENDATION"] = out.apply(map_recommendations, axis=1)

    st.subheader("System Status")
    last = out.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Health (0-100)", f"{int(last['HEALTH'])}")
    col2.metric("Qo (last)", f"{last['Qo']:.2f}" if pd.notnull(last['Qo']) else "n/a")
    col3.metric("WCUT % (last)", f"{last['WCUT']:.2f}" if pd.notnull(last['WCUT']) else "n/a")
    col4.metric("PI (last)", f"{last['PI']:.4f}" if pd.notnull(last['PI']) else "n/a")
    st.success(last["RECOMMENDATION"])

    # Plots (use built-in Streamlit charts to avoid extra deps)
    st.subheader("Trends")
    show_cols = st.multiselect("Select series to plot", ["Qo","WCUT","PI","PR","PWF","WHP","GOR"], default=["Qo","WCUT","PI"])
    plot_df = out[["DATE"] + [c for c in show_cols] + [f"{c}_ewma" for c in show_cols if f"{c}_ewma" in out.columns]].set_index("DATE")
    st.line_chart(plot_df)

    # Alerts table
    st.subheader("Alerts")
    alert_mask = (
        (out["WCUT_z"] > z_thr) |
        (out["PI_z"] < -z_thr) |
        (out["DHP_z"] > z_thr if "DHP_z" in out.columns else False) |
        (out["GOR_z"] > z_thr) |
        (out["Qo_z"] < -z_thr) |
        (out[[c for c in out.columns if c.endswith("_cp")]].sum(axis=1) > 0)
    )
    alerts = out.loc[alert_mask, ["DATE","Qo","WCUT","PI","GOR","WHP","RECOMMENDATION","HEALTH"]].tail(50)
    st.dataframe(alerts, use_container_width=True)

    # Download results
    buf = BytesIO()
    out.to_csv(buf, index=False)
    st.download_button("Download analysis CSV", buf.getvalue(), file_name="ews_output.csv", mime="text/csv")

else:
    st.stop()
