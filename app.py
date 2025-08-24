
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import requests
import smtplib
from email.message import EmailMessage

st.set_page_config(page_title="Production EWS v3 (Forecast + Notifications)", layout="wide")

# ------------------------------
# Helpers
# ------------------------------
def parse_datetime(col):
    return pd.to_datetime(col, errors="coerce")

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
    drawdown = (pr - pwf)
    with np.errstate(divide='ignore', invalid='ignore'):
        pi = qo / drawdown.replace(0, np.nan)
    return pi

def map_recommendations(row):
    recs = []
    if row.get("WCUT_z", 0) > 2 or row.get("WCUT_trend_7d", 0) > 0.1:
        recs.append("WCUT rising: investigate water source. Reduce drawdown (-5–10% choke).")
    if row.get("PI_z", 0) < -2 or row.get("PI_7d_drop", 0) < -0.15:
        recs.append("PI dropping: check for skin/scale. Consider well test and chemical treatment.")
    if (row.get("PWF_z", 0) > 2 or row.get("WHP_z", 0) > 2) and row.get("Qo_z", 0) <= 0:
        recs.append("Possible restriction. Verify lift & surface equipment.")
    if row.get("GOR_z", 0) > 2 or row.get("GOR_7d_rise", 0) > 0.15:
        recs.append("GOR rising: gas coning risk. Reduce drawdown and monitor.")
    if not recs:
        recs.append("System stable. Routine checks ok.")
    return " • ".join(recs)

def grade_health(row):
    score = 100
    penal = 0
    penal += 10 if row.get("WCUT_z",0) > 2 else 0
    penal += 15 if row.get("PI_z",0) < -2 else 0
    penal += 10 if row.get("PWF_z",0) > 2 else 0
    penal += 10 if row.get("GOR_z",0) > 2 else 0
    penal += 5 if row.get("Qo_z",0) < -2 else 0
    return max(0, score - penal)

def infer_columns(df):
    map_guess = {}
    candidates = {
        "date": ["date","day","time","timestamp"],
        "qo": ["oil","qo","oil_rate","oil production","stb"],
        "qw": ["water","qw","water_rate"],
        "wcut": ["water cut","wcut","wc"],
        "pr": ["reservoir pressure","pr"],
        "pwf": ["downhole pressure","pwf","bhp"],
        "whp": ["whp","wellhead pressure"],
        "qg": ["gas","qg"],
        "gor": ["gor","gas oil ratio"]
    }
    for key, keys in candidates.items():
        for k in keys:
            hits = [c for c in df.columns if k in c.lower()]
            if hits:
                map_guess[key] = hits[0]
                break
    return map_guess

# Notifications
def send_telegram(bot_token, chat_id, text):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": text})
        return r.status_code == 200
    except Exception as e:
        return False

def send_email(smtp_server, smtp_port, username, password, sender, recipient, subject, body):
    try:
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as s:
            s.starttls()
            s.login(username, password)
            s.send_message(msg)
        return True
    except Exception as e:
        return False

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.title("Controls")
st.sidebar.caption("Detection, Forecast & Notifications")

win = st.sidebar.slider("Rolling window (days)", 7, 60, 21, step=1)
z_thr = st.sidebar.slider("Anomaly z-score threshold", 1.0, 4.0, 2.0, 0.1)
cusum_k = st.sidebar.slider("CUSUM k", 0.1, 2.0, 0.5, 0.1)
cusum_h = st.sidebar.slider("CUSUM h", 2, 20, 6, 1)

# Forecasting controls
st.sidebar.markdown("### Forecasting")
forecast_days = st.sidebar.number_input("Forecast horizon (days)", 1, 90, 7, 1)
forecast_train_days = st.sidebar.number_input("Train on last N days", 7, 3650, 90, 1)
residual_threshold = st.sidebar.slider("Residual threshold (%)", 1.0, 50.0, 10.0, 0.5)

# Notifications
st.sidebar.markdown("### Notifications (optional)")
enable_telegram = st.sidebar.checkbox("Enable Telegram notifications", value=False)
tg_bot = st.sidebar.text_input("Telegram bot token", type="password")
tg_chat_id = st.sidebar.text_input("Telegram chat id")
enable_email = st.sidebar.checkbox("Enable Email notifications", value=False)
smtp_server = st.sidebar.text_input("SMTP server (e.g., smtp.gmail.com)")
smtp_port = st.sidebar.number_input("SMTP port", 1, 65535, 587)
smtp_user = st.sidebar.text_input("SMTP username (email)")
smtp_pass = st.sidebar.text_input("SMTP password", type="password")
email_sender = st.sidebar.text_input("Sender email")
email_recipient = st.sidebar.text_input("Recipient email")

st.title("⛽ Production Early Warning System v3")
st.write("Adds simple ML forecast (linear regression) + optional Telegram/Email notifications for alerts.")

# Data input
upload = st.file_uploader("Upload production CSV/Excel (date, rates, pressures...)", type=["csv","xlsx"])
if upload is None:
    st.info("Upload your production file to start analysis.")
    st.stop()

try:
    if upload.name.lower().endswith(".csv"):
        df = pd.read_csv(upload)
    else:
        df = pd.read_excel(upload)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

if df is None or len(df) == 0:
    st.error("Empty file.")
    st.stop()

colmap_guess = infer_columns(df)
with st.expander("↔ Map columns", expanded=True):
    cols = {k: st.selectbox(f"{k.upper()} column", ["<none>"] + list(df.columns), index=(["<none>"]+list(df.columns)).index(colmap_guess.get(k,"<none>"))) for k in ["date","qo","qw","wcut","pr","pwf","whp","qg","gor"]}

if cols["date"] == "<none>":
    st.error("Please select a date/time column.")
    st.stop()

df[cols["date"]] = parse_datetime(df[cols["date"]])
df = df.sort_values(cols["date"]).reset_index(drop=True)
df = df.dropna(subset=[cols["date"]])
df = df.rename(columns={cols["date"]: "DATE"})

k = {k:v for k,v in cols.items() if v != "<none>"}
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
    out[f"{col}_z"] = rolling_zscore(out[col], window=win)
    out[f"{col}_ewma"] = ewma_signal(out[col], span=win)
out["WCUT_trend_7d"] = pct_change(out["WCUT"], periods=7)
out["PI_7d_drop"] = pct_change(out["PI"], periods=7)
out["GOR_7d_rise"] = pct_change(out["GOR"], periods=7)

for col in ["Qo","WCUT","PI","GOR","PWF","WHP"]:
    try:
        cp = cusum_detect(out[col], k=cusum_k, h=cusum_h)
    except Exception:
        cp = []
    out[f"{col}_cp"] = 0
    if len(cp):
        out.loc[out.index.isin(cp), f"{col}_cp"] = 1

out["HEALTH"] = out.apply(grade_health, axis=1)
out["RECOMMENDATION"] = out.apply(map_recommendations, axis=1)

# KPI PANEL
st.subheader("📊 Production KPIs")
with st.container():
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Oil Rate (stb/d)", f"{out['Qo'].mean():.1f}" if out['Qo'].notna().any() else "n/a")
    c2.metric("Cumulative Oil (stb)", f"{out['Qo'].sum():.0f}" if out['Qo'].notna().any() else "n/a")
    c3.metric("Avg Water Cut (%)", f"{out['WCUT'].mean():.1f}" if out['WCUT'].notna().any() else "n/a")
    c4, c5, c6 = st.columns(3)
    c4.metric("Avg PI", f"{out['PI'].mean():.3f}" if out['PI'].notna().any() else "n/a")
    c5.metric("Avg GOR (scf/stb)", f"{out['GOR'].mean():.1f}" if out['GOR'].notna().any() else "n/a")
    alert_mask = (
        (out["WCUT_z"].abs() > z_thr) |
        (out["PI_z"].abs() > z_thr) |
        (out["GOR_z"].abs() > z_thr) |
        (out["Qo_z"].abs() > z_thr) |
        (out[[c for c in out.columns if c.endswith("_cp")]].sum(axis=1) > 0)
    )
    c6.metric("Anomaly-free days (%)", f"{(100.0*(len(out)-int(alert_mask.sum()))/len(out)):.1f}" if len(out) else "n/a")

# System status latest
st.subheader("System Status (Latest)")
last = out.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Health (0-100)", f"{int(last['HEALTH'])}")
col2.metric("Qo (last)", f"{last['Qo']:.2f}" if pd.notnull(last['Qo']) else "n/a")
col3.metric("WCUT % (last)", f"{last['WCUT']:.2f}" if pd.notnull(last['WCUT']) else "n/a")
col4.metric("PI (last)", f"{last['PI']:.4f}" if pd.notnull(last['PI']) else "n/a")
st.success(last["RECOMMENDATION"])

# Trends
st.subheader("Trends")
show_cols = st.multiselect("Select series to plot", ["Qo","WCUT","PI","PR","PWF","WHP","GOR"], default=["Qo","WCUT","PI"])
plot_df = out[["DATE"] + [c for c in show_cols] + [f"{c}_ewma" for c in show_cols if f"{c}_ewma" in out.columns]].set_index("DATE")
st.line_chart(plot_df)

# Early Warning Charts
st.subheader("⚠️ Early Warning Charts")
vars_to_plot = ["Qo","WCUT","PI","GOR","PWF","WHP"]
for var in vars_to_plot:
    if var not in out.columns or out[var].isna().all():
        continue
    fig, ax = plt.subplots(figsize=(9,3.8))
    ax.plot(out["DATE"], out[var], label=var)
    if f"{var}_ewma" in out.columns:
        ax.plot(out["DATE"], out[f"{var}_ewma"], linestyle="--", label=f"{var} EWMA")
    zcol = f"{var}_z"
    cpcol = f"{var}_cp"
    anomaly_idx = pd.Series(False, index=out.index)
    if zcol in out.columns:
        anomaly_idx |= (out[zcol].abs() > z_thr)
        ax2 = ax.twinx()
        ax2.plot(out["DATE"], out[zcol], alpha=0.2, label=f"{var} z")
        ax2.axhline(z_thr, linestyle="--", alpha=0.3)
        ax2.axhline(-z_thr, linestyle="--", alpha=0.3)
        ax2.set_ylabel("z-score")
    if cpcol in out.columns:
        anomaly_idx |= (out[cpcol] == 1)
    ax.scatter(out["DATE"][anomaly_idx], out[var][anomaly_idx], label="Anomaly", zorder=5)
    ax.set_title(f"Early Warning – {var}")
    ax.legend(loc="upper left")
    st.pyplot(fig)

# Simple ML forecaster (linear regression on date ordinal)
st.subheader("🔮 Forecast & Residual Monitoring (Qo)")
if out["Qo"].notna().sum() < 10:
    st.info("Need at least 10 Qo points to build a meaningful forecast. Skipping forecast.")
else:
    # Prepare training data
    df_fore = out.dropna(subset=["Qo"]).copy().reset_index(drop=True)
    df_fore["date_ordinal"] = df_fore["DATE"].map(lambda d: d.toordinal())
    train = df_fore.tail(int(min(len(df_fore), forecast_train_days)))
    X = train["date_ordinal"].values.reshape(-1,1)
    y = train["Qo"].values
    model = LinearRegression().fit(X, y)
    last_date = df_fore["DATE"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]
    Xf = np.array([d.toordinal() for d in future_dates]).reshape(-1,1)
    y_pred = model.predict(Xf)
    # Attach prediction to a frame
    pred_df = pd.DataFrame({"DATE": future_dates, "Qo_pred": y_pred})
    # Plot
    fig, ax = plt.subplots(figsize=(9,3.8))
    ax.plot(out["DATE"], out["Qo"], label="Qo")
    ax.plot(pred_df["DATE"], pred_df["Qo_pred"], label="Qo forecast", linestyle="--")
    ax.set_title("Qo Forecast (linear)")
    ax.legend()
    st.pyplot(fig)
    # residual monitoring on holdout: compute residuals on last train window
    train_pred = model.predict(X)
    residuals = y - train_pred
    resid_pct = (residuals / (train_pred + 1e-9)) * 100.0
    # Latest residual percent
    latest_resid_pct = resid_pct[-1] if len(resid_pct) else 0.0
    st.write(f"Latest forecast residual (percent) on training tail: {latest_resid_pct:.2f}%")
    if abs(latest_resid_pct) > residual_threshold:
        st.error("Forecast residual exceeds threshold — predictive alert: system deviating from expected trend.")
        forecast_alert = True
    else:
        st.success("Forecast residual within threshold.")
        forecast_alert = False

# Alerts table and auto-notify
st.subheader("Alerts & Notifications")
alerts = out.loc[alert_mask, ["DATE","Qo","WCUT","PI","GOR","WHP","RECOMMENDATION","HEALTH"]].tail(200)
st.dataframe(alerts, use_container_width=True)
st.write(f"Total alerts found: {int(alert_mask.sum())}")

auto_notify = st.checkbox("Auto-send notification on alerts (this session)")
if auto_notify and (alert_mask.any() or ( 'forecast_alert' in locals() and forecast_alert )):
    summary = f"Production EWS Alert - {datetime.utcnow().isoformat()}\\nAlerts: {int(alert_mask.sum())}\\nLatest rec: {last['RECOMMENDATION']}"
    sent_msgs = []
    if enable_telegram and tg_bot and tg_chat_id:
        ok = send_telegram(tg_bot, tg_chat_id, summary)
        sent_msgs.append(f"Telegram: {'OK' if ok else 'FAILED'}")
    if enable_email and smtp_server and smtp_user and smtp_pass and email_recipient and email_sender:
        ok = send_email(smtp_server, smtp_port, smtp_user, smtp_pass, email_sender, email_recipient, "EWS Alert", summary)
        sent_msgs.append(f"Email: {'OK' if ok else 'FAILED'}")
    st.write("Notification results:", ", ".join(sent_msgs) if sent_msgs else "No methods configured or failed.")

# Download analysis
buf = BytesIO()
out.to_csv(buf, index=False)
st.download_button("Download analysis CSV", buf.getvalue(), file_name="ews_output_v3.csv", mime="text/csv")
