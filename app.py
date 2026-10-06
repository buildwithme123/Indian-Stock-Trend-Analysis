"""
Indian Markets: Risk Analysis + ML Direction Prediction
--------------------------------------------------------
Streamlit web app version of the notebook project. Same corrected logic:
  - chronological train/test split (no shuffling time series)
  - features are stationary (returns, MA gap ratio, volatility) - never a raw price level
  - target is TOMORROW's direction, known features only up to today (no lookahead)
  - buy & hold = invested every day, unconditionally (no column decides it)
  - backtest compounds off the portfolio's own value, with a transaction cost per trade
  - VaR/CVaR via historical and parametric methods, across all 5 assets

Run locally:   streamlit run app.py
Deploy free:   push this repo to GitHub, then deploy at https://share.streamlit.io
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

st.set_page_config(page_title="Indian Markets: Risk & ML Strategy", layout="wide")

TICKERS = {
    "SBI": "SBIN.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
}
FEATURES = ["return_1day", "return_5day", "ma_gap", "volatility"]

# ------------------------------------------------------------------
# Data loading (cached so the app doesn't re-download on every click)
# ------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Downloading prices from Yahoo Finance...")
def load_prices(start: str, end: str | None):
    def one(symbol):
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df["Close"].dropna()
    return pd.DataFrame({name: one(sym) for name, sym in TICKERS.items()}).dropna()


def make_features(close: pd.Series) -> pd.DataFrame:
    feat = pd.DataFrame(index=close.index)
    feat["return_1day"] = close.pct_change()
    feat["return_5day"] = close.pct_change(5)
    feat["ma_gap"] = close.rolling(10).mean() / close.rolling(50).mean() - 1
    feat["volatility"] = feat["return_1day"].rolling(10).std()
    feat["target"] = (close.shift(-1) > close).astype(int)   # tomorrow's direction
    feat["fwd_return"] = close.shift(-1) / close - 1           # tomorrow's actual return
    # NOTE: "fwd_return" must be in this subset too. The very last trading day has no
    # real "tomorrow", so shift(-1) leaves it as NaN - but `NaN > close` silently evaluates
    # to False in pandas, so `target` becomes 0 (not NaN) for that row and dropna(subset=
    # ["target"]) alone never catches it. Without "fwd_return" here, that row survives with
    # fwd_return = NaN, which can end up in the test set and quietly break cumprod() in the
    # backtest (a single NaN poisons every value after it).
    return feat.dropna(subset=FEATURES + ["target", "fwd_return"]).copy()


def historical_var(r, conf):
    return -np.percentile(r, (1 - conf) * 100)

def parametric_var(r, conf):
    return -(r.mean() + stats.norm.ppf(1 - conf) * r.std())

def cvar(r, conf):
    v = historical_var(r, conf)
    return -r[r <= -v].mean()


def summarize(r: pd.Series, risk_free: float) -> pd.Series:
    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else np.nan
    equity = (1 + r).cumprod()
    mdd = (equity / equity.cummax() - 1).min()
    return pd.Series({"Annual Return": ann_ret, "Annual Volatility": ann_vol,
                       "Sharpe": sharpe, "Max Drawdown": mdd})


# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
st.sidebar.header("Settings")
asset = st.sidebar.selectbox("Asset to model", list(TICKERS.keys()), index=4)
start_date = st.sidebar.date_input("Start date", value=pd.Timestamp("2015-01-01"))
test_frac = st.sidebar.slider("Test set size (chronological, last N%)", 0.1, 0.4, 0.2, 0.05)
cost_bps = st.sidebar.slider("Transaction cost per trade (bps)", 0, 20, 10)
risk_free = st.sidebar.slider("Risk-free rate (annual, for Sharpe)", 0.0, 0.10, 0.065, 0.005)
cost = cost_bps / 10000

st.title("Indian Markets: Risk Analysis & ML Direction Prediction")
st.caption("SBI · HDFC Bank · TCS · Nifty 50 · Sensex  —  real NSE/BSE data via Yahoo Finance")

try:
    prices = load_prices(str(start_date), None)
except Exception as e:
    st.error(f"Couldn't download data: {e}")
    st.stop()

st.success(f"Loaded {len(prices)} trading days, {prices.index.min().date()} to {prices.index.max().date()}")

# ------------------------------------------------------------------
# Section 1: Price overview
# ------------------------------------------------------------------
st.header("1. Price Overview")
st.line_chart((prices / prices.iloc[0]), height=350)
st.caption("Growth of Rs 1 invested at the start, for all five assets.")

# ------------------------------------------------------------------
# Section 2: Risk metrics across all assets
# ------------------------------------------------------------------
st.header("2. Risk Analysis (all 5 assets)")
returns_all = prices.pct_change().dropna()

risk_rows = []
for name in returns_all.columns:
    r = returns_all[name]
    equity = (1 + r).cumprod()
    risk_rows.append({
        "Asset": name,
        "Annual Volatility": r.std() * np.sqrt(252),
        "Max Drawdown": (equity / equity.cummax() - 1).min(),
        "Historical VaR 99%": historical_var(r, 0.99),
        "Parametric VaR 99%": parametric_var(r, 0.99),
        "CVaR 99%": cvar(r, 0.99),
    })
risk_table = pd.DataFrame(risk_rows).set_index("Asset")
st.dataframe(risk_table.style.format("{:.4f}"))

investment = st.number_input("Show 1-day 99% historical VaR for an investment of (Rs)", value=1_000_000, step=100_000)
var_rupees = pd.DataFrame({
    "1-day 99% Historical VaR (Rs)": {name: historical_var(returns_all[name], 0.99) * investment for name in returns_all.columns}
})
st.dataframe(var_rupees.style.format("{:,.0f}"))

# ------------------------------------------------------------------
# Section 3: ML direction prediction
# ------------------------------------------------------------------
st.header(f"3. Direction Prediction — {asset}")

close = prices[asset]
data = make_features(close)

X, y = data[FEATURES], data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, shuffle=False)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

majority_class = y_train.mode()[0]
baseline_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))

col1, col2 = st.columns(2)
models = {}
for col, (name, clf) in zip(
    [col1, col2],
    [("Logistic Regression", LogisticRegression(random_state=0)),
     ("Random Forest", RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=30,
                                               class_weight="balanced", random_state=0, n_jobs=-1))]
):
    clf.fit(X_train_sc, y_train)
    pred = clf.predict(X_test_sc)
    models[name] = pred

    with col:
        st.subheader(name)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc:.3f}", delta=f"{acc - baseline_acc:+.3f} vs baseline")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")

        cm = confusion_matrix(y_test, pred)
        st.write("Confusion matrix (rows=actual Down/Up, cols=predicted Down/Up):")
        st.dataframe(pd.DataFrame(cm, index=["Actual Down", "Actual Up"], columns=["Pred Down", "Pred Up"]))

st.info(f"Baseline accuracy (always guess the majority class seen in training): **{baseline_acc:.3f}**")

# ------------------------------------------------------------------
# Section 4: Backtest
# ------------------------------------------------------------------
st.header("4. Strategy Backtest (out-of-sample test period)")

fwd_ret = data.loc[y_test.index, "fwd_return"]
initial_cash = 100_000

strategy_returns = {"Buy & Hold": fwd_ret}  # always invested, unconditionally - no column decides it
for name, pred in models.items():
    position = pd.Series(pred, index=y_test.index)
    trades = position.diff().abs().fillna(1)          # a trade happens on day 1 and whenever position flips
    strategy_returns[name] = position * fwd_ret - trades * cost

equity_curves = pd.DataFrame({
    name: initial_cash * (1 + r).cumprod() / initial_cash for name, r in strategy_returns.items()
})
st.line_chart(equity_curves, height=350)
st.caption("Growth of Rs 1 on the held-out test period, after transaction costs.")

perf_table = pd.DataFrame({name: summarize(r, risk_free) for name, r in strategy_returns.items()})
st.dataframe(perf_table.style.format("{:.3f}"))

st.markdown("""
**Reading this honestly:** if an ML strategy's Sharpe ratio is lower than Buy & Hold's despite
having lower volatility, that's not a bug — when excess returns are negative, dividing by a
*smaller* volatility makes the Sharpe ratio *more* negative, not less. Low volatility only
improves Sharpe when returns are positive.
""")

st.markdown("---")
st.caption(
    "Educational project, not investment advice. Past performance does not guarantee future "
    "results. Built with a chronological train/test split, no lookahead bias, and transaction "
    "costs included in the backtest."
)
