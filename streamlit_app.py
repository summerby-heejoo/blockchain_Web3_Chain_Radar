# === 셀 2: 앱 파일 생성 (app.py) ===
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

st.set_page_config(page_title="Crypto Backtest + Fear & Greed", layout="wide")
st.title("📈 Crypto Market Analysis: Fear & Greed + Backtest + Comparison")

# =============================================================
# 1. 데이터 로딩 함수
# =============================================================

@st.cache_data
def load_fng():
    url = "https://api.alternative.me/fng/?limit=90&format=json"
    r = requests.get(url).json()
    df = pd.DataFrame(r["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["fear"] = df["value"].astype(int)
    return df[["timestamp", "fear"]]

@st.cache_data
def load_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": 90}
    r = requests.get(url, params=params).json()
    df = pd.DataFrame(r["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["asset"] = coin_id.upper()
    return df

# Load all data
fng = load_fng()
btc = load_price("bitcoin")
eth = load_price("ethereum")
sol = load_price("solana")

# =============================================================
# 2. 병합 함수
# =============================================================
def merge_data(price_df):
    df = pd.merge_asof(
        price_df.sort_values("timestamp"),
        fng.sort_values("timestamp"),
        on="timestamp",
        direction="nearest"
    )
    return df

btc_m = merge_data(btc)
eth_m = merge_data(eth)
sol_m = merge_data(sol)

# =============================================================
# 3. 기본 시각화
# =============================================================
st.header("📊 90일 가격 비교 (BTC / ETH / SOL)")

price_all = pd.concat([btc_m, eth_m, sol_m])
fig_price = px.line(price_all, x="timestamp", y="price", color="asset")
st.plotly_chart(fig_price, use_container_width=True)

# =============================================================
# 4. 상관계수
# =============================================================
st.header("🔗 Fear & Greed 상관계수 (BTC, ETH, SOL)")

corr_btc = btc_m["fear"].corr(btc_m["price"])
corr_eth = eth_m["fear"].corr(eth_m["price"])
corr_sol = sol_m["fear"].corr(sol_m["price"])

col1, col2, col3 = st.columns(3)
col1.metric("BTC 상관계수", f"{corr_btc:.3f}")
col2.metric("ETH 상관계수", f"{corr_eth:.3f}")
col3.metric("SOL 상관계수", f"{corr_sol:.3f}")

# =============================================================
# 5. 백테스트 함수
# =============================================================
def backtest(df, lookahead=30):
    df = df.copy()
    df["future"] = df["price"].shift(-lookahead)
    df["return"] = (df["future"] - df["price"]) / df["price"]
    fear_buy = df[df["fear"] <= 20].dropna()
    greed_sell = df[df["fear"] >= 80].dropna()
    return fear_buy, greed_sell

lookahead = st.slider("백테스트 미래 기간 (일)", 7, 60, 30)

btc_fear, btc_greed = backtest(btc_m, lookahead)
eth_fear, eth_greed = backtest(eth_m, lookahead)
sol_fear, sol_greed = backtest(sol_m, lookahead)

# =============================================================
# 6. 전략별 누적 수익률
# =============================================================
def cumulative_return(df):
    df = df.copy()
    df["cumulative"] = (1 + df["return"]).cumprod()
    return df

btc_fear_cum = cumulative_return(btc_fear)
eth_fear_cum = cumulative_return(eth_fear)
sol_fear_cum = cumulative_return(sol_fear)

st.header("📈 전략별 누적 수익률 (Extreme Fear 매수 전략)")

fear_all = pd.concat([
    btc_fear_cum.assign(asset="BTC"),
    eth_fear_cum.assign(asset="ETH"),
    sol_fear_cum.assign(asset="SOL")
])

fig_fear = px.line(fear_all, x="timestamp", y="cumulative", color="asset")
st.plotly_chart(fig_fear, use_container_width=True)

# =============================================================
# 7. 전략 간 비교 (Fear buy vs Greed sell)
# =============================================================
def compare_strategies(fear_df, greed_df, asset):
    fear_total = (1 + fear_df["return"]).prod() - 1 if len(fear_df) > 0 else 0
    greed_total = (1 + greed_df["return"]).prod() - 1 if len(greed_df) > 0 else 0
    return pd.DataFrame({
        "strategy": ["Fear Buy", "Greed Sell"],
        "return": [fear_total * 100, greed_total * 100],
        "asset": asset
    })

compare_df = pd.concat([
    compare_strategies(btc_fear, btc_greed, "BTC"),
    compare_strategies(eth_fear, eth_greed, "ETH"),
    compare_strategies(sol_fear, sol_greed, "SOL")
])

st.header("⚔️ 전략 간 비교 (Fear Buy vs Greed Sell)")

fig_compare = px.bar(compare_df, x="asset", y="return", color="strategy", barmode="group")
st.plotly_chart(fig_compare, use_container_width=True)

# =============================================================
# 8. 수익률 분포 시각화
# =============================================================
st.header("📉 수익률 분포 (Extreme Fear 매수)")

fear_dist = pd.concat([
    btc_fear.assign(asset="BTC"),
    eth_fear.assign(asset="ETH"),
    sol_fear.assign(asset="SOL")
])

fig_dist = px.box(fear_dist, x="asset", y="return", title="Fear Buy 수익률 분포")
st.plotly_chart(fig_dist, use_container_width=True)
